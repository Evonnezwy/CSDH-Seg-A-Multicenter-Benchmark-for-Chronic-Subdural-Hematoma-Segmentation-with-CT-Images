import os
import argparse

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", default="0")
_pre_args, _ = _pre_parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import json
import math
import csv
import random
import re
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import SimpleITK as sitk

from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric

try:
    from monai.metrics import SurfaceDistanceMetric
    HAS_SURFACE_DISTANCE = True
except Exception:
    SurfaceDistanceMetric = None
    HAS_SURFACE_DISTANCE = False
tqdm = None

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from CMA_UGRMSF_ResUNet import UNet3DMMS_CMA_UGRMSF
from dataloader_JSZJ import load_split_json, JSZJValWholeVolumeDataset


SEED = 42
ROI_SIZE = (32, 192, 192)   # z, y, x
SW_BATCH_SIZE = 1
VAL_OVERLAP = 0.25
HD95_SPACING = (5.0, 0.5, 0.5)
PRED_THRESHOLD = 0.5

DEFAULT_VAL_JSON = "./val.json"
DEFAULT_CHECKPOINT = (
    "./bestDice.pth"
)
DEFAULT_SAVE_DIR = "./eval"


# =========================================================
# Helpers
# =========================================================
def make_low_kernels(kernel_sizes):
    return tuple((1, int(k), int(k)) for k in kernel_sizes)


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def autocast_context(device, enabled=True):
    if device.type == "cuda":
        try:
            return torch.amp.autocast(device_type="cuda", enabled=enabled)
        except Exception:
            return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        denominator = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_score.mean()
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_single_sample_from_probs(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs >= threshold).float()
    dims = (1, 2, 3, 4)
    intersection = torch.sum(preds * targets, dims)
    denominator = torch.sum(preds, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.item())


def iou_single_sample_from_probs(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs >= threshold).float()
    inter = torch.sum(preds * targets).item()
    pred_sum = torch.sum(preds).item()
    target_sum = torch.sum(targets).item()
    union = pred_sum + target_sum - inter
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    return float((inter + smooth) / (union + smooth))


def rvd_percent_from_probs(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs >= threshold).float()
    pred_sum = torch.sum(preds).item()
    target_sum = torch.sum(targets).item()
    if target_sum <= 0:
        return 0.0 if pred_sum <= 0 else float("nan")
    return float((pred_sum - target_sum) / (target_sum + smooth) * 100.0)


def get_hd95_penalty(shape_zyx, spacing=None):
    z, y, x = shape_zyx
    if spacing is None:
        sz, sy, sx = 1.0, 1.0, 1.0
    else:
        sz, sy, sx = spacing
    return math.sqrt((z * sz) ** 2 + (y * sy) ** 2 + (x * sx) ** 2)


def hd95_single_sample(pred, target, spacing=None):
    pred = pred.float()
    target = target.float()
    pred_fg = bool(torch.sum(pred).item() > 0)
    target_fg = bool(torch.sum(target).item() > 0)
    penalty = get_hd95_penalty(pred.shape[-3:], spacing=spacing)
    if (not pred_fg) and (not target_fg):
        return 0.0
    if pred_fg != target_fg:
        return float(penalty)
    metric = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="mean")
    metric(y_pred=pred, y=target, spacing=spacing)
    value = metric.aggregate().item()
    metric.reset()
    if value != value:
        return float(penalty)
    return float(value)


def asd_single_sample(pred, target, spacing=None):
    if not HAS_SURFACE_DISTANCE:
        return float("nan")
    pred = pred.float()
    target = target.float()
    pred_fg = bool(torch.sum(pred).item() > 0)
    target_fg = bool(torch.sum(target).item() > 0)
    penalty = get_hd95_penalty(pred.shape[-3:], spacing=spacing)
    if (not pred_fg) and (not target_fg):
        return 0.0
    if pred_fg != target_fg:
        return float(penalty)
    metric = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="mean")
    metric(y_pred=pred, y=target, spacing=spacing)
    value = metric.aggregate().item()
    metric.reset()
    if value != value:
        return float(penalty)
    return float(value)


def strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def parse_epoch_from_checkpoint(ckpt_path, checkpoint):
    if isinstance(checkpoint, dict) and checkpoint.get("epoch", None) is not None:
        try:
            return int(checkpoint["epoch"])
        except Exception:
            pass
    m = re.search(r"epoch(\d+)", Path(ckpt_path).name)
    if m:
        return int(m.group(1))
    return -1


def load_checkpoint_model(ckpt_path, device, args):
    cma_low_kernels = make_low_kernels(args.cma_low_kernel_sizes)
    model = UNet3DMMS_CMA_UGRMSF(
        input_ch=args.input_ch,
        output_ch=args.output_ch,
        init_feats=args.init_feats,
        cma_stages=tuple(args.cma_stages),
        cma_low_kernels=cma_low_kernels,
        cma_init_gamma=args.cma_init_gamma,
        aniso_xy_kernel=args.aniso_xy_kernel,
        aniso_z_kernel=args.aniso_z_kernel,
        use_cgcb=not args.no_cgcb,
        cgcb_reduction=args.cgcb_reduction,
        cgcb_init_beta=args.cgcb_init_beta,
        ugrmsf_stages=tuple(args.ugrmsf_stages),
        ugrmsf_fusion_channels=(args.init_feats if args.ugrmsf_fusion_channels <= 0 else args.ugrmsf_fusion_channels),
        ugrmsf_init_alpha=args.ugrmsf_init_alpha,
        ugrmsf_edge_init_alpha=args.ugrmsf_edge_init_alpha,
        ugrmsf_context_init_alpha=args.ugrmsf_context_init_alpha,
        ugrmsf_use_edge=not args.no_ugrmsf_edge,
        ugrmsf_use_crescent_context=not args.no_ugrmsf_crescent,
        ugrmsf_uncertainty_power=args.ugrmsf_uncertainty_power,
        ugrmsf_uncertainty_detach=not args.ugrmsf_no_detach_uncertainty,
    ).to(device)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))
    strict = not args.non_strict_load
    load_info = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print("[INFO] non-strict loading:", load_info)
    model.eval()
    epoch = parse_epoch_from_checkpoint(ckpt_path, checkpoint)
    best_val_dice = checkpoint.get("best_val_dice", None) if isinstance(checkpoint, dict) else None
    best_val_hd95 = checkpoint.get("best_val_hd95", None) if isinstance(checkpoint, dict) else None
    return model, checkpoint, epoch, best_val_dice, best_val_hd95


def summarize(values):
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan"), "p25": float("nan"), "p75": float("nan"), "p90": float("nan"), "p95": float("nan")}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "median": float(np.median(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)), "p95": float(np.percentile(arr, 95))}


def save_per_case_csv(csv_path, per_case_results):
    fieldnames = ["case_id", "loss", "dice", "iou", "hd95", "asd", "rvd_percent", "pred_positive_voxels", "gt_positive_voxels", "empty_pred", "empty_gt", "pred_nrrd_path", "prob_nrrd_path"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_case_results:
            writer.writerow(row)


def save_summary_json(json_path, summary):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def find_label_reference_path(case_info):
    for key in ["label", "label_path", "seg_path", "mask_path", "seg", "gt_path"]:
        if key in case_info and case_info[key] is not None and str(case_info[key]) != "":
            return case_info[key]
    raise KeyError(f"Cannot find label reference path in item keys: {list(case_info.keys())}")


def save_prediction_nrrd(pred_array_zyx, reference_nrrd_path, save_path):
    ref_img = sitk.ReadImage(str(reference_nrrd_path))
    pred_img = sitk.GetImageFromArray(pred_array_zyx.astype(np.uint8))
    pred_img.CopyInformation(ref_img)
    sitk.WriteImage(pred_img, str(save_path))


def save_probability_nrrd(prob_array_zyx, reference_nrrd_path, save_path):
    ref_img = sitk.ReadImage(str(reference_nrrd_path))
    prob_img = sitk.GetImageFromArray(prob_array_zyx.astype(np.float32))
    prob_img.CopyInformation(ref_img)
    sitk.WriteImage(prob_img, str(save_path))


@torch.no_grad()
def evaluate_and_save_predictions(model, items, loader, criterion, device, save_pred_dir, save_prob_dir=None, spacing=None, threshold=0.5, roi_size=ROI_SIZE, sw_batch_size=SW_BATCH_SIZE, val_overlap=VAL_OVERLAP, use_amp=True, skip_hd95=False, skip_asd=False, empty_cache_after_case=False):
    model.eval()
    per_case_results = []
    item_map = {str(item["case_id"]): item for item in items if "case_id" in item}

    for batch_idx, batch in enumerate(loader, 1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        labels = (labels > 0).float()
        if "case_id" in batch:
            case_id = str(batch["case_id"][0]) if isinstance(batch["case_id"], (list, tuple)) else str(batch["case_id"])
        else:
            case_id = f"case_{batch_idx:05d}"
        case_info = item_map.get(case_id, items[batch_idx - 1])
        label_ref_path = find_label_reference_path(case_info)

        with autocast_context(device, enabled=use_amp):
            logits = sliding_window_inference(inputs=images, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=val_overlap, mode="gaussian")
            loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        dice_value = dice_single_sample_from_probs(probs, labels, threshold=threshold)
        iou_value = iou_single_sample_from_probs(probs, labels, threshold=threshold)
        rvd_value = rvd_percent_from_probs(probs, labels, threshold=threshold)

        pred_cpu = preds.detach().cpu()
        label_cpu = labels.detach().cpu()
        hd95_value = float("nan") if skip_hd95 else hd95_single_sample(pred_cpu, label_cpu, spacing=spacing)
        asd_value = float("nan") if skip_asd else asd_single_sample(pred_cpu, label_cpu, spacing=spacing)

        pred_positive_voxels = int(torch.sum(preds).item())
        gt_positive_voxels = int(torch.sum(labels).item())
        empty_pred = pred_positive_voxels == 0
        empty_gt = gt_positive_voxels == 0

        pred_np = pred_cpu[0, 0].numpy().astype(np.uint8)
        prob_np = probs[0, 0].detach().cpu().numpy().astype(np.float32)

        pred_case_dir = save_pred_dir / case_id
        pred_case_dir.mkdir(parents=True, exist_ok=True)
        pred_nrrd_path = pred_case_dir / f"{case_id}_pred_bleeding.nrrd"
        save_prediction_nrrd(pred_np, label_ref_path, pred_nrrd_path)

        prob_nrrd_path = None
        if save_prob_dir is not None:
            prob_case_dir = save_prob_dir / case_id
            prob_case_dir.mkdir(parents=True, exist_ok=True)
            prob_nrrd_path = prob_case_dir / f"{case_id}_prob_bleeding.nrrd"
            save_probability_nrrd(prob_np, label_ref_path, prob_nrrd_path)

        result = {
            "case_id": case_id,
            "loss": float(loss.item()),
            "dice": float(dice_value),
            "iou": float(iou_value),
            "hd95": float(hd95_value) if not np.isnan(hd95_value) else "",
            "asd": float(asd_value) if not np.isnan(asd_value) else "",
            "rvd_percent": float(rvd_value) if not np.isnan(rvd_value) else "",
            "pred_positive_voxels": pred_positive_voxels,
            "gt_positive_voxels": gt_positive_voxels,
            "empty_pred": bool(empty_pred),
            "empty_gt": bool(empty_gt),
            "pred_nrrd_path": str(pred_nrrd_path),
            "prob_nrrd_path": str(prob_nrrd_path) if prob_nrrd_path is not None else "",
        }
        per_case_results.append(result)

        print(
            "[{}/{}] {} | loss={:.4f} | dice={:.4f} | iou={:.4f} | hd95={} | asd={} | rvd%={} | pred_pos={} | gt_pos={} | saved={}".format(
                batch_idx,
                len(loader),
                case_id,
                float(loss.item()),
                float(dice_value),
                float(iou_value),
                "NA" if skip_hd95 else f"{float(hd95_value):.4f}",
                "NA" if skip_asd or np.isnan(asd_value) else f"{float(asd_value):.4f}",
                "NA" if np.isnan(rvd_value) else f"{float(rvd_value):.2f}",
                pred_positive_voxels,
                gt_positive_voxels,
                str(pred_nrrd_path),
            )
        )

        del images, labels, logits, probs, preds, pred_cpu, label_cpu
        if device.type == "cuda" and empty_cache_after_case:
            torch.cuda.empty_cache()

    return per_case_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=str(_pre_args.gpu), help="Physical GPU id. Parsed before torch import.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--val_json", type=str, default=DEFAULT_VAL_JSON)
    parser.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)

    parser.add_argument("--input_ch", type=int, default=1)
    parser.add_argument("--output_ch", type=int, default=1)
    parser.add_argument("--init_feats", type=int, default=8)
    parser.add_argument("--cma_stages", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--cma_low_kernel_sizes", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--cma_init_gamma", type=float, default=0.1)
    parser.add_argument("--aniso_xy_kernel", type=int, default=7)
    parser.add_argument("--aniso_z_kernel", type=int, default=3)
    parser.add_argument("--no_cgcb", action="store_true")
    parser.add_argument("--cgcb_reduction", type=int, default=4)
    parser.add_argument("--cgcb_init_beta", type=float, default=0.05)
    parser.add_argument("--ugrmsf_stages", nargs="+", type=int, default=[10, 11])
    parser.add_argument("--ugrmsf_fusion_channels", type=int, default=0)
    parser.add_argument("--ugrmsf_init_alpha", type=float, default=0.0)
    parser.add_argument("--ugrmsf_edge_init_alpha", type=float, default=0.0)
    parser.add_argument("--ugrmsf_context_init_alpha", type=float, default=0.0)
    parser.add_argument("--no_ugrmsf_edge", action="store_true")
    parser.add_argument("--no_ugrmsf_crescent", action="store_true")
    parser.add_argument("--ugrmsf_uncertainty_power", type=float, default=1.0)
    parser.add_argument("--ugrmsf_no_detach_uncertainty", action="store_true")
    parser.add_argument("--non_strict_load", action="store_true")

    parser.add_argument("--roi_z", type=int, default=ROI_SIZE[0])
    parser.add_argument("--roi_y", type=int, default=ROI_SIZE[1])
    parser.add_argument("--roi_x", type=int, default=ROI_SIZE[2])
    parser.add_argument("--sw_batch_size", type=int, default=SW_BATCH_SIZE)
    parser.add_argument("--val_overlap", type=float, default=VAL_OVERLAP)
    parser.add_argument("--threshold", type=float, default=PRED_THRESHOLD)
    parser.add_argument("--skip_hd95", action="store_true")
    parser.add_argument("--skip_asd", action="store_true")
    parser.add_argument("--hd95_spacing", nargs=3, type=float, default=list(HD95_SPACING))
    parser.add_argument("--hd95_no_spacing", action="store_true")
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--empty_cache_after_case", action="store_true")
    parser.add_argument("--save_prob", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda") and (not args.disable_amp)
    roi_size = (int(args.roi_z), int(args.roi_y), int(args.roi_x))
    spacing = None if args.hd95_no_spacing else tuple(float(x) for x in args.hd95_spacing)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    pred_save_dir = save_dir / "predictions_nrrd"
    pred_save_dir.mkdir(parents=True, exist_ok=True)
    prob_save_dir = None
    if args.save_prob:
        prob_save_dir = save_dir / "probabilities_nrrd"
        prob_save_dir.mkdir(parents=True, exist_ok=True)

    print("Using device       :", device)
    print("Visible GPU        :", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if device.type == "cuda":
        print("GPU name           :", torch.cuda.get_device_name(0))
    print("Checkpoint         :", args.checkpoint)
    print("Val JSON           :", args.val_json)
    print("Save dir           :", save_dir)
    print("Prediction NRRD dir:", pred_save_dir)
    print("Save prob NRRD     :", bool(args.save_prob))
    print("ROI size zyx       :", roi_size)
    print("HD95/ASD spacing   :", spacing)
    print("Threshold          :", args.threshold)
    print("CMA stages         :", args.cma_stages)
    print("CMA low kernels    :", make_low_kernels(args.cma_low_kernel_sizes))
    print("CGCB enabled       :", not args.no_cgcb)
    print("UGRMSF stages        :", args.ugrmsf_stages)
    print("UGRMSF fusion ch     :", args.init_feats if args.ugrmsf_fusion_channels <= 0 else args.ugrmsf_fusion_channels)
    print("UGRMSF alpha init    :", args.ugrmsf_init_alpha)
    print("UGRMSF edge enabled  :", not args.no_ugrmsf_edge)
    print("UGRMSF crescent ctx  :", not args.no_ugrmsf_crescent)
    print("UGRMSF uncertainty   :", f"power={args.ugrmsf_uncertainty_power}, detach={not args.ugrmsf_no_detach_uncertainty}")
    print("num_workers        :", args.num_workers)
    print("pin_memory         :", args.pin_memory)

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    model, checkpoint, loaded_epoch, best_val_dice, best_val_hd95 = load_checkpoint_model(args.checkpoint, device, args)
    print("Loaded epoch       :", loaded_epoch)
    print("Best Dice in ckpt  :", best_val_dice)
    print("Best HD95 in ckpt  :", best_val_hd95)

    val_items = load_split_json(args.val_json)
    val_dataset = JSZJValWholeVolumeDataset(val_items)
    persistent_workers = bool(args.persistent_workers and args.num_workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=bool(args.pin_memory), persistent_workers=persistent_workers,
    )

    print("Val cases          :", len(val_items))
    print("=" * 100)

    per_case_results = evaluate_and_save_predictions(
        model=model, items=val_items, loader=val_loader, criterion=criterion, device=device,
        save_pred_dir=pred_save_dir, save_prob_dir=prob_save_dir, spacing=spacing,
        threshold=args.threshold, roi_size=roi_size, sw_batch_size=args.sw_batch_size,
        val_overlap=args.val_overlap, use_amp=use_amp, skip_hd95=args.skip_hd95,
        skip_asd=args.skip_asd, empty_cache_after_case=args.empty_cache_after_case,
    )

    dice_list = [float(x["dice"]) for x in per_case_results]
    iou_list = [float(x["iou"]) for x in per_case_results]
    loss_list = [float(x["loss"]) for x in per_case_results]
    hd95_list = [float(x["hd95"]) for x in per_case_results if x["hd95"] != ""]
    asd_list = [float(x["asd"]) for x in per_case_results if x["asd"] != ""]
    rvd_list = [float(x["rvd_percent"]) for x in per_case_results if x["rvd_percent"] != ""]

    empty_pred_count = sum(int(x["empty_pred"]) for x in per_case_results)
    empty_gt_count = sum(int(x["empty_gt"]) for x in per_case_results)

    worst_cases_by_hd95 = sorted(per_case_results, key=lambda x: float(x["hd95"]) if x["hd95"] != "" else -1, reverse=True)[:10]
    worst_cases_by_dice = sorted(per_case_results, key=lambda x: float(x["dice"]))[:10]

    summary = {
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint,
        "loaded_epoch": loaded_epoch,
        "best_val_dice_in_ckpt": best_val_dice,
        "best_val_hd95_in_ckpt": best_val_hd95,
        "split": "val",
        "n_cases": len(val_items),
        "hd95_spacing_zyx_mm": list(spacing) if spacing is not None else None,
        "prediction_threshold": args.threshold,
        "roi_size_zyx": list(roi_size),
        "cma_stages": list(args.cma_stages),
        "cma_low_kernels": [list(k) for k in make_low_kernels(args.cma_low_kernel_sizes)],
        "cma_init_gamma": args.cma_init_gamma,
        "aniso_xy_kernel": args.aniso_xy_kernel,
        "aniso_z_kernel": args.aniso_z_kernel,
        "cgcb_enabled": not args.no_cgcb,
        "cgcb_reduction": args.cgcb_reduction,
        "cgcb_init_beta": args.cgcb_init_beta,
        "ugrmsf_stages": list(args.ugrmsf_stages),
        "ugrmsf_fusion_channels": (args.init_feats if args.ugrmsf_fusion_channels <= 0 else args.ugrmsf_fusion_channels),
        "ugrmsf_init_alpha": args.ugrmsf_init_alpha,
        "ugrmsf_edge_init_alpha": args.ugrmsf_edge_init_alpha,
        "ugrmsf_context_init_alpha": args.ugrmsf_context_init_alpha,
        "ugrmsf_use_edge": not args.no_ugrmsf_edge,
        "ugrmsf_use_crescent_context": not args.no_ugrmsf_crescent,
        "ugrmsf_uncertainty_power": args.ugrmsf_uncertainty_power,
        "ugrmsf_uncertainty_detach": not args.ugrmsf_no_detach_uncertainty,
        "predictions_nrrd_dir": str(pred_save_dir),
        "probabilities_nrrd_dir": str(prob_save_dir) if prob_save_dir is not None else None,
        "loss_summary": summarize(loss_list),
        "dice_summary": summarize(dice_list),
        "iou_summary": summarize(iou_list),
        "hd95_summary": summarize(hd95_list) if not args.skip_hd95 else None,
        "asd_summary": summarize(asd_list) if not args.skip_asd else None,
        "rvd_percent_summary": summarize(rvd_list) if len(rvd_list) > 0 else None,
        "empty_pred_count": int(empty_pred_count),
        "empty_gt_count": int(empty_gt_count),
        "worst_cases_by_hd95": worst_cases_by_hd95,
        "worst_cases_by_dice": worst_cases_by_dice,
    }

    csv_path = save_dir / "per_case_metrics_val.csv"
    json_path = save_dir / "summary_val.json"
    save_per_case_csv(csv_path, per_case_results)
    save_summary_json(json_path, summary)

    print("\n" + "=" * 100)
    print("Evaluation finished.")
    print("Checkpoint         :", args.checkpoint)
    print("Loaded epoch       :", loaded_epoch)
    print("Val cases          :", len(val_items))
    print("Mean loss          :", summary["loss_summary"]["mean"])
    print("Mean Dice          :", summary["dice_summary"]["mean"])
    print("Median Dice        :", summary["dice_summary"]["median"])
    print("Mean IoU           :", summary["iou_summary"]["mean"])
    print("Median IoU         :", summary["iou_summary"]["median"])
    if not args.skip_hd95:
        print("Mean HD95          :", summary["hd95_summary"]["mean"])
        print("Median HD95        :", summary["hd95_summary"]["median"])
        print("P90 HD95           :", summary["hd95_summary"]["p90"])
    if not args.skip_asd and summary["asd_summary"] is not None:
        print("Mean ASD           :", summary["asd_summary"]["mean"])
        print("Median ASD         :", summary["asd_summary"]["median"])
    if summary.get("rvd_percent_summary", None) is not None:
        print("Mean RVD(%)        :", summary["rvd_percent_summary"]["mean"])
        print("Median RVD(%)      :", summary["rvd_percent_summary"]["median"])
    print("Empty pred count   :", empty_pred_count)
    print("Empty gt count     :", empty_gt_count)
    print("Prediction NRRD dir:", str(pred_save_dir))
    if prob_save_dir is not None:
        print("Probability NRRD dir:", str(prob_save_dir))
    print("Per-case CSV       :", str(csv_path))
    print("Summary JSON       :", str(json_path))
    print("=" * 100)


if __name__ == "__main__":
    main()
