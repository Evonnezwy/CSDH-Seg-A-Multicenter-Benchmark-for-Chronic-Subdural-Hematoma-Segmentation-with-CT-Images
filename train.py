import os
import sys
import argparse

_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--gpu", default="0")
_pre_args, _ = _pre_parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import math
import random
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

try:
    from torch.amp import autocast, GradScaler
    NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    NEW_AMP = False

from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from CesU_Net import CesU_Net
from dataloader import build_Center1_patch3d_loaders


SEED = 1
NUM_EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-5
ROI_SIZE = (32, 192, 192)  # z, y, x
SW_BATCH_SIZE = 1
VAL_OVERLAP = 0.25
HD95_SPACING = (5.0, 0.5, 0.5)
EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=str(_pre_args.gpu), help="Physical GPU id.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    parser.add_argument(
        "--save_dir",
        default=("./checkpoints_Center1_CesU_Net"),
    )

    parser.add_argument("--input_ch", type=int, default=1)
    parser.add_argument("--output_ch", type=int, default=1)
    parser.add_argument("--init_feats", type=int, default=8)

    parser.add_argument(
        "--cma_stages",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5]
    )
    parser.add_argument(
        nargs="+",
        type=int,
        default=[3, 5, 7]
    )
    parser.add_argument("--cma_init_gamma", type=float, default=0.1)
    parser.add_argument("--aniso_xy_kernel", type=int, default=7)
    parser.add_argument("--aniso_z_kernel", type=int, default=3)


    parser.add_argument("--use_cgcb", action="store_true", default=True)
    parser.add_argument("--no_cgcb", action="store_true", help="Disable bottleneck Crescent Global Context Block.")
    parser.add_argument("--cgcb_reduction", type=int, default=4)
    parser.add_argument("--cgcb_init_beta", type=float, default=0.05)


    parser.add_argument("--ugrmsf_stages", nargs="+", type=int, default=[10, 11],
                        help="Decoder stages for residual multi-scale fusion. Recommended first: 10 11 or 11.")
    parser.add_argument("--ugrmsf_fusion_channels", type=int, default=0,
                        help="Fusion channels in UGRMSF decoder. 0 means init_feats.")
    parser.add_argument("--ugrmsf_init_alpha", type=float, default=0.0,
                        help="Residual scale of UGRMSF logits. Default 0.0 means initial model equals CMA-FSE.")
    parser.add_argument("--ugrmsf_edge_init_alpha", type=float, default=0.0,
                        help="Residual scale inside feature-domain edge enhancement.")
    parser.add_argument("--ugrmsf_context_init_alpha", type=float, default=0.0,
                        help="Residual scale for crescent context in UGRMSF decoder.")
    parser.add_argument("--no_ugrmsf_edge", action="store_true",
                        help="Disable weak edge residual enhancement inside UGRMSF decoder.")
    parser.add_argument("--no_ugrmsf_crescent", action="store_true",
                        help="Disable crescent directional context inside UGRMSF decoder.")
    parser.add_argument("--ugrmsf_uncertainty_power", type=float, default=1.0,
                        help="Power applied to uncertainty gate 4*p*(1-p). Larger values make refinement more boundary-focused.")
    parser.add_argument("--ugrmsf_no_detach_uncertainty", action="store_true",
                        help="Do not detach uncertainty gate from base logits. Default detaches for stable refinement training.")

    # Inference / validation settings.
    parser.add_argument("--roi_z", type=int, default=ROI_SIZE[0])
    parser.add_argument("--roi_y", type=int, default=ROI_SIZE[1])
    parser.add_argument("--roi_x", type=int, default=ROI_SIZE[2])
    parser.add_argument("--sw_batch_size", type=int, default=SW_BATCH_SIZE)
    parser.add_argument("--val_overlap", type=float, default=VAL_OVERLAP)

    # Resume / initialization settings.
    parser.add_argument("--resume", action="store_true", default=True, help="Auto resume if checkpoint exists.")
    parser.add_argument("--no_resume", action="store_true", help="Disable resume and train from scratch.")
    parser.add_argument("--resume_path", default=None, help="Manually specify checkpoint path.")
    parser.add_argument("--strict_load", action="store_true", default=True)
    parser.add_argument("--non_strict_load", action="store_true")
    parser.add_argument("--init_model_path", default=None, help="Optional pretrained model initialization path.")

    # HD95 settings.
    parser.add_argument("--hd95_spacing", nargs=3, type=float, default=list(HD95_SPACING))
    parser.add_argument("--hd95_no_spacing", action="store_true")
    parser.add_argument("--val_hd95_every", type=int, default=1)
    parser.add_argument("--compute_train_hd95", action="store_true", help="Off by default to avoid memory spikes.")
    parser.add_argument("--empty_cache_after_val_batch", action="store_true")

    return parser.parse_args()

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
        if NEW_AMP:
            return autocast(device_type="cuda", enabled=enabled)
        return autocast(enabled=enabled)
    return nullcontext()


def make_grad_scaler(enabled=True):
    if NEW_AMP:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


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


def dice_coefficient_from_logits(logits, targets, threshold=0.5, smooth=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    dims = (1, 2, 3, 4)
    intersection = torch.sum(preds * targets, dims)
    denominator = torch.sum(preds, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.mean().item())


def dice_coefficient_from_probs(probs, targets, threshold=0.5, smooth=1e-6):
    preds = (probs >= threshold).float()
    dims = (1, 2, 3, 4)
    intersection = torch.sum(preds * targets, dims)
    denominator = torch.sum(preds, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return float(dice.mean().item())


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


def hd95_from_probs_cpu(probs, targets, threshold=0.5, spacing=None):
    probs_cpu = probs.detach().float().cpu()
    targets_cpu = targets.detach().float().cpu()
    preds_cpu = (probs_cpu >= threshold).float()
    vals = []
    for b in range(preds_cpu.shape[0]):
        vals.append(hd95_single_sample(preds_cpu[b:b + 1], targets_cpu[b:b + 1], spacing=spacing))
    return float(sum(vals) / len(vals))


def hd95_from_logits_cpu(logits, targets, threshold=0.5, spacing=None):
    return hd95_from_probs_cpu(torch.sigmoid(logits.detach()), targets, threshold=threshold, spacing=spacing)


def save_history_json(path, history_dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, ensure_ascii=False, indent=2)


def load_history_json(path):
    if not Path(path).exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _strip_module_prefix(state_dict):
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def save_checkpoint(save_path, epoch, model, optimizer, scheduler, scaler, train_metrics, val_metrics, best_val_dice, best_val_hd95):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "best_val_dice": best_val_dice,
        "best_val_hd95": best_val_hd95,
    }, save_path)


def find_latest_checkpoint(save_dir):
    save_dir = Path(save_dir)
    last_path = save_dir / "last_model.pth"
    if last_path.exists():
        return last_path
    candidates = sorted(list(save_dir.glob("epoch*.pth")) + list(save_dir.glob("*.pt")), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def resume_from_checkpoint(ckpt_path, model, optimizer=None, scheduler=None, scaler=None, device="cpu", strict=True):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = _strip_module_prefix(extract_state_dict(checkpoint))
    load_info = model.load_state_dict(state, strict=strict)
    if not strict:
        print("[INFO] Non-strict load info:", load_info)

    loaded_optimizer = loaded_scheduler = loaded_scaler = False
    if isinstance(checkpoint, dict):
        if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                loaded_optimizer = True
            except Exception as e:
                print(f"[WARN] Failed to load optimizer state: {e}")
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                loaded_scheduler = True
            except Exception as e:
                print(f"[WARN] Failed to load scheduler state: {e}")
        if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            try:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
                loaded_scaler = True
            except Exception as e:
                print(f"[WARN] Failed to load scaler state: {e}")
        epoch = int(checkpoint.get("epoch", 0))
        best_val_dice = float(checkpoint.get("best_val_dice", -1.0))
        best_val_hd95 = float(checkpoint.get("best_val_hd95", float("inf")))
    else:
        epoch = 0
        best_val_dice = -1.0
        best_val_hd95 = float("inf")
    return {
        "path": str(ckpt_path), "epoch": epoch, "start_epoch": epoch + 1,
        "best_val_dice": best_val_dice, "best_val_hd95": best_val_hd95,
        "loaded_optimizer": loaded_optimizer, "loaded_scheduler": loaded_scheduler, "loaded_scaler": loaded_scaler,
    }


def initialize_from_checkpoint(init_path, model, device):
    init_path = Path(init_path)
    if not init_path.exists():
        raise FileNotFoundError(f"init_model_path not found: {init_path}")
    checkpoint = torch.load(init_path, map_location=device)
    state = _strip_module_prefix(extract_state_dict(checkpoint))
    info = model.load_state_dict(state, strict=False)
    print("[INIT] Loaded pretrained weights non-strictly:", init_path)
    print("       Missing keys   :", list(info.missing_keys)[:30], "..." if len(info.missing_keys) > 30 else "")
    print("       Unexpected keys:", list(info.unexpected_keys)[:30], "..." if len(info.unexpected_keys) > 30 else "")


def fmt_metric(x, digits=4):
    if x is None:
        return "NA"
    try:
        if np.isnan(x):
            return "NA"
    except Exception:
        pass
    return f"{x:.{digits}f}"


# =========================================================
# Train / validation
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp=True, spacing=None, compute_hd95=False):
    model.train()
    running_loss = running_dice = running_hd95 = 0.0
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_dice = dice_coefficient_from_logits(logits.detach(), labels)
        if compute_hd95:
            batch_hd95 = hd95_from_logits_cpu(logits.detach(), labels, spacing=spacing)
            running_hd95 += batch_hd95
        running_loss += float(loss.item())
        running_dice += batch_dice
        num_batches += 1
        del images, labels, logits, loss

    avg_hd95 = running_hd95 / num_batches if compute_hd95 else None
    return running_loss / num_batches, running_dice / num_batches, avg_hd95


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, roi_size, sw_batch_size, val_overlap, use_amp=True, spacing=None, compute_hd95=True, empty_cache_after_batch=False):
    model.eval()
    running_loss = running_dice = running_hd95 = 0.0
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with autocast_context(device, enabled=use_amp):
            logits = sliding_window_inference(inputs=images, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model, overlap=val_overlap, mode="gaussian")
            loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        batch_dice = dice_coefficient_from_probs(probs, labels)
        if compute_hd95:
            running_hd95 += hd95_from_probs_cpu(probs, labels, spacing=spacing)
        running_loss += float(loss.item())
        running_dice += batch_dice
        num_batches += 1
        del images, labels, logits, probs, loss
        if empty_cache_after_batch and device.type == "cuda":
            torch.cuda.empty_cache()
    avg_hd95 = running_hd95 / num_batches if compute_hd95 else None
    return running_loss / num_batches, running_dice / num_batches, avg_hd95


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()
    seed_everything(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    last_model_path = save_dir / "last_model.pth"
    history_json_path = save_dir / "training_history.json"

    roi_size = (args.roi_z, args.roi_y, args.roi_x)
    hd95_spacing = None if args.hd95_no_spacing else tuple(float(v) for v in args.hd95_spacing)
    cma_low_kernels = make_low_kernels(args.cma_low_kernel_sizes)
    use_cgcb = bool(args.use_cgcb and not args.no_cgcb)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Using device:", device)
    print("Using seed  :", args.seed)
    print("Visible GPU :", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if device.type == "cuda":
        print("GPU name    :", torch.cuda.get_device_name(0))

    train_loader, val_loader = build_jszj_patch3d_loaders(seed=args.seed)

    model = UNet3DMMS_CMA_UGRMSF(
        input_ch=args.input_ch,
        output_ch=args.output_ch,
        init_feats=args.init_feats,
        cma_stages=tuple(args.cma_stages),
        cma_low_kernels=cma_low_kernels,
        cma_init_gamma=args.cma_init_gamma,
        aniso_xy_kernel=args.aniso_xy_kernel,
        aniso_z_kernel=args.aniso_z_kernel,
        use_cgcb=use_cgcb,
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

    criterion = DiceBCELoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=15)
    scaler = make_grad_scaler(enabled=use_amp)

    best_val_dice = -1.0
    best_val_hd95 = float("inf")
    start_epoch = 1
    resume_info = None

    do_resume = (not args.no_resume) and (args.resume or args.resume_path is not None)
    if do_resume:
        ckpt_path = Path(args.resume_path) if args.resume_path is not None else find_latest_checkpoint(save_dir)
        if ckpt_path is not None and ckpt_path.exists():
            strict = bool(args.strict_load and not args.non_strict_load)
            resume_info = resume_from_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, device, strict=strict)
            start_epoch = resume_info["start_epoch"]
            best_val_dice = resume_info["best_val_dice"]
            best_val_hd95 = resume_info["best_val_hd95"]
            print("\n[RESUME]")
            print("  checkpoint      :", resume_info["path"])
            print("  loaded epoch    :", resume_info["epoch"])
            print("  start epoch     :", start_epoch)
            print("  best_val_dice   :", best_val_dice)
            print("  best_val_hd95   :", best_val_hd95)
            print("  optimizer loaded:", resume_info["loaded_optimizer"])
            print("  scheduler loaded:", resume_info["loaded_scheduler"])
            print("  scaler loaded   :", resume_info["loaded_scaler"])
            print()
        else:
            print("[INFO] No checkpoint found. Training from scratch.")
            if args.init_model_path is not None:
                initialize_from_checkpoint(args.init_model_path, model, device)
    else:
        print("[INFO] Resume disabled. Training from scratch.")
        if args.init_model_path is not None:
            initialize_from_checkpoint(args.init_model_path, model, device)

    loaded_history = load_history_json(history_json_path) if resume_info is not None else None
    if loaded_history is not None:
        history = loaded_history
        history.setdefault("resume_events", [])
        history["resume_events"].append({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint": resume_info["path"],
            "loaded_epoch": resume_info["epoch"],
            "start_epoch": start_epoch,
        })
    else:
        history = {"config": {}, "epochs": [], "saved_checkpoints": [], "best": {"val_dice": None, "val_hd95": None, "best_dice_checkpoint": None, "best_hd95_checkpoint": None}, "resume_events": []}

    history["config"].update({
        "device": str(device),
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "seed": args.seed,
        "num_epochs_total": args.num_epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "roi_size_zyx": list(roi_size),
        "sw_batch_size": args.sw_batch_size,
        "val_overlap": args.val_overlap,
        "hd95_spacing_zyx": list(hd95_spacing) if hd95_spacing is not None else None,
        "model": "UNet3DMMS_CMA_UGRMSF",
        "model_kwargs": {
            "input_ch": args.input_ch,
            "output_ch": args.output_ch,
            "init_feats": args.init_feats,
            "cma_stages": list(args.cma_stages),
            "cma_low_kernels": [list(k) for k in cma_low_kernels],
            "cma_init_gamma": args.cma_init_gamma,
            "aniso_xy_kernel": args.aniso_xy_kernel,
            "aniso_z_kernel": args.aniso_z_kernel,
            "use_cgcb": use_cgcb,
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
        },
        "compute_train_hd95": bool(args.compute_train_hd95),
        "val_hd95_every": int(args.val_hd95_every),
        "save_dir": str(save_dir),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    if start_epoch > args.num_epochs:
        print(f"[INFO] start_epoch={start_epoch} > num_epochs={args.num_epochs}. Nothing to train.")
        return

    prev_lr = optimizer.param_groups[0]["lr"]
    for epoch in range(start_epoch, args.num_epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_dice, train_hd95 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            use_amp=use_amp, spacing=hd95_spacing, compute_hd95=args.compute_train_hd95,
        )
        compute_val_hd95 = args.val_hd95_every > 0 and (epoch % args.val_hd95_every == 0)
        val_loss, val_dice, val_hd95 = validate_one_epoch(
            model, val_loader, criterion, device,
            roi_size=roi_size, sw_batch_size=args.sw_batch_size, val_overlap=args.val_overlap,
            use_amp=use_amp, spacing=hd95_spacing, compute_hd95=compute_val_hd95,
            empty_cache_after_batch=args.empty_cache_after_val_batch,
        )

        scheduler.step(val_dice)
        new_lr = optimizer.param_groups[0]["lr"]

        print(
            "Epoch [{:03d}/{:03d}] | lr: {:.6e} | train_loss: {:.4f} | train_dice: {:.4f} | train_hd95: {} | "
            "val_loss: {:.4f} | val_dice: {:.4f} | val_hd95: {}".format(
                epoch, args.num_epochs, current_lr, train_loss, train_dice, fmt_metric(train_hd95),
                val_loss, val_dice, fmt_metric(val_hd95),
            )
        )
        if abs(new_lr - prev_lr) > 1e-15:
            print(">>> LR changed: {:.6e} -> {:.6e}".format(prev_lr, new_lr))
        prev_lr = new_lr

        train_metrics = {"loss": train_loss, "dice": train_dice, "hd95": train_hd95}
        val_metrics = {"loss": val_loss, "dice": val_dice, "hd95": val_hd95}
        improved_dice = val_dice > (best_val_dice + EPS)
        improved_hd95 = False if val_hd95 is None else val_hd95 < (best_val_hd95 - EPS)
        saved_this_epoch = False
        saved_ckpt_name = None

        if improved_dice or improved_hd95:
            tags = []
            if improved_dice:
                tags.append("bestDice")
                best_val_dice = val_dice
            if improved_hd95:
                tags.append("bestHD95")
                best_val_hd95 = val_hd95
            hd95_for_name = val_hd95 if val_hd95 is not None else float("nan")
            ckpt_name = "epoch{:03d}_valDice{:.4f}_valHD95{:.4f}_{}.pth".format(epoch, val_dice, hd95_for_name, "_".join(tags))
            ckpt_path = save_dir / ckpt_name
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, scaler, train_metrics, val_metrics, best_val_dice, best_val_hd95)
            saved_this_epoch = True
            saved_ckpt_name = ckpt_name
            if improved_dice:
                history["best"]["val_dice"] = val_dice
                history["best"]["best_dice_checkpoint"] = ckpt_name
            if improved_hd95:
                history["best"]["val_hd95"] = val_hd95
                history["best"]["best_hd95_checkpoint"] = ckpt_name
            history["saved_checkpoints"].append({"epoch": epoch, "checkpoint": ckpt_name, "val_dice": val_dice, "val_hd95": val_hd95, "tags": tags})
            print(">>> Checkpoint saved:", ckpt_name)

        save_checkpoint(last_model_path, epoch, model, optimizer, scheduler, scaler, train_metrics, val_metrics, best_val_dice, best_val_hd95)
        history["epochs"].append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_loss, "train_dice": train_dice, "train_hd95": train_hd95,
            "val_loss": val_loss, "val_dice": val_dice, "val_hd95": val_hd95,
            "saved_checkpoint": saved_this_epoch, "checkpoint_name": saved_ckpt_name,
        })
        save_history_json(history_json_path, history)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("Training finished.")
    print("Best val dice: {:.4f}".format(best_val_dice))
    print("Best val hd95: {:.4f}".format(best_val_hd95))
    print("Last model saved to:", last_model_path)
    print("History saved to   :", history_json_path)


if __name__ == "__main__":
    main()
