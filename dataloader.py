import json
import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader


TRAIN_JSON = Path("./train.json")
VAL_JSON = Path("./val.json")

EXPECTED_SHAPE_ZYX = (32, 512, 512)   
PATCH_SIZE_ZYX = (32, 192, 192)    

CLIP_MIN = -50.0
CLIP_MAX = 150.0

BATCH_SIZE_TRAIN = 2
BATCH_SIZE_VAL = 1
NUM_WORKERS = 0

POSITIVE_SAMPLE_PROB = 0.7

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)

def load_split_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    datalist = []

    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
        for case_id, info in obj["data"].items():
            ct_path = info.get("ct_path", None)
            seg_path = info.get("seg_path", None)
            if ct_path is None or seg_path is None:
                continue

            datalist.append({
                "case_id": case_id,
                "image": ct_path,
                "label": seg_path,
            })
        return datalist

    raise ValueError("unrecognized json format: {}".format(json_path))


def read_sitk(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img) 
    return img, arr

def normalize_ct(image_arr, clip_min=-50.0, clip_max=150.0):
    image_arr = np.clip(image_arr, clip_min, clip_max)
    image_arr = (image_arr - clip_min) / (clip_max - clip_min)
    return image_arr.astype(np.float32)

def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def sample_patch_start_by_foreground(label_arr, patch_size_zyx, positive_prob=0.7):

    z, y, x = label_arr.shape
    pz, py, px = patch_size_zyx

    assert pz <= z and py <= y and px <= x, "patch size exceeds image size"
    start_z = 0

    fg = np.where(label_arr > 0)
    has_fg = len(fg[0]) > 0

    use_positive = has_fg and (random.random() < positive_prob)

    if use_positive:
        idx = random.randint(0, len(fg[0]) - 1)
        cy = int(fg[1][idx])
        cx = int(fg[2][idx])

        start_y = cy - py // 2
        start_x = cx - px // 2

        start_y = clamp(start_y, 0, y - py)
        start_x = clamp(start_x, 0, x - px)
    else:
        start_y = random.randint(0, y - py)
        start_x = random.randint(0, x - px)

    return start_z, start_y, start_x


def crop_patch(arr, start_z, start_y, start_x, patch_size_zyx):
    pz, py, px = patch_size_zyx
    return arr[
        start_z:start_z + pz,
        start_y:start_y + py,
        start_x:start_x + px
    ]

def random_augment_patch(image_arr, label_arr):
    if random.random() < 0.5:
        image_arr = np.flip(image_arr, axis=2)
        label_arr = np.flip(label_arr, axis=2)

    if random.random() < 0.5:
        image_arr = np.flip(image_arr, axis=1)
        label_arr = np.flip(label_arr, axis=1)

    if random.random() < 0.2:
        image_arr = np.flip(image_arr, axis=0)
        label_arr = np.flip(label_arr, axis=0)

    return image_arr, label_arr

class Center1TrainPatchDataset(Dataset):
    def __init__(self, items, patch_size_zyx=(32, 192, 192), positive_prob=0.7):
        self.items = items
        self.patch_size_zyx = patch_size_zyx
        self.positive_prob = positive_prob

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        case_id = item["case_id"]

        _, image_arr = read_sitk(item["image"])
        _, label_arr = read_sitk(item["label"])

        if tuple(image_arr.shape) != EXPECTED_SHAPE_ZYX:
            raise ValueError(
                "case {} image shape {} != expected {}".format(
                    case_id, image_arr.shape, EXPECTED_SHAPE_ZYX
                )
            )

        if tuple(label_arr.shape) != EXPECTED_SHAPE_ZYX:
            raise ValueError(
                "case {} label shape {} != expected {}".format(
                    case_id, label_arr.shape, EXPECTED_SHAPE_ZYX
                )
            )

        label_arr = (label_arr > 0).astype(np.uint8)
        image_arr = normalize_ct(image_arr, CLIP_MIN, CLIP_MAX)

        start_z, start_y, start_x = sample_patch_start_by_foreground(
            label_arr,
            self.patch_size_zyx,
            positive_prob=self.positive_prob
        )

        image_patch = crop_patch(image_arr, start_z, start_y, start_x, self.patch_size_zyx)
        label_patch = crop_patch(label_arr, start_z, start_y, start_x, self.patch_size_zyx)

        image_patch, label_patch = random_augment_patch(image_patch, label_patch)

        image_patch = np.ascontiguousarray(image_patch)
        label_patch = np.ascontiguousarray(label_patch)

        image_tensor = torch.from_numpy(image_patch).unsqueeze(0).float()   
        label_tensor = torch.from_numpy(label_patch).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "case_id": case_id,
            "patch_start": (start_z, start_y, start_x),
        }

class Center1ValWholeVolumeDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        case_id = item["case_id"]

        _, image_arr = read_sitk(item["image"])
        _, label_arr = read_sitk(item["label"])

        if tuple(image_arr.shape) != EXPECTED_SHAPE_ZYX:
            raise ValueError(
                "case {} image shape {} != expected {}".format(
                    case_id, image_arr.shape, EXPECTED_SHAPE_ZYX
                )
            )

        if tuple(label_arr.shape) != EXPECTED_SHAPE_ZYX:
            raise ValueError(
                "case {} label shape {} != expected {}".format(
                    case_id, label_arr.shape, EXPECTED_SHAPE_ZYX
                )
            )

        label_arr = (label_arr > 0).astype(np.uint8)
        image_arr = normalize_ct(image_arr, CLIP_MIN, CLIP_MAX)

        image_arr = np.ascontiguousarray(image_arr)
        label_arr = np.ascontiguousarray(label_arr)

        image_tensor = torch.from_numpy(image_arr).unsqueeze(0).float()
        label_tensor = torch.from_numpy(label_arr).unsqueeze(0).float()

        return {
            "image": image_tensor,
            "label": label_tensor,
            "case_id": case_id,
        }

def build_Center1_patch3d_loaders(seed=42):
    train_items = load_split_json(TRAIN_JSON)
    val_items = load_split_json(VAL_JSON)

    print("Train cases:", len(train_items))
    print("Val cases  :", len(val_items))

    train_ds = Center1TrainPatchDataset(
        train_items,
        patch_size_zyx=PATCH_SIZE_ZYX,
        positive_prob=POSITIVE_SAMPLE_PROB,
    )

    val_ds = Center1ValWholeVolumeDataset(val_items)

    g_train = torch.Generator()
    g_train.manual_seed(seed)

    g_val = torch.Generator()
    g_val.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        worker_init_fn=seed_worker,
        generator=g_train,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE_VAL,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True if NUM_WORKERS > 0 else False,
        worker_init_fn=seed_worker,
        generator=g_val,
    )

    return train_loader, val_loader
