# datasets_multitask.py
# Multi-task datasets for:
# - VisDrone detection
# - DOTA + iSAID detection/segmentation
# - WildUAV depth + segmentation
# - UAVScenes depth + segmentation

from __future__ import annotations
import os, glob, math
from typing import Dict, List, Tuple, Optional, Any, Sequence
import re, random
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from calibration_results import scenename_to_calibration

try:
    from terra_depth_utils import raycast_terra_depth_tile
except ImportError:
    # If Terra helper isn't available, we'll silently fall back to LiDAR.
    raycast_terra_depth_tile = None
import open3d as o3d

# =========================
# Unified class space
# =========================

UNIFIED_CLASSES = [
    "background",        # 0
    "person",            # 1
    "bicycle",           # 2
    "car",               # 3
    "motorcycle",        # 4
    "bus",               # 5
    "truck",             # 6
    "van",               # 7
    "tricycle",          # 8
    "small-vehicle",     # 9
    "large-vehicle",     # 10
    "plane",             # 11
    "ship",              # 12
    "storage-tank",      # 13
    "harbor",            # 14
    "bridge",            # 15
    "helicopter",        # 16
    "soccer-field",      # 17
    "tennis-court",      # 18
    "basketball-court",  # 19
    "roundabout",        # 20
    "swimming-pool",     # 21
    "baseball-diamond",  # 22

    # --- WildUAV + UAVScenes semantics ---
    "sky",               # 23
    "tree",              # 24
    "tree-fallen",       # 25
    "ground-dirt",       # 26
    "ground-vegetation", # 27
    "rock",              # 28
    "water",             # 29
    "building",          # 30
    "fence",             # 31
    "road",              # 32
    "sidewalk",          # 33
    "roof",              # 34
    "field-green",       # 35  
    "field-wild",        # 36
    "solar-board",       # 37
    "umbrella",          # 38
    "roof-transparent",  # 39
    "car-park",          # 40
    "container",         # 41
    "airstrip",          # 42
    "traffic-barrier",   # 43
]
UNIFIED_NAME_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(UNIFIED_CLASSES)}

# VisDrone class-id -> unified
# 0: ignored regions, 1: pedestrian, 2: people, 3: bicycle, 4: car,
# 5: van, 6: truck, 7: tricycle, 8: awning-tricycle,
# 9: bus, 10: motor, 11: others
VISDRONE_TO_UNIFIED: Dict[int, int] = {
    0: UNIFIED_NAME_TO_ID["background"],
    1: UNIFIED_NAME_TO_ID["person"],
    2: UNIFIED_NAME_TO_ID["person"],
    3: UNIFIED_NAME_TO_ID["bicycle"],
    4: UNIFIED_NAME_TO_ID["car"],
    5: UNIFIED_NAME_TO_ID["van"],
    6: UNIFIED_NAME_TO_ID["truck"],
    7: UNIFIED_NAME_TO_ID["tricycle"],
    8: UNIFIED_NAME_TO_ID["tricycle"],
    9: UNIFIED_NAME_TO_ID["bus"],
    10: UNIFIED_NAME_TO_ID["motorcycle"],
    11: UNIFIED_NAME_TO_ID["background"],
}

# DOTA name -> unified
DOTA_NAME_TO_UNIFIED: Dict[str, int] = {
    "plane": UNIFIED_NAME_TO_ID["plane"],
    "ship": UNIFIED_NAME_TO_ID["ship"],
    "storage-tank": UNIFIED_NAME_TO_ID["storage-tank"],
    "harbor": UNIFIED_NAME_TO_ID["harbor"],
    "bridge": UNIFIED_NAME_TO_ID["bridge"],
    "helicopter": UNIFIED_NAME_TO_ID["helicopter"],
    "soccer-ball-field": UNIFIED_NAME_TO_ID["soccer-field"],
    "soccer-field": UNIFIED_NAME_TO_ID["soccer-field"],
    "tennis-court": UNIFIED_NAME_TO_ID["tennis-court"],
    "basketball-court": UNIFIED_NAME_TO_ID["basketball-court"],
    "roundabout": UNIFIED_NAME_TO_ID["roundabout"],
    "swimming-pool": UNIFIED_NAME_TO_ID["swimming-pool"],
    "baseball-diamond": UNIFIED_NAME_TO_ID["baseball-diamond"],
    "large-vehicle": UNIFIED_NAME_TO_ID["large-vehicle"],
    "small-vehicle": UNIFIED_NAME_TO_ID["small-vehicle"],
    "ground-track-field": UNIFIED_NAME_TO_ID["soccer-field"],
}

# Minimal iSAID color -> unified mapping (extend as needed)
ISAID_COLOR_TO_UNIFIED: Dict[Tuple[int, int, int], int] = {
    # Placeholder example; extend with your real palette.
    (0, 0, 255): UNIFIED_NAME_TO_ID["background"],
}

# WildUAV RGB -> unified
WILDUAV_COLOR_TO_UNIFIED: Dict[Tuple[int,int,int], int] = {
    # Class: RGB -> unified
    (0, 255, 255): UNIFIED_NAME_TO_ID["sky"],               # Sky
    (0, 127,   0): UNIFIED_NAME_TO_ID["tree"],    # Deciduous trees
    (19, 132, 69): UNIFIED_NAME_TO_ID["tree"],   # Coniferous trees
    (0,  53,  65): UNIFIED_NAME_TO_ID["tree"],       # Fallen trees
    (130, 76,  0): UNIFIED_NAME_TO_ID["ground-dirt"],       # Dirt ground
    (152, 251,152): UNIFIED_NAME_TO_ID["ground-vegetation"],# Ground vegetation
    (151,126,171): UNIFIED_NAME_TO_ID["rock"],              # Rocks
    (0,   0, 255): UNIFIED_NAME_TO_ID["water"],             # Water plane
    (250,150,  0): UNIFIED_NAME_TO_ID["building"],          # Building
    (115,176,195): UNIFIED_NAME_TO_ID["fence"],             # Fence
    (128, 64,128): UNIFIED_NAME_TO_ID["road"],              # Road
    (255, 77,228): UNIFIED_NAME_TO_ID["sidewalk"],          # Sidewalk
    (123,123,123): UNIFIED_NAME_TO_ID["car"],               # Static car
    (255,255,255): UNIFIED_NAME_TO_ID["car"],               # Moving car
    (200,  0,  0): UNIFIED_NAME_TO_ID["person"],            # People
    (0,   0,   0): UNIFIED_NAME_TO_ID["background"],        # Empty
}

def wild_uav_rgb_to_unified(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert WildUAV 3-channel color mask [H,W,3] to unified label IDs [H,W].
    """
    h, w, _ = mask_rgb.shape
    out = np.full((h, w), UNIFIED_NAME_TO_ID["background"], dtype=np.int64)
    for (r, g, b), cid in WILDUAV_COLOR_TO_UNIFIED.items():
        m = (mask_rgb[..., 0] == r) & (mask_rgb[..., 1] == g) & (mask_rgb[..., 2] == b)
        out[m] = cid
    return out

# UAVScenes integer ID -> unified
UAVSCENES_ID_TO_UNIFIED: Dict[int, int] = {
    0:  UNIFIED_NAME_TO_ID["background"],     # background
    1:  UNIFIED_NAME_TO_ID["roof"],           # roof
    2:  UNIFIED_NAME_TO_ID["road"],           # dirt_motor_road
    3:  UNIFIED_NAME_TO_ID["road"],           # paved_motor_road
    4:  UNIFIED_NAME_TO_ID["water"],          # river
    5:  UNIFIED_NAME_TO_ID["swimming-pool"],  # pool
    6:  UNIFIED_NAME_TO_ID["bridge"],         # bridge
    7:  UNIFIED_NAME_TO_ID["background"],     # unlabeled / misc
    8:  UNIFIED_NAME_TO_ID["building"],       # likely building / structure
    9:  UNIFIED_NAME_TO_ID["container"],      # container
    10: UNIFIED_NAME_TO_ID["airstrip"],       # airstrip
    11: UNIFIED_NAME_TO_ID["traffic-barrier"],# traffic_barrier
    12: UNIFIED_NAME_TO_ID["background"],     # misc
    13: UNIFIED_NAME_TO_ID["field-green"],    # green_field
    14: UNIFIED_NAME_TO_ID["field-wild"],     # wild_field
    15: UNIFIED_NAME_TO_ID["solar-board"],    # solar_board
    16: UNIFIED_NAME_TO_ID["umbrella"],       # umbrella
    17: UNIFIED_NAME_TO_ID["roof-transparent"], # transparent_roof
    18: UNIFIED_NAME_TO_ID["car-park"],       # car_park
    19: UNIFIED_NAME_TO_ID["sidewalk"],       # paved_walk
    20: UNIFIED_NAME_TO_ID["car"],            # sedan
    21: UNIFIED_NAME_TO_ID["background"],     # misc / unknown
    22: UNIFIED_NAME_TO_ID["background"],     # misc / unknown
    23: UNIFIED_NAME_TO_ID["background"],     # misc / unknown
    24: UNIFIED_NAME_TO_ID["truck"],          # truck
    25: UNIFIED_NAME_TO_ID["truck"],          # unknown vehicle -> treat as truck
}

def uavscenes_id_to_unified(mask_id: np.ndarray) -> np.ndarray:
    """
    Convert UAVScenes integer mask [H,W] to unified label IDs [H,W].
    Any ID not in the table falls back to background.
    """
    out = np.full_like(mask_id, UNIFIED_NAME_TO_ID["background"], dtype=np.int64)
    for k, v in UAVSCENES_ID_TO_UNIFIED.items():
        out[mask_id == k] = v
    return out

# =========================
# Utility: norm & resize
# =========================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_tensor(img_pil: Image.Image, normalize: bool = True) -> torch.Tensor:
    x = np.asarray(img_pil, dtype=np.float32) / 255.0
    if x.ndim == 2:  # grayscale
        x = np.stack([x, x, x], axis=-1)
    if normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)

def _resize_pil_keep(img: Image.Image, size: int) -> Tuple[Image.Image, float, float]:
    """
    Resize image to a square canvas of (size, size) while keeping aspect ratio.
    Output:
        img_resized: PIL.Image (size x size)
        sx, sy: scale factors applied to original w,h (new/original)
    """
    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(img_resized, (0, 0))
    sx = new_w / float(w)
    sy = new_h / float(h)
    return canvas, sx, sy

def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), resample=Image.NEAREST)

def project_lidar_to_depth_tile(
    pts_lidar_xyz: np.ndarray,
    K: np.ndarray,
    R_cam_lidar: np.ndarray,
    t_cam_lidar: np.ndarray,
    full_W: int,
    full_H: int,
    x0: int,
    y0: int,
    tile_W: int,
    tile_H: int,
    max_depth: float = 200.0,
) -> np.ndarray:
    """
    Project LiDAR points into a single 2D tile of the image.

    Returns:
        depth_tile: (tile_H, tile_W) float32, 0 where no LiDAR hits.
    """
    if pts_lidar_xyz.size == 0:
        return np.zeros((tile_H, tile_W), dtype=np.float32)

    # LiDAR -> camera
    xyz_l = pts_lidar_xyz[:, :3].T  # (3,N)
    xyz_c = R_cam_lidar @ xyz_l + t_cam_lidar.reshape(3, 1)

    X, Y, Z = xyz_c[0], xyz_c[1], xyz_c[2]

    # Only in front of camera and within range
    valid = (Z > 0.1) & (Z < max_depth)
    if not np.any(valid):
        return np.zeros((tile_H, tile_W), dtype=np.float32)

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy

    # integer pixel coords
    u = u.astype(np.int32)
    v = v.astype(np.int32)

    # restrict to this tile only
    inside = (
        (u >= x0) & (u < x0 + tile_W) &
        (v >= y0) & (v < y0 + tile_H)
    )
    if not np.any(inside):
        return np.zeros((tile_H, tile_W), dtype=np.float32)

    u = u[inside] - x0
    v = v[inside] - y0
    Z = Z[inside]

    depth = np.full((tile_H, tile_W), max_depth, dtype=np.float32)
    # z-buffer: keep nearest hit per pixel
    np.minimum.at(depth, (v, u), Z)
    depth[depth >= max_depth - 1e-4] = 0.0  # 0 = no LiDAR
    return depth

# =========================
# CenterNet target helpers
# =========================

def _gaussian2d(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def _gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size
    a1, b1, c1 = 1, (height + width), width * height * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(0.0, b1 * b1 - 4 * a1 * c1))) / 2
    a2, b2, c2 = 4, 2 * (height + width), (1 - min_overlap) * width * height
    r2 = (b2 + math.sqrt(max(0.0, b2 * b2 - 4 * a2 * c2))) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (height + width), (min_overlap - 1) * width * height
    r3 = (b3 + math.sqrt(max(0.0, b3 * b3 - 4 * a3 * c3))) / (2 * a3) if a3 != 0 else 0.0
    return max(0.0, min(r1, r2, r3))

def _draw_umich_gaussian(heatmap, center, radius, k=1.0):
    diameter = 2 * radius + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    H, W = heatmap.shape[:2]
    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)
    if right <= 0 or bottom <= 0:
        return
    masked = heatmap[y - top:y + bottom, x - left:x + right]
    masked_g = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if masked.size > 0 and masked_g.size > 0:
        np.maximum(masked, masked_g * k, out=masked)

def _boxes_to_centernet_targets(
    img_h: int,
    img_w: int,
    boxes_xywh: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    out_stride: int = 4,
    rotation_mode: str = "none",
    angles_rad: Optional[np.ndarray] = None,
    polys_xy: Optional[np.ndarray] = None,
    min_radius: int = 0,
    radius_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Build CenterNet-style targets for a set of boxes (in image pixels).
    """
    assert rotation_mode in ("none", "angle", "poly8")
    Hs, Ws = img_h // out_stride, img_w // out_stride
    heatmap = np.zeros((num_classes, Hs, Ws), dtype=np.float32)

    inds: List[int] = []
    whs: List[List[float]] = []
    offs: List[List[float]] = []
    reg_mask: List[int] = []
    labs: List[int] = []
    angle_cs: List[List[float]] = []
    poly8s: List[List[float]] = []

    for i, ((x, y, w, h), lab) in enumerate(zip(boxes_xywh, labels)):
        lab = int(lab)
        if lab == 0 or w <= 0 or h <= 0:
            continue
        cx = (x + 0.5 * w) / out_stride
        cy = (y + 0.5 * h) / out_stride
        if cx < 0 or cy < 0 or cx >= Ws or cy >= Hs:
            continue

        r_raw = _gaussian_radius((math.ceil(h / out_stride), math.ceil(w / out_stride)))
        radius = int(max(min_radius, radius_scale * max(0.0, r_raw)))
        _draw_umich_gaussian(heatmap[lab], (cx, cy), radius)

        cx_i, cy_i = int(cx), int(cy)
        inds.append(cy_i * Ws + cx_i)
        whs.append([w / out_stride, h / out_stride])
        offs.append([cx - cx_i, cy - cy_i])
        labs.append(lab)
        reg_mask.append(1)

        if rotation_mode == "angle":
            theta = 0.0 if angles_rad is None else float(angles_rad[i])
            angle_cs.append([math.cos(theta), math.sin(theta)])
        if rotation_mode == "poly8":
            if polys_xy is None:
                poly = [0.0] * 8
            else:
                poly = list(polys_xy[i])
            poly8s.append(poly)

    out: Dict[str, torch.Tensor] = {
        "heatmap": torch.from_numpy(heatmap),
        "ind": torch.tensor(inds, dtype=torch.long) if inds else torch.zeros((0,), dtype=torch.long),
        "wh": torch.tensor(whs, dtype=torch.float32) if whs else torch.zeros((0, 2), dtype=torch.float32),
        "off": torch.tensor(offs, dtype=torch.float32) if offs else torch.zeros((0, 2), dtype=torch.float32),
        "reg_mask": torch.tensor(reg_mask, dtype=torch.uint8) if reg_mask else torch.zeros((0,), dtype=torch.uint8),
        "labels": torch.tensor(labs, dtype=torch.long) if labs else torch.zeros((0,), dtype=torch.long),
        "out_stride": torch.tensor(out_stride, dtype=torch.int64),
    }
    if rotation_mode == "angle":
        out["angle_cossin"] = (
            torch.tensor(angle_cs, dtype=torch.float32)
            if angle_cs
            else torch.zeros((0, 2), dtype=torch.float32)
        )
    if rotation_mode == "poly8":
        out["poly8"] = (
            torch.tensor(poly8s, dtype=torch.float32)
            if poly8s
            else torch.zeros((0, 8), dtype=torch.float32)
        )
    return out

# =========================
# VisDrone Detection
# =========================

class VisDroneDetDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        size: int = 512,
        normalize: bool = True,
        generate_centernet: bool = True,
        out_stride: int = 4,
        rotation_mode: str = "none",
    ):
        super().__init__()
        assert split in ("train", "val", "test")
        self.size = size
        self.normalize = normalize
        self.generate_centernet = generate_centernet
        self.out_stride = out_stride
        self.rotation_mode = rotation_mode

        # schedulable GT heatmap knobs (overwritten by trainer)
        self.hm_min_radius = 1
        self.hm_radius_scale = 1.0

        self.img_dir = os.path.join(root, split, "images")
        self.ann_dir = os.path.join(root, split, "annotations")

        exts = ("*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG")
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.img_dir, e)))
        self.img_paths = sorted(files)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _load_annotations(self, img_path: str) -> Tuple[List[List[float]], List[int]]:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(self.ann_dir, base + ".txt")
        boxes: List[List[float]] = []
        labels: List[int] = []
        if not os.path.isfile(ann_path):
            return boxes, labels

        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    continue
                x, y, w, h = [float(v) for v in parts[:4]]
                cls_id = int(parts[5])
                if cls_id not in VISDRONE_TO_UNIFIED:
                    continue
                u_id = VISDRONE_TO_UNIFIED[cls_id]
                if u_id == UNIFIED_NAME_TO_ID["background"]:
                    continue
                boxes.append([x, y, w, h])
                labels.append(u_id)
        return boxes, labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        W0, H0 = img.size

        img_resized, sx, sy = _resize_pil_keep(img, self.size)
        image_t = _to_tensor(img_resized, normalize=self.normalize)

        boxes, labels = self._load_annotations(img_path)
        target: Dict[str, Any] = {}
        if boxes:
            boxes_arr = np.array(boxes, dtype=np.float32)
            boxes_arr[:, 0] *= sx
            boxes_arr[:, 1] *= sy
            boxes_arr[:, 2] *= sx
            boxes_arr[:, 3] *= sy
            labels_arr = np.array(labels, dtype=np.int64)

            target["boxes_xywh"] = torch.from_numpy(boxes_arr)
            target["labels"] = torch.from_numpy(labels_arr)

            if self.generate_centernet:
                ct = _boxes_to_centernet_targets(
                    img_h=self.size,
                    img_w=self.size,
                    boxes_xywh=boxes_arr,
                    labels=labels_arr,
                    num_classes=len(UNIFIED_CLASSES),
                    out_stride=self.out_stride,
                    rotation_mode=self.rotation_mode,
                    angles_rad=None,
                    polys_xy=None,
                    min_radius=self.hm_min_radius,
                    radius_scale=self.hm_radius_scale,
                )
                target["centernet"] = ct
        else:
            target["boxes_xywh"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            if self.generate_centernet:
                ct = _boxes_to_centernet_targets(
                    img_h=self.size,
                    img_w=self.size,
                    boxes_xywh=np.zeros((0, 4), np.float32),
                    labels=np.zeros((0,), np.int64),
                    num_classes=len(UNIFIED_CLASSES),
                    out_stride=self.out_stride,
                    rotation_mode=self.rotation_mode,
                    angles_rad=None,
                    polys_xy=None,
                    min_radius=self.hm_min_radius,
                    radius_scale=self.hm_radius_scale,
                )
                target["centernet"] = ct

        meta = {
            "img_path": img_path,
            "orig_size": (H0, W0),
            "resize_size": (self.size, self.size),
            "sx": sx,
            "sy": sy,
            "dataset": "visdrone",
        }
        return {"image": image_t, "target": target, "meta": meta}

# =========================
# DOTA + iSAID Detection / Segmentation
# =========================

class DOTAISAIDDataset(Dataset):
    """
    Uses DOTA labelTxt oriented boxes (and optional iSAID semantic masks).
    Images:     root/<split>/images/<base>.png
    DOTA labels:root/<split>/labelTxt-v1.0/labelTxt/<base>.txt
    iSAID masks:root/<split>/images/<base>_instance_color_RGB.png (optional)
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        size: int = 512,
        normalize: bool = True,
        use_segmentation: bool = False,
        use_detection: bool = True,
        generate_centernet: bool = True,
        out_stride: int = 4,
        rotation_mode: str = "angle",
        labeltxt_relpath: str = "labelTxt-v1.0/labelTxt",
        log_unmapped: bool = False,
    ):
        super().__init__()
        assert split in ("train", "val")
        self.size = size
        self.normalize = normalize
        self.use_segmentation = use_segmentation
        self.use_detection = use_detection
        self.generate_centernet = generate_centernet
        self.out_stride = out_stride
        self.rotation_mode = rotation_mode
        self.log_unmapped = log_unmapped

        # schedulable GT heatmap knobs
        self.hm_min_radius = 1
        self.hm_radius_scale = 1.0

        self.img_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, labeltxt_relpath)

        self.img_dir = os.path.join(root, split, "images")
        self.label_dir = os.path.join(root, split, labeltxt_relpath)

        exts = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
        files: List[str] = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.img_dir, e)))

        # >>> PATCH: keep only base RGB tiles, drop iSAID instance / semantic masks
        def _is_base_rgb(path: str) -> bool:
            name = os.path.basename(path)
            bad_tokens = (
                "instance_id_RGB",
                "instance_color_RGB",
                "instance_id_rgb",
                "instance_color_rgb",
                "semantic_RGB",
                "semantic_rgb",
            )
            return not any(tok in name for tok in bad_tokens)

        files = [p for p in files if _is_base_rgb(p)]
        self.img_paths = sorted(files)

    def __len__(self) -> int:
        return len(self.img_paths)

    @staticmethod
    def _parse_dota_label_file(path: str) -> List[Tuple[np.ndarray, str]]:
        """Return list of (poly8, class_name)."""
        recs: List[Tuple[np.ndarray, str]] = []
        if not os.path.isfile(path):
            return recs
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                coords = np.array([float(v) for v in parts[:8]], dtype=np.float32)
                cname = parts[8]
                recs.append((coords, cname))
        return recs

    def _color_to_semantic_ids(self, color_img: np.ndarray, base: str) -> np.ndarray:
        h, w, _ = color_img.shape
        sem = np.zeros((h, w), dtype=np.int64)
        flat = color_img.reshape(-1, 3)
        for rgb, uid in ISAID_COLOR_TO_UNIFIED.items():
            rgb_arr = np.array(rgb, dtype=np.uint8)
            mask = np.all(flat == rgb_arr, axis=1)
            sem.reshape(-1)[mask] = uid
        return sem

    def _poly8_to_cxcywh_theta(self, poly8: np.ndarray) -> Tuple[float, float, float, float, float]:
        poly = poly8.reshape(4, 2)
        xs = poly[:, 0]
        ys = poly[:, 1]
        cx = float(xs.mean())
        cy = float(ys.mean())
        w = float(xs.max() - xs.min())
        h = float(ys.max() - ys.min())
        theta = 0.0  # placeholder; true orientation ignored here
        return cx, cy, w, h, theta

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.img_paths[idx]
        base = os.path.splitext(os.path.basename(img_path))[0]

        img = Image.open(img_path).convert("RGB")
        W0, H0 = img.size

        img_resized, sx, sy = _resize_pil_keep(img, self.size)
        image_t = _to_tensor(img_resized, normalize=self.normalize)

        target: Dict[str, Any] = {}

        # segmentation (iSAID)
        if self.use_segmentation:
            # try color or id-style masks
            cand_paths = [
                os.path.join(self.img_dir, base + "_instance_color_RGB.png"),
                os.path.join(self.img_dir, base + "_instance_id_RGB.png"),
            ]
            color_path = None
            for p in cand_paths:
                if os.path.isfile(p):
                    color_path = p
                    break

            if color_path is not None:
                color = Image.open(color_path).convert("RGB")
                color = _resize_mask(color, self.size)
                color_np = np.asarray(color, dtype=np.uint8)
                sem_ids = self._color_to_semantic_ids(color_np, base)
                target["segmentation"] = torch.from_numpy(sem_ids)

        # detection (DOTA)
        det_recs = (
            self._parse_dota_label_file(os.path.join(self.label_dir, base + ".txt"))
            if self.use_detection
            else []
        )
        boxes_xywh: List[List[float]] = []
        labels: List[int] = []
        angles: List[float] = []
        polys_scaled: List[List[float]] = []

        for poly8, cname in det_recs:
            cname_l = cname.lower()
            if cname_l not in DOTA_NAME_TO_UNIFIED:
                if self.log_unmapped:
                    print(f"[DOTA WARN] '{cname}' not mapped -> skip")
                continue
            uid = DOTA_NAME_TO_UNIFIED[cname_l]
            if uid == UNIFIED_NAME_TO_ID["background"]:
                continue

            poly = poly8.copy().reshape(4, 2)
            poly[:, 0] *= sx
            poly[:, 1] *= sy
            poly_flat = poly.reshape(-1)

            cx, cy, w, h, theta = self._poly8_to_cxcywh_theta(poly)
            boxes_xywh.append([cx - 0.5 * w, cy - 0.5 * h, w, h])
            labels.append(uid)
            angles.append(theta)
            polys_scaled.append(poly_flat.tolist())

        if self.use_detection:
            if boxes_xywh:
                boxes_arr = np.array(boxes_xywh, dtype=np.float32)
                labels_arr = np.array(labels, dtype=np.int64)
                target["boxes_xywh"] = torch.from_numpy(boxes_arr)
                target["labels"] = torch.from_numpy(labels_arr)
            else:
                target["boxes_xywh"] = torch.zeros((0, 4), dtype=torch.float32)
                target["labels"] = torch.zeros((0,), dtype=torch.int64)

            if self.generate_centernet:
                if boxes_xywh:
                    b = np.array(boxes_xywh, np.float32)
                    l = np.array(labels, np.int64)
                    a = np.array(angles, np.float32) if (self.rotation_mode == "angle") else None
                    p = np.array(polys_scaled, np.float32) if (self.rotation_mode == "poly8") else None
                else:
                    b = np.zeros((0, 4), np.float32)
                    l = np.zeros((0,), np.int64)
                    a = None
                    p = None
                ct = _boxes_to_centernet_targets(
                    img_h=self.size,
                    img_w=self.size,
                    boxes_xywh=b,
                    labels=l,
                    num_classes=len(UNIFIED_CLASSES),
                    out_stride=self.out_stride,
                    rotation_mode=self.rotation_mode,
                    angles_rad=a,
                    polys_xy=p,
                    min_radius=self.hm_min_radius,
                    radius_scale=self.hm_radius_scale,
                )
                target["centernet"] = ct

        meta = {
            "img_path": img_path,
            "orig_size": (H0, W0),
            "resize_size": (self.size, self.size),
            "sx": sx,
            "sy": sy,
            "dataset": "dota_isaid",
        }
        return {"image": image_t, "target": target, "meta": meta}

# =========================
# WildUAV segmentation / depth
# =========================

class WildUAVSegDepthDataset(Dataset):
    """
    WildUAV mapping set, using your layout:

      root/
        images/seqXX_img/######.png
        depth/seqXX_depth/depth/######.npy
        semantic_extension/seqXX_semantic/semantic/######.png
        train.txt / val.txt

    task = "seg"  -> RGB + seg
    task = "depth"-> RGB + depth
    task = "both" -> RGB + seg + depth
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        task: str = "seg",
        normalize: bool = True,
        tile_mode: bool = False,
        tile_size: int = 512,
        tile_overlap: float = 0.25,
    ):
        assert task in ("seg", "depth", "both")
        self.root = root
        self.split = split
        self.task = task
        self.normalize = normalize
        self.tile_mode = tile_mode
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.samples = self._build_index()  # list of dicts describing base images

        if len(self.samples) == 0:
            raise ValueError(
                f"[WildUAV] {split} split yielded 0 base images. "
                f"Check {root}/train.txt, {root}/val.txt and directory layout."
            )

        if self.tile_mode:
            self.samples = self._expand_tiles(self.samples)

    # ---- index building ----

    def _parse_line(self, raw: str):
        raw = raw.strip()
        if not raw:
            return None
        # Case 1: "seq00 000123"
        parts = raw.split()
        if len(parts) >= 2 and parts[0].startswith("seq"):
            seq = parts[0]
            frame = parts[1]
            return seq, frame

        # Case 2: any path-like thing; get seqXX and basename
        base = os.path.splitext(os.path.basename(raw))[0]
        m = re.search(r"seq\d+", raw)
        if not m:
            return None
        seq = m.group(0)
        return seq, base

    def _build_index(self):
        split_file = os.path.join(self.root, f"{self.split}.txt")
        seq_frames = []

        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                for line in f:
                    if not line.strip() or line.lstrip().startswith("#"):
                        continue
                    parsed = self._parse_line(line)
                    if parsed is None:
                        continue
                    seq, frame = parsed
                    # zero-pad to 6 digits if needed
                    frame = frame.zfill(6)
                    seq_frames.append((seq, frame))

        # fallback: scan images if txt gives nothing
        if not seq_frames:
            images_root = os.path.join(self.root, "images")
            for seq_dir in sorted(os.listdir(images_root)):
                if not seq_dir.startswith("seq"):
                    continue
                full_seq = seq_dir.split("_")[0]  # e.g. "seq00_img" -> "seq00"
                img_dir = os.path.join(images_root, seq_dir)
                for fn in sorted(os.listdir(img_dir)):
                    if not fn.lower().endswith(".png"):
                        continue
                    frame = os.path.splitext(fn)[0]
                    frame = frame.zfill(6)
                    seq_frames.append((full_seq, frame))

        samples = []
        for seq, frame in seq_frames:
            img_path = os.path.join(
                self.root, "images", f"{seq}_img", f"{frame}.png"
            )
            depth_path = os.path.join(
                self.root, "depth", f"{seq}_depth", "depth", f"{frame}.npy"
            )
            seg_path = os.path.join(
                self.root, "semantic_extension", f"{seq}_semantic", "semantic", f"{frame}.png"
            )

            has_img = os.path.exists(img_path)
            has_depth = os.path.exists(depth_path)
            has_seg = os.path.exists(seg_path)

            if not has_img:
                continue
            if self.task == "seg" and not has_seg:
                continue
            if self.task == "depth" and not has_depth:
                continue
            if self.task == "both" and (not has_seg or not has_depth):
                continue

            samples.append(
                {
                    "seq": seq,
                    "frame": frame,
                    "img": img_path,
                    "depth": depth_path if has_depth else None,
                    "seg": seg_path if has_seg else None,
                }
            )

        print(
            f"[WildUAV] Built {len(samples)} {self.split} samples "
            f"for task={self.task} from root={self.root}"
        )
        return samples

    def _expand_tiles(self, base_samples):
        tiled = []
        stride = int(self.tile_size * (1.0 - self.tile_overlap))
        for s in base_samples:
            # we don't know H,W here yet, so we store a "whole image" sample;
            # you can either look up sizes ahead of time or tile lazily in __getitem__.
            # For now we just carry the base info and tile in __getitem__.
            tiled.append({**s, "tile": None, "stride": stride})
        return tiled

    # ---- Dataset API ----

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]

        # --- load image ---
        img = Image.open(rec["img"]).convert("RGB")
        img = np.array(img, dtype=np.uint8)  # H,W,3

        # --- load segmentation (if needed) ---
        seg = None
        if self.task in ("seg", "both") and rec.get("seg") is not None:
            seg_rgb = np.array(
                Image.open(rec["seg"]).convert("RGB"), dtype=np.uint8
            )  # [H,W,3]
            seg = wild_uav_rgb_to_unified(seg_rgb)  # [H,W] of unified IDs

        # --- load depth (if needed) ---
        depth = None
        if self.task in ("depth", "both") and rec.get("depth") is not None:
            depth = np.load(rec["depth"]).astype(np.float32)  # [H,W]

        # --- basic train-time augmentations ---
        if self.split == "train":
            # 1) horizontal flip (shared across image/seg/depth)
            if random.random() < 0.5:
                img = np.ascontiguousarray(np.fliplr(img))
                if seg is not None:
                    seg = np.ascontiguousarray(np.fliplr(seg))
                if depth is not None:
                    depth = np.ascontiguousarray(np.fliplr(depth))

            # 2) mild brightness / contrast jitter on RGB only
            if random.random() < 0.8:
                img_f = img.astype(np.float32)

                # brightness
                b = 1.0 + 0.2 * (random.random() * 2.0 - 1.0)  # ~[0.8, 1.2]
                img_f = img_f * b

                # contrast (optional, also mild)
                if random.random() < 0.8:
                    c = 1.0 + 0.2 * (random.random() * 2.0 - 1.0)  # ~[0.8, 1.2]
                    mean = img_f.mean(axis=(0, 1), keepdims=True)
                    img_f = (img_f - mean) * c + mean

                img = np.clip(img_f, 0.0, 255.0).astype(np.uint8)

        # --- to tensors ---
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3,H,W]

        target: Dict[str, Any] = {}
        if self.task in ("seg", "both") and seg is not None:
            target["mask"] = torch.from_numpy(seg).long()  # [H,W]
        if self.task in ("depth", "both") and depth is not None:
            depth_t = torch.from_numpy(depth).unsqueeze(0)  # [1,H,W]
            if self.task == "depth":
                target["mask"] = depth_t

        meta = {
            "seq": rec["seq"],
            "frame": rec["frame"],
            "img_path": rec["img"],
            "seg_path": rec.get("seg"),
            "depth_path": rec.get("depth"),
            "dataset": "WildUAV",
        }

        return {"image": img_t, "target": target, "meta": meta}

# =========================
# UAVScenes segmentation / depth
# =========================

def _find_sampleinfos_json(seq_dir: str) -> Optional[str]:
    """
    Find the per-sequence *sampleinfos* JSON for a UAVScenes sequence.

    Typically something like:
      .../interval1_CAM_LIDAR/interval1_AMtown01/samplesinfos_interpolated.json
    """
    pats = [
        os.path.join(seq_dir, "*ample*interpolated*.json"),
        os.path.join(seq_dir, "*SampleInfos*interpolated*.json"),
    ]
    for pat in pats:
        cands = glob.glob(pat)
        if cands:
            cands.sort()
            return cands[0]
    return None

class UAVScenesSegDepthDataset(Dataset):
    """
    UAVScenes camera segmentation (+ optional depth).

    Layout under root (what you described):
      root/
        interval1_CAM_LIDAR/
          interval1_AMtown01/
            interval1_CAM/1658....png (or possibly .jpg)
        interval1_CAM_label/
          interval1_AMtown01/
            interval1_CAM_label_id/1658....png

    We index from label-id PNGs and then:
      - derive scene name from the label path
      - look up a camera image in the corresponding CAM_LIDAR dir
        by *basename* (filename without extension).
    """

    def __init__(
        self,
        root: str,
        intervals: Sequence[str] = ("interval1",),
        task: str = "seg",
        normalize: bool = True,
        tile_mode: bool = False,
        tile_size: int = 512,
        tile_overlap: float = 0.25,
        depth_root: Optional[str] = None,
        split: str = "train",
        depth_source: str = "lidar",   # "lidar" or "terra"
        terra_root: Optional[str] = None,
    ):
        super().__init__()
        assert task in ("seg", "depth")
        if depth_source not in ("lidar", "terra"):
            raise ValueError(f"Unknown depth_source '{depth_source}' (expected 'lidar' or 'terra')")

        self.root = root
        self.intervals = list(intervals)
        self.task = task
        self.normalize = normalize
        self.tile_mode = tile_mode
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.depth_root = depth_root
        self.split = split
        self.depth_source = depth_source

        # If we're using Terra and no explicit root was provided,
        # default to the same layout as the debug script:
        #   <root>/terra_3dmap_pointcloud_mesh
        self.terra_root = terra_root or (
            os.path.join(root, "terra_3dmap_pointcloud_mesh") if depth_source == "terra" else None
        )

        # Per (interval, scene) cache of per-frame K / T from sampleinfos_interpolated.json
        # key: (interval, scene) -> { frame_base : {"K":..., "R_cw":..., "t_cw":..., "W":..., "H":...}, ... }
        self._sampleinfo_index: Dict[Tuple[str, str], Optional[Dict[str, Dict[str, Any]]]] = {}

        self.samples: List[Dict[str, Any]] = []
        self.tiles: List[Dict[str, Any]] = []

        for interval in self.intervals:
            cam_lidar_root = os.path.join(root, f"{interval}_CAM_LIDAR")
            cam_label_root = os.path.join(root, f"{interval}_CAM_label")

            if not os.path.isdir(cam_lidar_root) or not os.path.isdir(cam_label_root):
                print(f"[UAVScenes] skip interval {interval}: "
                      f"missing {cam_lidar_root} or {cam_label_root}")
                continue

            # All label ID PNGs:
            seg_glob = os.path.join(
                cam_label_root, "*", f"{interval}_CAM_label_id", "*.png"
            )
            seg_paths = sorted(glob.glob(seg_glob))
            print(f"[UAVScenes] interval {interval}: found {len(seg_paths)} label PNGs "
                  f"with pattern {seg_glob}")

            kept = 0
            missing_img = 0
            debug_printed = 0  # for a few representative misses

            # Build a cache of camera basenames per scene so we don't
            # re-list the directory 100k+ times
            cam_cache: Dict[str, Dict[str, str]] = {}  # scene -> {basename: full_path}

            for seg_path in seg_paths:
                # seg_path = root/interval1_CAM_label/<scene>/interval1_CAM_label_id/<fname>.png
                parts = seg_path.split(os.sep)
                if len(parts) < 4:
                    continue
                scene = parts[-3]             # <scene>
                fname = parts[-1]             # 1658....png
                base = os.path.splitext(fname)[0]

                # Build camera cache for this scene if we haven't yet
                if scene not in cam_cache:
                    cam_dir = os.path.join(
                        cam_lidar_root, scene, f"{interval}_CAM"
                    )
                    if not os.path.isdir(cam_dir):
                        if debug_printed < 5:
                            print(f"[UAVScenes-debug] scene '{scene}' has no cam dir: {cam_dir}")
                            debug_printed += 1
                        cam_cache[scene] = {}
                    else:
                        try:
                            cam_files = os.listdir(cam_dir)
                        except FileNotFoundError:
                            if debug_printed < 5:
                                print(f"[UAVScenes-debug] cannot list cam dir: {cam_dir}")
                                debug_printed += 1
                            cam_cache[scene] = {}
                        else:
                            mapping = {}
                            for cf in cam_files:
                                b = os.path.splitext(cf)[0]
                                mapping[b] = os.path.join(cam_dir, cf)
                            cam_cache[scene] = mapping
                            if debug_printed < 3:
                                # Show a little preview of what's actually there
                                preview = ", ".join(list(mapping.keys())[:5])
                                print(f"[UAVScenes-debug] cam dir for scene '{scene}': "
                                      f"{cam_dir} (examples: {preview})")
                                debug_printed += 1

                cam_map = cam_cache.get(scene, {})
                img_path = cam_map.get(base, None)
                if img_path is None:
                    missing_img += 1
                    if debug_printed < 8:
                        print(
                            f"[UAVScenes-debug] no camera image for label:\n"
                            f"  seg : {seg_path}\n"
                            f"  base: '{base}'\n"
                            f"  cam_dir: {os.path.join(cam_lidar_root, scene, f'{interval}_CAM')}"
                        )
                        debug_printed += 1
                    continue

                depth_path = None
                if self.depth_root is not None:
                    depth_path = os.path.join(
                        self.depth_root,
                        interval,
                        scene,
                        f"{interval}_CAM",
                        base + ".npy",
                    )
                    if self.task == "depth" and not os.path.isfile(depth_path):
                        # No depth for this sample in depth-only mode
                        continue

                self.samples.append(
                    {
                        "img_path": img_path,
                        "seg_path": seg_path,
                        "depth_path": depth_path,
                        "interval": interval,
                        "scene": scene,
                        "frame": base,
                    }
                )
                kept += 1

            print(
                f"[UAVScenes] interval {interval}: kept {kept} samples, "
                f"{missing_img} labels missing matching camera images"
            )

        print(
            f"[UAVScenes] built {len(self.samples)} samples "
            f"for task={self.task} from root={self.root}"
        )

        # ---- NEW: build per-(interval, scene) sequence index ----
        self._build_sequence_index()

        if self.tile_mode and self.samples:
            self._build_tiles()

    def _build_tiles(self):
        self.tiles = []
        stride = int(self.tile_size * (1.0 - self.tile_overlap))
        for idx, s in enumerate(self.samples):
            img = Image.open(s["img_path"])
            W, H = img.size
            x_starts = list(range(0, max(1, W - self.tile_size + 1), stride))
            y_starts = list(range(0, max(1, H - self.tile_size + 1), stride))
            if x_starts[-1] != W - self.tile_size:
                x_starts.append(max(0, W - self.tile_size))
            if y_starts[-1] != H - self.tile_size:
                y_starts.append(max(0, H - self.tile_size))
            for y0 in y_starts:
                for x0 in x_starts:
                    self.tiles.append(
                        {
                            "sample_idx": idx,
                            "x0": x0,
                            "y0": y0,
                            "W": W,
                            "H": H,
                        }
                    )

    def _build_sequence_index(self) -> None:
        """
        Build mapping from (interval, scene) -> [sample indices] sorted by frame.

        This does NOT change __len__ / __getitem__; it's only a convenience
        structure for sequence-level operations (e.g., Gaussian splatting over
        a whole video).
        """
        from typing import Tuple, List as _List

        self.sequence_index: Dict[Tuple[str, str], _List[int]] = {}

        # Group indices by (interval, scene)
        for idx, s in enumerate(self.samples):
            key = (s["interval"], s["scene"])
            if key not in self.sequence_index:
                self.sequence_index[key] = []
            self.sequence_index[key].append(idx)

        # Sort each sequence by frame (numeric if possible, otherwise lexicographic)
        def _frame_sort_key(frame: str):
            try:
                return float(frame)
            except ValueError:
                return frame

        for key, idxs in self.sequence_index.items():
            idxs.sort(key=lambda i: _frame_sort_key(self.samples[i]["frame"]))

    def get_sequence_keys(self):
        """
        Return list of (interval, scene) keys present in this dataset.

        Example:
            for interval, scene in ds.get_sequence_keys():
                ...
        """
        return list(self.sequence_index.keys())

    def get_sequence_indices(self, interval: str, scene: str):
        """
        Return ordered sample indices for a given (interval, scene).

        Example:
            idxs = ds.get_sequence_indices("interval1", "interval1_AMtown01")
            for i in idxs:
                sample = ds[i]
        """
        return self.sequence_index.get((interval, scene), [])

    def iter_sequences(self):
        """
        Iterate over all sequences as (interval, scene, indices).

        Example:
            for interval, scene, idxs in ds.iter_sequences():
                ...
        """
        for (interval, scene), idxs in self.sequence_index.items():
            yield interval, scene, idxs

    def __len__(self) -> int:
        return len(self.tiles) if self.tile_mode else len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import re
        import glob as _glob

        # ---- 1) Basic sample + tile handling ----
        if self.tile_mode:
            tile = self.tiles[idx]
            s = self.samples[tile["sample_idx"]]
            x0, y0 = tile["x0"], tile["y0"]
        else:
            s = self.samples[idx]
            x0 = y0 = 0

        img = Image.open(s["img_path"]).convert("RGB")
        W0, H0 = img.size
        if self.tile_mode:
            img = img.crop((x0, y0, x0 + self.tile_size, y0 + self.tile_size))
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))
        image_t = torch.from_numpy(img_np)

        depth_t = None
        seg_t = None

        # ------------------------------------------------------------------
        # 2) Depth: precomputed .npy if available, otherwise Terra or LiDAR
        # ------------------------------------------------------------------
        if self.task == "depth":
            # Prefer precomputed depth if present (for future use)
            if s["depth_path"] is not None and os.path.isfile(s["depth_path"]):
                depth = np.load(s["depth_path"]).astype(np.float32)
                if depth.ndim == 2:
                    depth = depth[None, ...]  # [1,H,W]
                elif depth.ndim == 3 and depth.shape[0] != 1:
                    raise ValueError(f"Depth tensor has unexpected shape: {depth.shape} for {s['depth_path']}")

                if self.tile_mode:
                    depth_tile = depth[:, y0 : y0 + self.tile_size, x0 : x0 + self.tile_size]
                else:
                    depth_tile = depth

                depth_t = torch.from_numpy(depth_tile)
            else:
                # For UAVScenes we can do either Terra (via mesh) or LiDAR projection
                import glob as _glob

                if not hasattr(self, "_calib_cache"):
                    self._calib_cache: Dict[Tuple[str, str], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
                if not hasattr(self, "_lidar_index"):
                    self._lidar_index: Dict[Tuple[str, str], Dict[str, str]] = {}
                if not hasattr(self, "_sampleinfo_index"):
                    # may have been set in __init__, but be defensive
                    self._sampleinfo_index: Dict[Tuple[str, str], Optional[Dict[str, Dict[str, Any]]]] = {}

                scene_key = (s["interval"], s["scene"])
                depth_t = None

                # --- 2(a) Per-scene sampleinfos (T4x4 + P3x3) from samplesinfos_interpolated.json ---
                sample_map = self._sampleinfo_index.get(scene_key, None)
                if scene_key not in self._sampleinfo_index:
                    seq_dir = os.path.dirname(os.path.dirname(s["img_path"]))
                    json_path = _find_sampleinfos_json(seq_dir)
                    if json_path is None:
                        self._sampleinfo_index[scene_key] = None
                        sample_map = None
                    else:
                        try:
                            with open(json_path, "r") as f:
                                infos = json.load(f)

                            # Match OriginalSceneName to the scene *without* the interval prefix:
                            #   'interval1_AMtown01' -> 'AMtown01'
                            scene_full = s["scene"]
                            parts = scene_full.split("_", 1)
                            short_scene = parts[1] if len(parts) > 1 else scene_full

                            filtered = [
                                rec for rec in infos
                                if str(rec.get("OriginalSceneName", "")) == short_scene
                            ]
                            if not filtered:
                                filtered = infos  # fallback: use all entries

                            sample_map = {}
                            for rec in filtered:
                                img_name = rec.get("OriginalImageName")
                                if not img_name:
                                    continue
                                base = os.path.splitext(os.path.basename(img_name))[0]

                                try:
                                    T = np.array(rec["T4x4"], dtype=np.float32).reshape(4, 4)
                                    K_s = np.array(rec["P3x3"], dtype=np.float32).reshape(3, 3)
                                except Exception:
                                    continue

                                # T is camera->world (R_wc, t_wc); we want world->camera (R_cw, t_cw)
                                R_wc = T[:3, :3]
                                t_wc = T[:3, 3]
                                R_cw = R_wc.T
                                t_cw = -R_cw @ t_wc

                                W_s = int(rec.get("Width", W0))
                                H_s = int(rec.get("Height", H0))
                                sample_map[base] = {
                                    "K": K_s,
                                    "R_cw": R_cw,
                                    "t_cw": t_cw,
                                    "W": W_s,
                                    "H": H_s,
                                }

                            if not sample_map:
                                sample_map = None

                            self._sampleinfo_index[scene_key] = sample_map
                        except Exception as e:
                            print(f"[UAVScenes] failed to parse sampleinfos for {scene_key}: {e}")
                            self._sampleinfo_index[scene_key] = None
                            sample_map = None
                else:
                    sample_map = self._sampleinfo_index[scene_key]

                # --- 2(b) Static calibration ONLY for LiDAR (unchanged behaviour) ---
                if scene_key not in self._calib_cache:
                    scene_dir_name = s["scene"]  # e.g. 'interval1_AMtown01'
                    parts = scene_dir_name.split("_", 1)
                    short = parts[1] if len(parts) > 1 else scene_dir_name
                    short = short.split("-")[0]  # 'AMtown01-007' -> 'AMtown01'

                    calib = scenename_to_calibration.get(short, None)
                    if calib is None:
                        self._calib_cache[scene_key] = None
                    else:
                        K0 = np.array(calib["camera_intrinsic"], dtype=np.float32).reshape(3, 3)
                        R0 = np.array(calib["camera_ext_R"], dtype=np.float32).reshape(3, 3)
                        t0 = np.array(calib["camera_ext_t"], dtype=np.float32)
                        self._calib_cache[scene_key] = (K0, R0, t0)

                KRT_lidar = self._calib_cache.get(scene_key, None)

                # --- 2(c) Terra mesh depth using per-frame sampleinfos (preferred) ---
                if (
                    self.depth_source == "terra"
                    and self.terra_root is not None
                    and raycast_terra_depth_tile is not None
                ):
                    K_terra = R_terra = t_terra = None

                    # Prefer per-frame K/R/t from sampleinfos_interpolated.json
                    if sample_map is not None:
                        frame_key = s["frame"]  # label / image basename
                        info = sample_map.get(frame_key, None)
                        if info is not None:
                            K_terra = info["K"]
                            R_terra = info["R_cw"]
                            t_terra = info["t_cw"]

                    # Fallback: scene-level calibration if sampleinfos missing
                    if K_terra is None and KRT_lidar is not None:
                        K_terra, R_terra, t_terra = KRT_lidar

                    if K_terra is not None:
                        tile_W = self.tile_size if self.tile_mode else W0
                        tile_H = self.tile_size if self.tile_mode else H0
                        max_depth = getattr(self, "max_depth", 200.0)

                        try:
                            depth_np = raycast_terra_depth_tile(
                                scene_name=s["scene"],
                                terra_root=self.terra_root,
                                K=K_terra,
                                R_cam_world=R_terra,
                                t_cam_world=t_terra,
                                full_W=W0,
                                full_H=H0,
                                x0=x0,
                                y0=y0,
                                tile_W=tile_W,
                                tile_H=tile_H,
                                max_depth=max_depth,
                            )
                        except Exception as e:
                            print(f"[UAVScenes depth] Terra raycast failed for {scene_key}: {e}")
                            depth_np = None

                        if depth_np is not None:
                            depth_t = torch.from_numpy(depth_np.astype(np.float32)).unsqueeze(0)

                # --- 2(d) LiDAR depth: same projection code as before ---
                if depth_t is None and self.depth_source in ("lidar", "terra"):
                    if scene_key not in self._lidar_index:
                        # Build mapping from camera timestamp -> LiDAR file
                        scene_dir = os.path.dirname(os.path.dirname(s["img_path"]))
                        lidar_dir = os.path.join(scene_dir, f"{s['interval']}_LIDAR")

                        mapping: Dict[str, str] = {}
                        if os.path.isdir(lidar_dir):
                            pattern = os.path.join(lidar_dir, "image*_lidar*.txt")
                            for p in _glob.glob(pattern):
                                fname = os.path.basename(p)
                                m = re.match(
                                    r"image(?P<cam_ts>[\d\.]+)_lidar(?P<lidar_ts>[\d\.]+)\.txt",
                                    fname,
                                )
                                if not m:
                                    continue
                                cam_ts = m.group("cam_ts")
                                mapping[cam_ts] = p
                        self._lidar_index[scene_key] = mapping

                    lidar_index = self._lidar_index.get(scene_key, {})
                    cam_ts = str(s["frame"])
                    lidar_path = lidar_index.get(cam_ts, None)
                    if lidar_path is None:
                        # fallback: base of image filename
                        base = os.path.splitext(os.path.basename(s["img_path"]))[0]
                        lidar_path = lidar_index.get(base, None)

                    if (
                        KRT_lidar is not None
                        and lidar_path is not None
                        and os.path.isfile(lidar_path)
                    ):
                        K_lidar, R_lidar, t_lidar = KRT_lidar

                        try:
                            pts = np.loadtxt(lidar_path, dtype=np.float32)
                        except Exception as e:
                            print(f"[UAVScenes depth] failed to read LiDAR {lidar_path}: {e}")
                            pts = None

                        if pts is not None:
                            if pts.ndim == 1:
                                if pts.size < 3:
                                    pts = None
                                else:
                                    pts = pts.reshape(1, -1)

                        if pts is not None:
                            pts_xyz = pts[:, :3]

                            tile_W = self.tile_size if self.tile_mode else W0
                            tile_H = self.tile_size if self.tile_mode else H0
                            max_depth = getattr(self, "max_depth", 200.0)

                            depth_np = project_lidar_to_depth_tile(
                                pts_xyz,
                                K_lidar,
                                R_lidar,
                                t_lidar,
                                W0,
                                H0,
                                x0,
                                y0,
                                tile_W,
                                tile_H,
                                max_depth=max_depth,
                            )
                            depth_t = torch.from_numpy(depth_np).unsqueeze(0)

                # Final safety
                assert depth_t is not None, "Depth map missing (Terra + runtime LiDAR projection failed)"

        # ------------------------------------------------------------------
        # 3) Segmentation (unchanged, except we leave depth alone)
        # ------------------------------------------------------------------
        if os.path.isfile(s["seg_path"]):
            # seg is label-ID PNG: 0..25
            seg_img = Image.open(s["seg_path"]).convert("L")
            if self.tile_mode:
                seg_img = seg_img.crop((x0, y0, x0 + self.tile_size, y0 + self.tile_size))
            seg_id = np.asarray(seg_img, dtype=np.int64)      # [H,W] IDs
            seg_np = uavscenes_id_to_unified(seg_id)          # [H,W] unified
            seg_t = torch.from_numpy(seg_np)

        # ------------------------------------------------------------------
        # 4) Build target + meta
        # ------------------------------------------------------------------
        target: Dict[str, Any] = {}
        if self.task == "seg":
            assert seg_t is not None, "Segmentation mask missing"
            target["mask"] = seg_t
        elif self.task == "depth":
            target["mask"] = depth_t

        meta = {
            "img_path": s["img_path"],
            "seg_path": s["seg_path"],
            "depth_path": s["depth_path"],
            "interval": s["interval"],
            "scene": s["scene"],
            "frame": s["frame"],
            "orig_size": (H0, W0),
            "dataset": "UAVScenes",
        }
        if self.tile_mode:
            meta["tile_xy"] = (x0, y0)
            meta["tile_size"] = self.tile_size

        return {"image": image_t, "target": target, "meta": meta}

# =========================
# COLMAP depth dataset
# =========================

def read_colmap_dense_depth(path: str) -> np.ndarray:
    """
    Read COLMAP dense stereo depth map:
        *.geometric.bin
        *.photometric.bin

    COLMAP dense maps are stored as:
        ASCII header: width&height&channels&
        followed by float32 data.

    Returns:
        depth: [H,W] float32
    """
    with open(path, "rb") as f:
        header = b""
        amp_count = 0

        while amp_count < 3:
            c = f.read(1)
            if not c:
                raise RuntimeError(f"Invalid COLMAP dense file header: {path}")
            header += c
            if c == b"&":
                amp_count += 1

        header_str = header.decode("ascii")
        parts = header_str.split("&")[:3]

        if len(parts) != 3:
            raise RuntimeError(f"Could not parse COLMAP dense header: {header_str}")

        width = int(parts[0])
        height = int(parts[1])
        channels = int(parts[2])

        array = np.fromfile(f, dtype=np.float32)

    expected = width * height * channels
    if array.size != expected:
        raise RuntimeError(
            f"COLMAP dense file size mismatch for {path}: "
            f"header says {width}x{height}x{channels}={expected}, "
            f"but got {array.size} float32 values."
        )

    # COLMAP dense maps use Fortran-style ordering.
    array = array.reshape((width, height, channels), order="F")
    array = np.transpose(array, (1, 0, 2))  # [H,W,C]

    if channels == 1:
        return array[:, :, 0].astype(np.float32)

    return array.astype(np.float32)

class COLMAPDepthDataset(Dataset):
    """
    Simple COLMAP depth dataset.

    Expected minimal layout:

      root/
        flight_20260405_114449/
          images/
            00001632_1775414860317570816.jpg
            ...

          depth_geometric/
            00001632_1775414860317570816.npy
            ...

    Also supports root itself being one flight folder:

      root/
        images/
        depth_geometric/

    Each depth .npy should be float32 [H,W].
    Invalid / hole pixels should be 0, NaN, inf, or <= min_depth.

    Returns the same depth format as UAVScenes/WildUAV:

      {
        "image":  [3,H,W] float in [0,1],
        "target": {"mask": [1,H,W] float depth},
        "meta": {...}
      }
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_dir_name: str = "images",
        depth_dir_name: str = "depth_geometric",
        normalize: bool = False,
        min_depth: float = 1e-3,
        max_depth: float = 10000.0,
        require_depth: bool = True,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.image_dir_name = image_dir_name
        self.depth_dir_name = depth_dir_name
        self.normalize = normalize
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.require_depth = bool(require_depth)

        self.samples = self._build_index()

        if len(self.samples) == 0:
            raise ValueError(
                f"[COLMAPDepthDataset] Found 0 samples under root={root}. "
                f"Expected flight folders containing {image_dir_name}/ and {depth_dir_name}/."
            )

        print(
            f"[COLMAPDepthDataset] built {len(self.samples)} samples "
            f"from root={root}, split={split}"
        )

    def _find_depth_for_image(self, depth_dir: str, image_name: str) -> Optional[str]:
        """
        Supports several naming conventions.

        image:
        00001632_1775414860317570816.jpg

        accepted depth names:
        00001632_1775414860317570816.jpg.geometric.bin
        00001632_1775414860317570816.geometric.bin
        00001632_1775414860317570816.jpg.npy
        00001632_1775414860317570816.npy
        """
        base = os.path.basename(image_name)
        stem = os.path.splitext(base)[0]

        candidates = [
            os.path.join(depth_dir, base + ".geometric.bin"),
            os.path.join(depth_dir, stem + ".geometric.bin"),
            os.path.join(depth_dir, base + ".bin"),
            os.path.join(depth_dir, stem + ".bin"),

            # Optional converted formats if we make them later.
            os.path.join(depth_dir, stem + ".npy"),
            os.path.join(depth_dir, base + ".npy"),
            os.path.join(depth_dir, base + ".geometric.npy"),
            os.path.join(depth_dir, stem + ".geometric.npy"),
        ]

        for p in candidates:
            if os.path.isfile(p):
                return p

        return None

    def _build_index(self) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []

        if not os.path.isdir(self.root):
            raise ValueError(f"[COLMAPDepthDataset] root does not exist: {self.root}")

        # Allow root itself to be one flight folder.
        if os.path.isdir(os.path.join(self.root, self.image_dir_name)):
            flight_dirs = [self.root]
        else:
            flight_dirs = [
                os.path.join(self.root, d)
                for d in sorted(os.listdir(self.root))
                if os.path.isdir(os.path.join(self.root, d))
            ]

        exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

        for flight_dir in flight_dirs:
            flight = os.path.basename(flight_dir.rstrip("/\\"))
            image_dir = os.path.join(flight_dir, self.image_dir_name)
            depth_dir = os.path.join(flight_dir, self.depth_dir_name)

            if not os.path.isdir(image_dir):
                continue
            if not os.path.isdir(depth_dir):
                continue

            image_paths: List[str] = []
            for e in exts:
                image_paths.extend(glob.glob(os.path.join(image_dir, e)))
            image_paths = sorted(image_paths)

            missing_depth = 0

            for img_path in image_paths:
                image_name = os.path.basename(img_path)
                depth_path = self._find_depth_for_image(depth_dir, image_name)

                if depth_path is None:
                    missing_depth += 1
                    if self.require_depth:
                        continue

                samples.append(
                    {
                        "flight": flight,
                        "img": img_path,
                        "depth": depth_path,
                        "image_name": image_name,
                    }
                )

            print(
                f"[COLMAPDepthDataset] flight={flight} "
                f"images={len(image_paths)} kept={len(samples)} missing_depth={missing_depth}"
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _timestamp_from_name(name: str) -> Optional[int]:
        """
        Tries to parse timestamp from names like:
          00001632_1775414860317570816.jpg
        """
        stem = os.path.splitext(os.path.basename(name))[0]
        parts = stem.split("_")
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except Exception:
                return None
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]

        img_pil = Image.open(rec["img"]).convert("RGB")
        W0, H0 = img_pil.size
        img_t = _to_tensor(img_pil, normalize=self.normalize)

        if rec["depth"] is None:
            raise RuntimeError(f"Missing depth for image: {rec['img']}")

        depth_path = rec["depth"]
        ext = os.path.splitext(depth_path)[1].lower()

        if depth_path.lower().endswith(".bin"):
            depth = read_colmap_dense_depth(depth_path).astype(np.float32)
        elif ext == ".npy":
            depth = np.load(depth_path).astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported COLMAP depth format: {depth_path}")

        if depth.ndim == 3:
            # Allow [H,W,1] or [1,H,W], collapse to [H,W].
            if depth.shape[-1] == 1:
                depth = depth[..., 0]
            elif depth.shape[0] == 1:
                depth = depth[0]
            else:
                raise RuntimeError(
                    f"Unexpected depth shape {depth.shape}: {rec['depth']}"
                )

        if depth.ndim != 2:
            raise RuntimeError(
                f"Expected depth [H,W], got {depth.shape}: {rec['depth']}"
            )

        valid = (
            np.isfinite(depth)
            & (depth > self.min_depth)
            & (depth < self.max_depth)
        )
        depth_clean = np.where(valid, depth, 0.0).astype(np.float32)
        depth_t = torch.from_numpy(depth_clean).unsqueeze(0)  # [1,H,W]

        valid_count = int(valid.sum())
        valid_ratio = float(valid_count) / float(valid.size) if valid.size > 0 else 0.0

        if valid_count > 0:
            valid_vals = depth[valid]
            depth_min = float(np.min(valid_vals))
            depth_max = float(np.max(valid_vals))
            depth_mean = float(np.mean(valid_vals))
            depth_median = float(np.median(valid_vals))
        else:
            depth_min = 0.0
            depth_max = 0.0
            depth_mean = 0.0
            depth_median = 0.0

        meta = {
            "dataset": "colmap",
            "flight": rec["flight"],
            "image_name": rec["image_name"],
            "timestamp_ns": self._timestamp_from_name(rec["image_name"]),
            "img_path": rec["img"],
            "depth_path": rec["depth"],
            "orig_size": (H0, W0),
            "depth_valid_count": valid_count,
            "depth_valid_ratio": valid_ratio,
            "depth_min": depth_min,
            "depth_max": depth_max,
            "depth_mean": depth_mean,
            "depth_median": depth_median,
        }

        return {
            "image": img_t,
            "target": {"mask": depth_t},
            "meta": meta,
        }


# =========================
# Multi-task wrapper + collates
# =========================

class CombinedMultiTaskDataset(Dataset):
    """
    Balanced multi-dataset wrapper.

    - mode="train":
        * Each dataset contributes the same number of samples per epoch
          (n_per = ceil(sum(L_i) / D)).
        * Large datasets are randomly under-sampled.
        * Small datasets are randomly over-sampled.
        * The per-dataset subsets/duplicates are re-drawn each epoch by
          calling `reshuffle_epoch()`.

    - mode="eval":
        * Datasets are simply concatenated in order, no balancing.

    NOTE: For this wrapper, set DataLoader(shuffle=False); we handle the
    mixing and randomness inside the dataset itself.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        mode: str = "train",
        seed: Optional[int] = None,
    ):
        assert len(datasets) >= 1
        self.datasets = list(datasets)
        self.lengths = [len(d) for d in self.datasets]
        self.num_datasets = len(self.datasets)
        self.mode = mode
        self.rng = random.Random(seed)

        # list[(ds_idx, local_idx)] describing this epoch's ordering
        self.epoch_indices: List[Tuple[int, int]] = []

        if self.mode == "train":
            self._build_balanced_epoch()
        else:
            self._build_concat_indices()

    # ---------- index builders ----------

    def _build_concat_indices(self) -> None:
        """Eval mode: just concatenate all datasets in order."""
        epoch_indices: List[Tuple[int, int]] = []
        for ds_idx, L in enumerate(self.lengths):
            for j in range(L):
                epoch_indices.append((ds_idx, j))
        self.epoch_indices = epoch_indices

    def _build_balanced_epoch(self) -> None:
        """
        Train mode: each dataset contributes the same number of samples n_per.

        n_per = ceil(sum(L_i) / D)
        - if L_i >= n_per: random subset without replacement
        - if L_i <  n_per: random oversampling (repeated shuffled passes)
        """
        D = self.num_datasets
        L_sum = sum(self.lengths)
        if D == 0 or L_sum == 0:
            self.epoch_indices = []
            return

        n_per = math.ceil(L_sum / D)  # target per dataset

        per_ds_indices: List[List[int]] = []
        for ds_idx, L in enumerate(self.lengths):
            if L == 0:
                per_ds_indices.append([])
                continue

            if L >= n_per:
                # random subset without replacement
                idxs = list(range(L))
                self.rng.shuffle(idxs)
                idxs = idxs[:n_per]
            else:
                # need to oversample: repeated shuffled passes
                reps = math.ceil(n_per / L)
                base = list(range(L))
                all_idx: List[int] = []
                for _ in range(reps):
                    self.rng.shuffle(base)
                    all_idx.extend(base)
                idxs = all_idx[:n_per]
            per_ds_indices.append(idxs)

        # Interleave datasets to keep batches mixed:
        # [d0_i0, d1_i0, ..., dD-1_i0, d0_i1, ...]
        epoch_indices: List[Tuple[int, int]] = []
        for k in range(n_per):
            for ds_idx in range(D):
                ds_list = per_ds_indices[ds_idx]
                if k < len(ds_list):
                    epoch_indices.append((ds_idx, ds_list[k]))

        self.epoch_indices = epoch_indices

    # ---------- public API ----------

    def reshuffle_epoch(self, seed: Optional[int] = None) -> None:
        """
        Rebuild the balanced index list for a new epoch (train mode only).

        Call this once per epoch in your training loop, before creating the
        DataLoader iterator.
        """
        if self.mode != "train":
            return
        if seed is not None:
            self.rng.seed(seed)
        self._build_balanced_epoch()

    def __len__(self) -> int:
        return len(self.epoch_indices)

    def __getitem__(self, idx: int):
        ds_idx, local_idx = self.epoch_indices[idx]
        return self.datasets[ds_idx][local_idx]

def collate_det(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    targets = [b["target"] for b in batch]
    meta = [b["meta"] for b in batch]
    return images, targets, meta

# Target spatial size for segmentation / depth batches
SEG_TARGET_SIZE = 512  # change if you want a different size


def _resize_depth_masked(
    depth: torch.Tensor,
    target_size: int,
    valid_min_depth: float = 1e-6,
    valid_weight_threshold: float = 0.25,
) -> torch.Tensor:
    """
    Resize a depth map while preserving invalid holes.

    depth: [1,H,W]
      - valid pixels: finite and > valid_min_depth
      - invalid / holes: 0

    Returns:
      resized depth [1,target_size,target_size], with invalid pixels set to 0.

    Why:
      Direct bilinear interpolation turns holes into fake positive depth.
      This resizes depth*valid and valid separately, then divides.
    """
    if depth.ndim != 3 or depth.shape[0] != 1:
        raise RuntimeError(f"Expected depth [1,H,W], got {depth.shape}")

    d = depth.float().unsqueeze(0)  # [1,1,H,W]
    valid = (torch.isfinite(d) & (d > valid_min_depth)).float()

    d_safe = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_weighted = d_safe * valid

    d_sum = F.interpolate(
        d_weighted,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    w_sum = F.interpolate(
        valid,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )

    out = d_sum / w_sum.clamp_min(1e-6)
    out = torch.where(
        w_sum >= valid_weight_threshold,
        out,
        torch.zeros_like(out),
    )

    return out[0]  # [1,target_size,target_size]


def collate_seg_or_depth(batch, target_size: int = SEG_TARGET_SIZE):
    """
    Collate function for segmentation or depth batches.

    - Resizes all images to [3, target_size, target_size]
    - Resizes all masks:
        * seg:   [target_size, target_size] long
        * depth: [1, target_size, target_size] float

    Depth resizing preserves invalid holes as zero.
    """
    images = []
    masks = []
    meta = []

    for b in batch:
        img = b["image"]             # [C,H,W]
        m = b["target"]["mask"]      # [H,W] seg or [1,H,W] depth

        # --- resize image ---
        img4 = img.unsqueeze(0)  # [1,C,H,W]
        img4 = F.interpolate(
            img4,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        img_resized = img4[0]  # [C,target_size,target_size]

        # --- resize mask/depth ---
        if m.ndim == 2:
            # segmentation mask [H,W]
            m4 = m.unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]
            m4 = F.interpolate(
                m4,
                size=(target_size, target_size),
                mode="nearest",
            )
            m_resized = m4[0, 0].long()  # [target_size,target_size]

        elif m.ndim == 3 and m.shape[0] == 1:
            # depth map [1,H,W], preserving holes/invalid pixels as 0
            m_resized = _resize_depth_masked(
                m,
                target_size=target_size,
                valid_min_depth=1e-6,
                valid_weight_threshold=0.25,
            )

        else:
            raise RuntimeError(
                f"Unexpected mask shape {m.shape} in collate_seg_or_depth"
            )

        images.append(img_resized)
        masks.append(m_resized)
        meta.append(b["meta"])

    images = torch.stack(images, dim=0)  # [B,3,H,W]
    masks = torch.stack(masks, dim=0)    # [B,H,W] or [B,1,H,W]

    return images, masks, meta
