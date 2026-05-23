#!/usr/bin/env python3
from __future__ import annotations
"""
visualize_handoff_models.py

One-stop visualization script for the current handoff models:
  1) segmentation teacher
  2) depth teacher
  3) segmentation split model, with ViT and/or no-ViT variants
  4) depth split model, with ViT and/or no-ViT variants

Inputs:
  - default project datasets: UAVScenes segmentation/depth and/or COLMAP depth
  - arbitrary image folder: runs models on images only, no GT panels

This script does not train or modify checkpoints.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from datasets_multitask import (
    COLMAPDepthDataset,
    UAVScenesSegDepthDataset,
    UNIFIED_CLASSES,
    collate_seg_or_depth,
)
from unified_depth_teacher import UnifiedDepthTeacher
from unified_seg_teacher import UnifiedSegmentationTeacher


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CollateSegOrDepth:
    def __init__(self, target_size: int) -> None:
        self.target_size = int(target_size)

    def __call__(self, batch):
        return collate_seg_or_depth(batch, target_size=self.target_size)


class FolderImageDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int, exts: Tuple[str, ...]) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.paths: List[Path] = []
        for ext in exts:
            self.paths.extend(sorted(self.image_dir.rglob(f"*{ext}")))
        self.paths = sorted(set(self.paths))
        if not self.paths:
            raise RuntimeError(f"No images found in {self.image_dir} with extensions {exts}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        ten = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        dummy = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        meta = {"path": str(path), "filename": path.name, "dataset": "folder"}
        return ten, dummy, meta


def folder_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    metas = [b[2] for b in batch]
    return images, targets, metas


def load_ckpt_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt["model_state"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError(f"Unsupported checkpoint type from {path}: {type(ckpt)}")


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state.keys()):
        return state
    return {k[len("module."):]: v if k.startswith("module.") else v for k, v in state.items()}


def load_model_compatible(model: torch.nn.Module, path: str, device: torch.device, label: str) -> None:
    state = strip_module_prefix(load_ckpt_state(path, device))
    own = model.state_dict()
    copied = 0
    skipped: List[str] = []
    for k, v in state.items():
        if k in own and own[k].shape == v.shape:
            own[k].copy_(v)
            copied += 1
        else:
            skipped.append(k)
    model.load_state_dict(own, strict=True)
    print(f"[{label}] loaded compatible tensors={copied}, skipped={len(skipped)} from {path}")
    if skipped[:10]:
        print(f"[{label}] first skipped keys: {skipped[:10]}")
    if copied == 0:
        raise RuntimeError(f"No compatible tensors copied for {label} from {path}")


def ensure_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        bad = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(f"[{name}] non-finite output: bad_values={bad}/{x.numel()}")


def pred_depth_to_meters(pred_raw: torch.Tensor, prediction_units: str, max_depth_meters: float) -> torch.Tensor:
    if prediction_units == "normalized":
        return pred_raw * float(max_depth_meters)
    if prediction_units == "meters":
        return pred_raw
    raise ValueError(f"Unknown prediction_units={prediction_units}")


def image_tensor_to_uint8(img: torch.Tensor) -> np.ndarray:
    arr = img.detach().cpu().float().clamp(0, 1).numpy()
    arr = np.transpose(arr, (1, 2, 0))
    return (arr * 255.0 + 0.5).astype(np.uint8)


def make_palette(num_classes: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    palette = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)
    return palette


PALETTE = make_palette(len(UNIFIED_CLASSES))


def seg_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.int64)
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(PALETTE))
    out[valid] = PALETTE[mask[valid]]
    return out


def normalize_heatmap(values: np.ndarray, valid: Optional[np.ndarray] = None, *, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = valid & np.isfinite(arr)
    if vmin is None or vmax is None:
        if np.any(valid):
            vals = arr[valid]
            lo = float(np.percentile(vals, 2.0)) if vmin is None else float(vmin)
            hi = float(np.percentile(vals, 98.0)) if vmax is None else float(vmax)
        else:
            lo, hi = 0.0, 1.0
    else:
        lo, hi = float(vmin), float(vmax)
    if not np.isfinite(lo):
        lo = 0.0
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    # blue -> cyan -> green -> yellow -> red
    x = (norm * 4.0).astype(np.float32)
    r = np.clip(np.minimum(np.maximum(x - 2.0, 0.0), 1.0) + np.minimum(np.maximum(x - 3.0, 0.0), 1.0), 0, 1)
    g = np.clip(np.minimum(np.maximum(x - 1.0, 0.0), 1.0) - np.minimum(np.maximum(x - 3.0, 0.0), 1.0), 0, 1)
    b = np.clip(1.0 - np.minimum(np.maximum(x - 1.0, 0.0), 1.0), 0, 1)
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (r * 255).astype(np.uint8)
    rgb[..., 1] = (g * 255).astype(np.uint8)
    rgb[..., 2] = (b * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def normalize_gray(values: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    else:
        valid = valid & np.isfinite(arr)
    out = np.zeros((*arr.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = arr[valid]
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255.0 + 0.5).astype(np.uint8)
    out[..., 0] = gray
    out[..., 1] = gray
    out[..., 2] = gray
    out[~valid] = 0
    return out


def mask_to_rgb(valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*valid.shape, 3), dtype=np.uint8)
    out[valid] = 255
    return out


def add_title(img: np.ndarray, title: str, height: int = 30) -> np.ndarray:
    pil = Image.fromarray(img)
    canvas = Image.new("RGB", (pil.width, pil.height + height), (20, 20, 20))
    canvas.paste(pil, (0, height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), title, fill=(255, 255, 255))
    return np.asarray(canvas)


def make_grid(panels: List[Tuple[str, np.ndarray]], cols: int) -> Image.Image:
    titled = [add_title(img, title) for title, img in panels]
    h = max(x.shape[0] for x in titled)
    w = max(x.shape[1] for x in titled)
    padded = []
    for x in titled:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:x.shape[0], :x.shape[1]] = x
        padded.append(canvas)
    rows = int(np.ceil(len(padded) / float(cols)))
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, x in enumerate(padded):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = x
    return Image.fromarray(grid)


def safe_stem(meta: Dict[str, Any], idx: int) -> str:
    for key in ("filename", "image_path", "rgb_path", "path"):
        if key in meta:
            return Path(str(meta[key])).stem[:80]
    return f"sample_{idx:04d}"


def depth_stats(pred_m: np.ndarray, gt_m: Optional[np.ndarray], eps: float) -> Dict[str, float]:
    pred = np.asarray(pred_m, dtype=np.float32)
    valid_pred = np.isfinite(pred)
    stats: Dict[str, float] = {}
    if np.any(valid_pred):
        vals = pred[valid_pred]
        stats.update(
            pred_min_m=float(np.min(vals)),
            pred_p01_m=float(np.percentile(vals, 1)),
            pred_p50_m=float(np.percentile(vals, 50)),
            pred_p99_m=float(np.percentile(vals, 99)),
            pred_max_m=float(np.max(vals)),
            pred_mean_m=float(np.mean(vals)),
            pred_std_m=float(np.std(vals)),
        )
    if gt_m is not None:
        gt = np.asarray(gt_m, dtype=np.float32)
        valid = np.isfinite(gt) & (gt > eps)
        stats["valid_ratio"] = float(np.mean(valid)) if valid.size else 0.0
        if np.any(valid):
            abs_err = np.abs(pred[valid] - gt[valid])
            stats.update(
                mae_m=float(np.mean(abs_err)),
                rmse_m=float(np.sqrt(np.mean((pred[valid] - gt[valid]) ** 2))),
                rel=float(np.mean(abs_err / np.maximum(gt[valid], eps))),
            )
    return stats


def seg_stats(pred: np.ndarray, gt: Optional[np.ndarray]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if gt is not None:
        valid = (gt >= 0) & (gt < len(UNIFIED_CLASSES))
        if np.any(valid):
            stats["pixel_acc"] = float(np.mean(pred[valid] == gt[valid]))
    unique, counts = np.unique(pred, return_counts=True)
    top = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:10]
    stats["pred_unique_classes"] = int(len(unique))
    stats["pred_top_classes"] = [
        {"id": int(i), "name": UNIFIED_CLASSES[int(i)] if 0 <= int(i) < len(UNIFIED_CLASSES) else str(i), "pixels": int(c)}
        for i, c in top
    ]
    return stats


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------


def build_seg_model(args: argparse.Namespace, device: torch.device, ckpt: str, *, split: bool, label: str) -> UnifiedSegmentationTeacher:
    model = UnifiedSegmentationTeacher(
        num_classes=len(UNIFIED_CLASSES),
        model_type=args.model_type,
        use_split=split,
        split_frontend_type="custom_stage0" if split else "legacy",
        custom_spatial_ch=args.custom_spatial_ch,
        custom_token_dim=args.custom_token_dim,
        custom_patch_size=args.custom_patch_size,
        custom_keep_ratio=args.custom_keep_ratio,
        custom_vit_depth_enc=args.custom_vit_depth_enc,
        custom_vit_depth_dec=args.custom_vit_depth_dec,
        custom_vit_heads=args.custom_vit_heads,
    ).to(device)
    load_model_compatible(model, ckpt, device, label)
    model.eval()
    return model


def build_depth_model(args: argparse.Namespace, device: torch.device, ckpt: str, *, split: bool, label: str) -> UnifiedDepthTeacher:
    model = UnifiedDepthTeacher(
        model_type=args.model_type,
        use_split=split,
        split_frontend_type="custom_stage0" if split else "legacy",
        custom_spatial_ch=args.custom_spatial_ch,
        custom_token_dim=args.custom_token_dim,
        custom_patch_size=args.custom_patch_size,
        custom_keep_ratio=args.custom_keep_ratio,
        custom_vit_depth_enc=args.custom_vit_depth_enc,
        custom_vit_depth_dec=args.custom_vit_depth_dec,
        custom_vit_heads=args.custom_vit_heads,
        depth_head_mid_channels=(None if args.depth_mid_channels <= 0 else args.depth_mid_channels),
        positive_depth=True,
    ).to(device)
    load_model_compatible(model, ckpt, device, label)
    model.eval()
    return model


def set_split_variant(model: torch.nn.Module, use_vit: bool, args: argparse.Namespace) -> None:
    if not hasattr(model, "backbone") or not hasattr(model.backbone, "set_custom_split_use_vit"):
        raise RuntimeError("Requested split variant on a model without custom split support.")
    model.backbone.set_custom_split_use_vit(bool(use_vit))
    if hasattr(model.backbone, "set_custom_split_fixed_keep"):
        model.backbone.set_custom_split_fixed_keep(bool(args.fixed_keep), int(args.fixed_keep_seed))
    if hasattr(model.backbone, "custom_split"):
        cs = model.backbone.custom_split
        cs.keep_ratio = float(args.custom_keep_ratio)
        if hasattr(cs, "vit") and hasattr(cs.vit, "keep_ratio"):
            cs.vit.keep_ratio = float(args.custom_keep_ratio)


# -----------------------------------------------------------------------------
# Dataset builders
# -----------------------------------------------------------------------------


def make_subset(ds: Dataset, num_samples: int, stride: int) -> Dataset:
    n = len(ds)
    if n == 0:
        raise RuntimeError("Dataset has zero samples.")
    if num_samples <= 0 or num_samples >= n:
        idxs = list(range(0, n, max(1, stride)))
    else:
        idxs = np.linspace(0, n - 1, num=num_samples, dtype=np.int64).tolist()
    return Subset(ds, idxs)


def build_sources(args: argparse.Namespace) -> List[Tuple[str, Dataset, str, int]]:
    """Returns list of (source_name, dataset, task_type, target_size)."""
    sources: List[Tuple[str, Dataset, str, int]] = []
    has_seg = bool(args.seg_teacher_ckpt or args.seg_split_ckpt)
    has_depth = bool(args.depth_teacher_ckpt or args.depth_split_ckpt)

    if args.input_dir:
        ds = FolderImageDataset(
            args.input_dir,
            image_size=args.folder_image_size,
            exts=tuple(args.image_exts),
        )
        sources.append(("folder_images", ds, "folder", args.folder_image_size))
        return sources

    if args.dataset_mode in ("uavscenes", "mixed"):
        if args.data_root is None:
            raise ValueError("--data-root is required for UAVScenes dataset sources.")
        if has_seg and args.include_seg_dataset:
            seg_ds = UAVScenesSegDepthDataset(
                root=args.data_root,
                intervals=args.uav_intervals,
                task="seg",
                split=args.val_split,
            )
            sources.append(("uavscenes_seg", seg_ds, "seg", args.seg_size))
        if has_depth and args.include_depth_dataset:
            depth_ds = UAVScenesSegDepthDataset(
                root=args.data_root,
                intervals=args.uav_intervals,
                task="depth",
                normalize=False,
                tile_mode=False,
                depth_root=args.uav_depth_root,
                depth_source=args.uav_depth_source,
                terra_root=args.uav_terra_root,
                split=args.val_split,
            )
            sources.append(("uavscenes_depth", depth_ds, "depth", args.depth_size))

    if args.dataset_mode in ("colmap", "mixed"):
        if has_depth and args.include_depth_dataset:
            if args.colmap_root is None:
                raise ValueError("--colmap-root is required for COLMAP depth source.")
            col_ds = COLMAPDepthDataset(
                root=args.colmap_root,
                split="val",
                normalize=False,
                min_depth=args.colmap_min_depth,
                max_depth=args.colmap_max_depth,
            )
            sources.append(("colmap_depth", col_ds, "depth", args.depth_size))

    if not sources:
        raise RuntimeError("No visualization sources were built. Check ckpts, --dataset-mode, and include flags.")
    return sources


# -----------------------------------------------------------------------------
# Inference + visualization
# -----------------------------------------------------------------------------


@torch.no_grad()
def run_seg(model: UnifiedSegmentationTeacher, images: torch.Tensor, variant_name: str) -> torch.Tensor:
    x = UnifiedSegmentationTeacher.normalize_midas(images)
    logits, _ = model(x, return_backbone=False)
    if logits.shape[-2:] != images.shape[-2:]:
        logits = F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)
    ensure_finite(variant_name, logits)
    return logits


@torch.no_grad()
def run_depth(model: UnifiedDepthTeacher, images: torch.Tensor, args: argparse.Namespace, variant_name: str) -> torch.Tensor:
    x = UnifiedDepthTeacher.normalize_midas(images)
    pred, _ = model(x, return_backbone=False)
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if pred.shape[-2:] != images.shape[-2:]:
        pred = F.interpolate(pred, size=images.shape[-2:], mode="bilinear", align_corners=False)
    ensure_finite(variant_name, pred)
    return pred_depth_to_meters(pred, args.prediction_units, args.max_depth_meters)


def run_source(
    *,
    args: argparse.Namespace,
    source_name: str,
    dataset: Dataset,
    task_type: str,
    target_size: int,
    models: Dict[str, torch.nn.Module],
    device: torch.device,
    out_dir: Path,
) -> None:
    subset = make_subset(dataset, args.num_samples, args.sample_stride)
    if task_type == "folder":
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"), collate_fn=folder_collate)
    else:
        loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"), collate_fn=CollateSegOrDepth(target_size))

    src_dir = out_dir / source_name
    src_dir.mkdir(parents=True, exist_ok=True)
    summary: List[Dict[str, Any]] = []

    for i, (images, targets, metas) in enumerate(loader):
        if i >= args.num_samples:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        meta = metas[0] if isinstance(metas, list) else {}
        rgb = image_tensor_to_uint8(images[0])
        panels: List[Tuple[str, np.ndarray]] = [("RGB", rgb)]
        sample_stats: Dict[str, Any] = {"index": i, "source": source_name, "task_type": task_type, "meta": meta}

        gt_seg_np: Optional[np.ndarray] = None
        gt_depth_np: Optional[np.ndarray] = None
        valid_depth: Optional[np.ndarray] = None
        if task_type == "seg" and targets.ndim == 3:
            gt_seg_np = targets[0].detach().cpu().numpy().astype(np.int64)
            panels.append(("GT seg", seg_to_rgb(gt_seg_np)))
        if task_type == "depth" and targets.ndim == 4 and targets.shape[1] == 1:
            gt_depth_np = targets[0, 0].detach().cpu().numpy().astype(np.float32)
            valid_depth = np.isfinite(gt_depth_np) & (gt_depth_np > args.eps)
            panels.append(("GT depth", normalize_heatmap(gt_depth_np, valid_depth, vmin=0.0, vmax=args.max_depth_meters)))
            if args.show_local_norm:
                panels.append(("GT depth local", normalize_gray(gt_depth_np, valid_depth)))
            panels.append(("Valid depth", mask_to_rgb(valid_depth)))

        # Segmentation teacher.
        if "seg_teacher" in models:
            logits = run_seg(models["seg_teacher"], images, "seg_teacher")
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
            panels.append(("Seg teacher", seg_to_rgb(pred)))
            sample_stats["seg_teacher"] = seg_stats(pred, gt_seg_np)

        # Segmentation split variants.
        if "seg_split" in models:
            seg_model = models["seg_split"]
            variants = []
            if args.split_variants in ("vit", "both"):
                variants.append(("vit", True))
            if args.split_variants in ("novit", "both"):
                variants.append(("no_vit", False))
            for vname, use_vit in variants:
                set_split_variant(seg_model, use_vit, args)
                logits = run_seg(seg_model, images, f"seg_split_{vname}")
                pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int64)
                panels.append((f"Seg split {vname}", seg_to_rgb(pred)))
                sample_stats[f"seg_split_{vname}"] = seg_stats(pred, gt_seg_np)

        # Depth teacher.
        if "depth_teacher" in models:
            pred_m = run_depth(models["depth_teacher"], images, args, "depth_teacher")
            arr = pred_m[0, 0].detach().cpu().numpy().astype(np.float32)
            panels.append(("Depth teacher", normalize_heatmap(arr, np.isfinite(arr), vmin=0.0, vmax=args.max_depth_meters)))
            if args.show_local_norm:
                panels.append(("Depth teacher local", normalize_gray(arr, np.isfinite(arr))))
            if gt_depth_np is not None:
                panels.append(("Teacher err", normalize_heatmap(np.abs(arr - gt_depth_np), valid_depth)))
            sample_stats["depth_teacher"] = depth_stats(arr, gt_depth_np, args.eps)

        # Depth split variants.
        if "depth_split" in models:
            depth_model = models["depth_split"]
            variants = []
            if args.split_variants in ("vit", "both"):
                variants.append(("vit", True))
            if args.split_variants in ("novit", "both"):
                variants.append(("no_vit", False))
            depth_variant_arrays: Dict[str, np.ndarray] = {}
            for vname, use_vit in variants:
                set_split_variant(depth_model, use_vit, args)
                pred_m = run_depth(depth_model, images, args, f"depth_split_{vname}")
                arr = pred_m[0, 0].detach().cpu().numpy().astype(np.float32)
                depth_variant_arrays[vname] = arr
                panels.append((f"Depth split {vname}", normalize_heatmap(arr, np.isfinite(arr), vmin=0.0, vmax=args.max_depth_meters)))
                if args.show_local_norm:
                    panels.append((f"Depth split {vname} local", normalize_gray(arr, np.isfinite(arr))))
                if gt_depth_np is not None:
                    panels.append((f"Split {vname} err", normalize_heatmap(np.abs(arr - gt_depth_np), valid_depth)))
                sample_stats[f"depth_split_{vname}"] = depth_stats(arr, gt_depth_np, args.eps)
            if "vit" in depth_variant_arrays and "no_vit" in depth_variant_arrays:
                diff = np.abs(depth_variant_arrays["vit"] - depth_variant_arrays["no_vit"])
                panels.append(("Depth |vit-no_vit|", normalize_heatmap(diff, np.isfinite(diff))))
                sample_stats["depth_split_vit_vs_no_vit"] = {
                    "mae_m": float(np.mean(diff[np.isfinite(diff)])),
                    "max_m": float(np.max(diff[np.isfinite(diff)])),
                }

        grid = make_grid(panels, cols=args.grid_cols)
        stem = safe_stem(meta, i)
        out_png = src_dir / f"{i:04d}_{stem}.png"
        out_json = src_dir / f"{i:04d}_{stem}.json"
        grid.save(out_png)
        with open(out_json, "w") as f:
            json.dump(sample_stats, f, indent=2, default=str)
        summary.append(sample_stats)
        print(f"[{source_name}] wrote {out_png}")

    with open(src_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


# -----------------------------------------------------------------------------
# Args / main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize teacher and split handoff models.")

    # Checkpoints. Any subset can be supplied.
    p.add_argument("--seg-teacher-ckpt", type=str, default=None)
    p.add_argument("--depth-teacher-ckpt", type=str, default=None)
    p.add_argument("--seg-split-ckpt", type=str, default=None)
    p.add_argument("--depth-split-ckpt", type=str, default=None)

    # Dataset or folder input.
    p.add_argument("--dataset-mode", type=str, default="mixed", choices=["uavscenes", "colmap", "mixed"], help="Used unless --input-dir is provided.")
    p.add_argument("--input-dir", type=str, default=None, help="Optional folder of RGB images. If set, no dataset GT is used.")
    p.add_argument("--image-exts", type=str, nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"])
    p.add_argument("--folder-image-size", type=int, default=512)

    # UAVScenes / COLMAP.
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--uav-intervals", type=str, nargs="+", default=["interval1"])
    p.add_argument("--val-split", type=str, default="val")
    p.add_argument("--uav-depth-root", type=str, default=None)
    p.add_argument("--uav-depth-source", type=str, default="terra", choices=["lidar", "terra"])
    p.add_argument("--uav-terra-root", type=str, default=None)
    p.add_argument("--colmap-root", type=str, default=None)
    p.add_argument("--colmap-min-depth", type=float, default=1e-3)
    p.add_argument("--colmap-max-depth", type=float, default=200.0)
    p.add_argument("--include-seg-dataset", action="store_true", default=True)
    p.add_argument("--no-include-seg-dataset", dest="include_seg_dataset", action="store_false")
    p.add_argument("--include-depth-dataset", action="store_true", default=True)
    p.add_argument("--no-include-depth-dataset", dest="include_depth_dataset", action="store_false")

    # Model config.
    p.add_argument("--model-type", type=str, default="DPT_Hybrid", choices=["DPT_Hybrid"])
    p.add_argument("--seg-size", type=int, default=512)
    p.add_argument("--depth-size", type=int, default=512)
    p.add_argument("--prediction-units", type=str, default="normalized", choices=["normalized", "meters"])
    p.add_argument("--max-depth-meters", type=float, default=200.0)
    p.add_argument("--depth-mid-channels", type=int, default=96, help="Use 96 for current safe depth head checkpoints; use 0 for wrapper default.")

    # Split config. Defaults match the 64-channel split variant.
    p.add_argument("--custom-spatial-ch", type=int, default=64)
    p.add_argument("--custom-token-dim", type=int, default=128)
    p.add_argument("--custom-patch-size", type=int, default=4)
    p.add_argument("--custom-keep-ratio", type=float, default=0.25)
    p.add_argument("--custom-vit-depth-enc", type=int, default=2)
    p.add_argument("--custom-vit-depth-dec", type=int, default=4)
    p.add_argument("--custom-vit-heads", type=int, default=8)
    p.add_argument("--split-variants", type=str, default="both", choices=["vit", "novit", "both"])
    p.add_argument("--fixed-keep", action="store_true", default=True)
    p.add_argument("--random-keep", dest="fixed_keep", action="store_false")
    p.add_argument("--fixed-keep-seed", type=int, default=12345)

    # Runtime/output.
    p.add_argument("--num-samples", type=int, default=12)
    p.add_argument("--sample-stride", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="viz/handoff_models")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--grid-cols", type=int, default=3)
    p.add_argument("--show-local-norm", action="store_true", default=True)
    p.add_argument("--no-local-norm", dest="show_local_norm", action="store_false")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Config] device={device}")
    print(f"[Config] split_variants={args.split_variants} custom_spatial_ch={args.custom_spatial_ch} token_dim={args.custom_token_dim} keep_ratio={args.custom_keep_ratio}")

    models: Dict[str, torch.nn.Module] = {}
    if args.seg_teacher_ckpt:
        models["seg_teacher"] = build_seg_model(args, device, args.seg_teacher_ckpt, split=False, label="seg_teacher")
    if args.depth_teacher_ckpt:
        models["depth_teacher"] = build_depth_model(args, device, args.depth_teacher_ckpt, split=False, label="depth_teacher")
    if args.seg_split_ckpt:
        models["seg_split"] = build_seg_model(args, device, args.seg_split_ckpt, split=True, label="seg_split")
    if args.depth_split_ckpt:
        models["depth_split"] = build_depth_model(args, device, args.depth_split_ckpt, split=True, label="depth_split")

    if not models:
        raise RuntimeError("Provide at least one checkpoint argument.")

    with open(out_dir / "class_names.json", "w") as f:
        json.dump({str(i): name for i, name in enumerate(UNIFIED_CLASSES)}, f, indent=2)

    sources = build_sources(args)
    for source_name, dataset, task_type, target_size in sources:
        print(f"[Source] {source_name}: task={task_type} samples={len(dataset)} target_size={target_size}")
        run_source(
            args=args,
            source_name=source_name,
            dataset=dataset,
            task_type=task_type,
            target_size=target_size,
            models=models,
            device=device,
            out_dir=out_dir,
        )

    print(f"[Done] wrote visualizations under {out_dir}")


if __name__ == "__main__":
    main()
