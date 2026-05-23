#!/usr/bin/env python3
"""
Batch inference script for unified multitask models.
Runs detection, depth, and segmentation on all images in an input folder,
saving results into per-image subfolders under the output directory.
"""

import argparse
import cv2
import numpy as np
import torch
import os
import json
from pathlib import Path
from datetime import datetime

# --- Custom Model Imports ---
from datasets_multitask import UNIFIED_CLASSES, UNIFIED_NAME_TO_ID
from unified_depth_teacher import UnifiedDepthTeacher
from unified_seg_teacher import UnifiedSegmentationTeacher

# --- Configuration ---
MODEL_TYPE = "DPT_Large"
DEPTH_CKPT = "/home/albus/Documents/AureliusIndustries/demoNovember2025/centurion-aurelius-industries/ai-model/checkpoints/best_depth_teacher.pth"
SEG_CKPT = "/home/albus/Documents/AureliusIndustries/demoNovember2025/centurion-aurelius-industries/ai-model/checkpoints/best_seg_teacher.pth"

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

SEGMENTATION_INTEREST_CLASSES = [
    "building", "roof", "roof-transparent", "truck", "bus", "car",
    "person", "van", "tricycle", "small-vehicle", "large-vehicle", "plane"
]
INTEREST_IDS = [UNIFIED_NAME_TO_ID[name] for name in SEGMENTATION_INTEREST_CLASSES if name in UNIFIED_NAME_TO_ID]
MIN_SEG_PIXEL_AREA = 100

# --- Color Configuration ---
CUSTOM_COLOR_MAP_RGB = {
    "background":         (0, 0, 0),
    "car":                (0, 255, 0),
    "truck":              (50, 205, 50),
    "bus":                (34, 139, 34),
    "van":                (0, 255, 127),
    "bridge":             (139, 69, 19),
    "swimming-pool":      (30, 144, 255),
    "road":               (255, 255, 150),
    "roof":               (0, 0, 255),
    "roof-transparent":   (0, 100, 255),
    "field-green":        (112, 194, 33),
    "field-wild":         (34, 139, 34),
    "building":           (0, 165, 255),
    "person":             (0, 0, 255),
}

CUSTOM_COLOR_MAP_BGR = {k: v[::-1] for k, v in CUSTOM_COLOR_MAP_RGB.items()}


# --- Helper Functions ---
def _build_seg_palette() -> np.ndarray:
    num_classes = len(UNIFIED_CLASSES)
    rng = np.random.default_rng(1234)
    colors = rng.integers(0, 255, size=(num_classes, 3), dtype=np.uint8)
    for name, color in CUSTOM_COLOR_MAP_RGB.items():
        if name in UNIFIED_NAME_TO_ID:
            idx = UNIFIED_NAME_TO_ID[name]
            colors[idx] = np.array(color, dtype=np.uint8)
    return colors


SEG_COLORS = _build_seg_palette()


def seg_to_color(seg_tensor: torch.Tensor) -> np.ndarray:
    seg_np = seg_tensor.detach().cpu().numpy().astype(np.int64)
    seg_np = np.clip(seg_np, 0, len(UNIFIED_CLASSES) - 1)
    return SEG_COLORS[seg_np]


def depth_to_rgb(depth_tensor: torch.Tensor) -> np.ndarray:
    d = depth_tensor.detach().cpu().float()
    d_min, d_max = float(d.min()), float(d.max())
    if d_max <= d_min + 1e-6:
        d_norm = torch.zeros_like(d)
    else:
        d_norm = (d - d_min) / (d_max - d_min + 1e-6)
    gray = (d_norm.numpy() * 255.0).round().astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def overlay_images(base: np.ndarray, overlay: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    out = (1.0 - alpha) * base.astype(np.float32) + alpha * overlay.astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _infer_seg_num_classes_from_ckpt(ckpt_path: str, default: int = 1) -> int:
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
        keys_to_check = ["seg_head.classifier.weight"]
        keys_to_check.extend([k for k in state_dict.keys() if k.endswith("classifier.weight")])
        for k in keys_to_check:
            if k in state_dict:
                return int(state_dict[k].shape[0])
    except Exception as e:
        print(f"⚠️  Could not infer seg classes: {e}")
    return default


def extract_boxes_from_segmentation(seg_mask: np.ndarray) -> tuple:
    categories = []
    bounding_boxes = []
    for class_id in INTEREST_IDS:
        binary_mask = np.where(seg_mask == class_id, 255, 0).astype(np.uint8)
        if not np.any(binary_mask):
            continue
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_SEG_PIXEL_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            categories.append(UNIFIED_CLASSES[class_id])
            bounding_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
    return categories, bounding_boxes


# --- Model Loading ---
def load_models():
    """Load all three unified models once."""
    print(f"📦 Loading Unified Multitask Models on {DEVICE}...")

    print("  Loading Depth Model...")
    depth_model = UnifiedDepthTeacher(model_type=MODEL_TYPE, hub_repo="intel-isl/MiDaS")
    depth_model.to(DEVICE)
    depth_ckpt = torch.load(DEPTH_CKPT, map_location=DEVICE)
    depth_model.load_state_dict(depth_ckpt.get("model_state", depth_ckpt.get("state_dict", depth_ckpt)), strict=False)
    depth_model.eval()

    print("  Loading Segmentation Model...")
    seg_num_classes = _infer_seg_num_classes_from_ckpt(SEG_CKPT)
    seg_model = UnifiedSegmentationTeacher(
        num_classes=seg_num_classes,
        model_type=MODEL_TYPE,
        hub_repo="intel-isl/MiDaS",
    )
    seg_model.to(DEVICE)
    seg_ckpt = torch.load(SEG_CKPT, map_location=DEVICE)
    seg_model.load_state_dict(seg_ckpt.get("model_state", seg_ckpt.get("state_dict", seg_ckpt)), strict=False)
    seg_model.eval()

    print("✅ All Models Loaded!\n")
    return depth_model, seg_model


# --- Per-Image Inference ---
def run_inference(frame: np.ndarray, depth_model, seg_model) -> tuple:
    """Run all three inference heads on a single BGR frame."""
    orig_h, orig_w = frame.shape[:2]

    # Letterbox padding to square
    scale = min(IMG_SIZE / orig_w, IMG_SIZE / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = cv2.resize(frame, (new_w, new_h))
    img_padded = np.full((IMG_SIZE, IMG_SIZE, 3), 128, dtype=np.uint8)
    dw, dh = (IMG_SIZE - new_w) // 2, (IMG_SIZE - new_h) // 2
    img_padded[dh:dh + new_h, dw:dw + new_w] = img_resized

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    x = x.to(memory_format=torch.channels_last)
    x_norm = (x - 0.5) / 0.5

    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    with torch.no_grad():
        with torch.cuda.stream(stream1):
            depth_pred, _ = depth_model(x_norm, return_backbone=False)
        with torch.cuda.stream(stream2):
            seg_logits, _ = seg_model(x_norm, return_backbone=False)

    torch.cuda.synchronize()

    depth_map = (depth_pred[0, 0] if depth_pred.ndim == 4 else depth_pred[0]) * 255.0
    seg_mask_tensor = seg_logits.argmax(dim=1)[0]
    seg_mask_np = seg_mask_tensor.cpu().numpy().astype(np.uint8)

    depth_rgb = depth_to_rgb(depth_map)
    seg_rgb = seg_to_color(seg_mask_tensor)
    img_rgb_uint8 = img_rgb.astype(np.uint8)

    categories, bounding_boxes = extract_boxes_from_segmentation(seg_mask_np)

    # Detection visualisation
    detection_vis = img_rgb_uint8.copy()
    for i, bbox in enumerate(bounding_boxes):
        bx1, by1 = bbox["x"], bbox["y"]
        bx2, by2 = bx1 + bbox["width"], by1 + bbox["height"]
        label = categories[i]
        cls_id = UNIFIED_NAME_TO_ID.get(label, 0)
        color = tuple(int(c) for c in SEG_COLORS[cls_id])
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)
        cv2.rectangle(detection_vis, (bx1, by1), (bx2, by2), color, 2)
        (t_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(detection_vis, (bx1, by1 - 20), (bx1 + t_w, by1), color, -1)
        cv2.putText(detection_vis, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Segmentation class summary (>1% coverage)
    unique, counts = np.unique(seg_mask_np, return_counts=True)
    seg_class_summary = {}
    for u, c in zip(unique, counts):
        if u < len(UNIFIED_CLASSES):
            pct = (c / seg_mask_np.size) * 100
            if pct > 1.0:
                seg_class_summary[UNIFIED_CLASSES[u]] = f"{pct:.1f}%"

    composite_vis = overlay_images(depth_rgb, seg_rgb, alpha=0.5)

    # Crop letterbox padding and restore original resolution
    def _to_bgr_original(img_rgb_arr: np.ndarray) -> np.ndarray:
        cropped = img_rgb_arr[dh:dh + new_h, dw:dw + new_w]
        return cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), (orig_w, orig_h))

    frames_dict = {
        "original":    _to_bgr_original(img_rgb_uint8),
        "detection":   _to_bgr_original(detection_vis),
        "segmentation": _to_bgr_original(seg_rgb),
        "depth":       _to_bgr_original(depth_rgb),
        "composite":   _to_bgr_original(composite_vis),
    }

    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "original_shape": (orig_h, orig_w),
        "categories": categories,
        "bounding_boxes": bounding_boxes,
        "detection_count": len(categories),
        "segmentation_summary": seg_class_summary,
    }

    return frames_dict, metadata


# --- Batch Runner ---
def collect_images(input_dir: Path) -> list[Path]:
    """Collect all supported image paths from input_dir (non-recursive)."""
    images = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    return images


def process_folder(input_dir: Path, output_dir: Path, depth_model, seg_model) -> None:
    """Run inference on every image in input_dir and save results to output_dir."""
    image_paths = collect_images(input_dir)

    if not image_paths:
        print(f"⚠️  No supported images found in '{input_dir}'.")
        print(f"   Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    print(f"📂 Found {len(image_paths)} image(s) in '{input_dir}'")
    print(f"📁 Results will be saved to '{output_dir}'\n")

    batch_summary = []
    failed = []

    # record avg inference time
    latencies = []

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] 🖼  Processing: {img_path.name}")

        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"   ❌ Could not read image, skipping.\n")
            failed.append(img_path.name)
            continue

        print(f"   Shape: {frame.shape}")

        # Create a dedicated subfolder named after the image stem
        img_out_dir = output_dir / img_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        start_time.record()
        try:
            frames_dict, metadata = run_inference(frame, depth_model, seg_model)
        except Exception as e:
            print(f"   ❌ Inference failed: {e}\n")
            failed.append(img_path.name)
            continue

        end_time.record()
        torch.cuda.synchronize()
        latency_ms = start_time.elapsed_time(end_time)
        latencies.append(latency_ms)

        # Save visualisation frames
        for key, img in frames_dict.items():
            out_path = img_out_dir / f"{key}.jpg"
            cv2.imwrite(str(out_path), img)

        # Save per-image metadata
        metadata["image_name"] = img_path.name
        meta_path = img_out_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Console summary
        print(f"   ✅ Objects detected: {metadata['detection_count']}")
        if metadata["categories"]:
            print(f"   📦 Categories: {', '.join(sorted(set(metadata['categories'])))}")
        if metadata["segmentation_summary"]:
            summary_str = ", ".join(
                f"{k} {v}" for k, v in metadata["segmentation_summary"].items()
            )
            print(f"   🗺  Seg coverage: {summary_str}")
        print(f"   💾 Saved → {img_out_dir}\n")

        batch_summary.append({
            "image": img_path.name,
            "output_folder": img_path.stem,
            **{k: v for k, v in metadata.items() if k != "bounding_boxes"},
        })

    # Save a top-level batch summary JSON
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_timestamp": datetime.utcnow().isoformat(),
                "input_dir": str(input_dir),
                "total_images": len(image_paths),
                "processed": len(batch_summary),
                "failed": failed,
                "results": batch_summary,
                "average_inference_time_ms": sum(latencies) / len(latencies) if latencies else None,
            },
            f,
            indent=2,
        )

    print("=" * 60)
    print(f"✅ Batch complete!")
    print(f"   Inference time: {sum(latencies) / len(latencies):.1f} ms/image (GPU)")
    print(f"   Processed : {len(batch_summary)} / {len(image_paths)}")
    if failed:
        print(f"   Failed     : {len(failed)} — {', '.join(failed)}")
    print(f"   Summary    : {summary_path}")


# --- Entry Point ---
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch inference: detection, depth, and segmentation on a folder of images."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the folder containing input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Root output folder. Each image gets its own subfolder inside it. "
            "Defaults to '<input_dir>_inference_results'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir: Path = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"❌ '{input_dir}' is not a valid directory.")
        return

    output_dir: Path = (
        args.output_dir.resolve()
        if args.output_dir
        else input_dir.parent / f"{input_dir.name}_inference_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_model, seg_model = load_models()
    process_folder(input_dir, output_dir, depth_model, seg_model)


if __name__ == "__main__":
    main()