#!/usr/bin/env python3
"""
Simple inference script for testing unified multitask models
Runs detection, depth, and segmentation on a sample image
"""

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
MODEL_TYPE = "DPT_Hybrid"  # or "DPT_Large", "DPT_Base", etc.
DEPTH_CKPT = "/home/albus/Documents/AureliusIndustries/demoMay2026/ian-new-models/best_depth_teacher.pth"
SEG_CKPT = "/home/albus/Documents/AureliusIndustries/demoMay2026/ian-new-models/best_seg_teacher.pth"

IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_IMAGE_PATH = "sample_image.jpg"
OUTPUT_DIR = "inference_results"

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
        print(f"⚠️ Could not infer seg classes: {e}")
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

def load_models():
    """Load all three unified models"""
    print(f"📦 Loading Unified Multitask Models on {DEVICE}...")
    
    # Depth Model
    print("  Loading Depth Model...")
    depth_model = UnifiedDepthTeacher(model_type=MODEL_TYPE, hub_repo="intel-isl/MiDaS")
    depth_model.to(DEVICE)
    depth_ckpt = torch.load(DEPTH_CKPT, map_location=DEVICE)
    depth_model.load_state_dict(depth_ckpt.get("model_state", depth_ckpt.get("state_dict", depth_ckpt)), strict=False)
    depth_model.eval()
    
    # Segmentation Model
    print("  Loading Segmentation Model...")
    seg_num_classes = _infer_seg_num_classes_from_ckpt(SEG_CKPT)
    seg_model = UnifiedSegmentationTeacher(
        num_classes=seg_num_classes, 
        model_type=MODEL_TYPE, 
        hub_repo="intel-isl/MiDaS"
    )
    seg_model.to(DEVICE)
    seg_ckpt = torch.load(SEG_CKPT, map_location=DEVICE)
    seg_model.load_state_dict(seg_ckpt.get("model_state", seg_ckpt.get("state_dict", seg_ckpt)), strict=False)
    seg_model.eval()
    
    print("✅ All Models Loaded!")
    return depth_model, seg_model

def run_inference(frame, depth_model, seg_model):
    """Run inference on a single frame"""
    orig_h, orig_w = frame.shape[:2]
    
    # Letterbox padding
    target_size = IMG_SIZE
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img_resized = cv2.resize(frame, (new_w, new_h))
    img_padded = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    dw, dh = (target_size - new_w) // 2, (target_size - new_h) // 2
    img_padded[dh:dh+new_h, dw:dw+new_w] = img_resized
    
    # Prepare input
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_np = img_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    x_norm = (x - 0.5) / 0.5
    
    print("🔄 Running inference...")
    with torch.no_grad():
        # Depth
        depth_pred, _ = depth_model(x_norm, return_backbone=False)
        depth_map = (depth_pred[0, 0] if depth_pred.ndim == 4 else depth_pred[0]) * 255.0
        
        # Segmentation
        seg_logits, _ = seg_model(x_norm, return_backbone=False)
        seg_mask_tensor = seg_logits.argmax(dim=1)[0]
        seg_mask_np = seg_mask_tensor.cpu().numpy().astype(np.uint8)
    
    # Process results
    depth_rgb = depth_to_rgb(depth_map)
    seg_rgb = seg_to_color(seg_mask_tensor)
    img_rgb_uint8 = img_rgb.astype(np.uint8)
    
    # Extract boxes
    categories, bounding_boxes = extract_boxes_from_segmentation(seg_mask_np)
    
    # Create detection visualization
    detection_vis = img_rgb_uint8.copy()
    for i, bbox in enumerate(bounding_boxes):
        bx1, by1 = bbox['x'], bbox['y']
        bx2, by2 = bx1 + bbox['width'], by1 + bbox['height']
        label = categories[i]
        cls_id = UNIFIED_NAME_TO_ID.get(label, 0)
        color = tuple(int(c) for c in SEG_COLORS[cls_id])  # RGB
        
        # Dynamic Text Color
        luminance = (0.299 * color[0]) + (0.587 * color[1]) + (0.114 * color[2])
        text_color = (0, 0, 0) if luminance > 127 else (255, 255, 255)
        
        cv2.rectangle(detection_vis, (bx1, by1), (bx2, by2), color, 2)
        (t_w, t_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(detection_vis, (bx1, by1 - 20), (bx1 + t_w, by1), color, -1)
        cv2.putText(detection_vis, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    # Class summary
    unique, counts = np.unique(seg_mask_np, return_counts=True)
    seg_class_summary = {}
    for u, c in zip(unique, counts):
        if u < len(UNIFIED_CLASSES):
            class_name = UNIFIED_CLASSES[u]
            percentage = (c / seg_mask_np.size) * 100
            if percentage > 1.0: 
                seg_class_summary[class_name] = f"{percentage:.1f}%"
    
    # Composite
    composite_vis = overlay_images(depth_rgb, seg_rgb, alpha=0.5)
    
    # Helper to convert back to original size BGR
    def process_final(img):
        valid = img[dh:dh+new_h, dw:dw+new_w]
        return cv2.resize(cv2.cvtColor(valid, cv2.COLOR_RGB2BGR), (orig_w, orig_h))
    
    frames_dict = {
        "detection": process_final(detection_vis),
        "segmentation": process_final(seg_rgb),
        "depth": process_final(depth_rgb),
        "composite": process_final(composite_vis),
        "original": process_final(img_rgb_uint8)
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

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if sample image exists
    if not os.path.exists(SAMPLE_IMAGE_PATH):
        print(f"❌ Error: {SAMPLE_IMAGE_PATH} not found!")
        return
    
    print(f"📷 Loading sample image: {SAMPLE_IMAGE_PATH}")
    frame = cv2.imread(SAMPLE_IMAGE_PATH)
    
    if frame is None:
        print(f"❌ Error: Could not read {SAMPLE_IMAGE_PATH}")
        return
    
    print(f"   Image shape: {frame.shape}")
    
    # Load models
    depth_model, seg_model = load_models()
    
    # Run inference
    frames_dict, metadata = run_inference(frame, depth_model, seg_model)
    
    # Save results
    print("\n💾 Saving results...")
    for key, img in frames_dict.items():
        output_path = os.path.join(OUTPUT_DIR, f"result_{key}.jpg")
        cv2.imwrite(output_path, img)
        print(f"   ✅ {output_path}")
    
    # Save metadata
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ {metadata_path}")
    
    # Print results
    print("\n📊 Inference Results:")
    print(f"   Timestamp: {metadata['timestamp']}")
    if metadata['categories']:
        print(f"   Categories: {', '.join(set(metadata['categories']))}")
    print(f"\n   Segmentation Summary:")
    for cls_name, percentage in metadata['segmentation_summary'].items():
        print(f"      {cls_name}: {percentage}")
    
    print(f"\n✅ Inference complete! Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
