#!/usr/bin/env python3
"""
Interactive viewer for batch inference results.
Shows original / segmentation / depth side-by-side for each result folder.

Controls:
  →  /  d  /  Space   : next image
  ←  /  a             : previous image
  1                   : show original only
  2                   : show segmentation only
  3                   : show depth only
  4                   : show composite only
  0                   : show all (default)
  f                   : toggle fullscreen
  s                   : save current view as PNG
  q  /  Esc           : quit
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ── Layout ────────────────────────────────────────────────────────────────────
PANEL_LABELS  = ["original", "segmentation", "depth"]   # filenames (no ext)
WINDOW_TITLE  = "Inference Viewer"
PANEL_GAP     = 6        # px between panels
LABEL_HEIGHT  = 36       # px header bar per panel
FONT          = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE    = 0.75
FONT_THICK    = 2
BG_COLOR      = (18, 18, 18)
LABEL_BG      = (35, 35, 35)
LABEL_FG      = (220, 220, 220)
COUNTER_FG    = (140, 140, 140)
MAX_WIDTH     = 1800     # cap total window width
MAX_HEIGHT    = 960      # cap total window height


# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_result_folders(root: Path) -> list[Path]:
    """Return sorted subfolders that contain at least one panel image."""
    folders = sorted(
        p for p in root.iterdir()
        if p.is_dir() and any((p / f"{label}.jpg").exists() for label in PANEL_LABELS)
    )
    return folders


def load_panels(folder: Path, mode: int) -> list[tuple[str, np.ndarray]]:
    """
    Load panel images for the given display mode.
    mode 0 = all three side-by-side
    mode 1/2/3/4 = single panel (original/seg/depth/composite)
    """
    single_map = {1: "original", 2: "segmentation", 3: "depth", 4: "composite"}
    labels = [single_map[mode]] if mode in single_map else PANEL_LABELS

    panels = []
    for label in labels:
        path = folder / f"{label}.jpg"
        if not path.exists():
            # create a placeholder
            img = np.full((480, 640, 3), 40, dtype=np.uint8)
            cv2.putText(img, f"{label} not found", (20, 240),
                        FONT, 0.8, (100, 100, 100), 2)
        else:
            img = cv2.imread(str(path))
            if img is None:
                img = np.full((480, 640, 3), 40, dtype=np.uint8)
        panels.append((label, img))
    return panels


def make_label_bar(text: str, width: int) -> np.ndarray:
    bar = np.full((LABEL_HEIGHT, width, 3), LABEL_BG, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICK)
    x = max(0, (width - tw) // 2)
    y = (LABEL_HEIGHT + th) // 2
    cv2.putText(bar, text.upper(), (x, y), FONT, FONT_SCALE, LABEL_FG, FONT_THICK, cv2.LINE_AA)
    return bar


def build_frame(folder: Path, idx: int, total: int, mode: int,
                target_h: int, target_w_each: int) -> np.ndarray:
    """Compose the full display frame."""
    panels = load_panels(folder, mode)
    n = len(panels)

    strips = []
    for label, img in panels:
        # Resize to target height keeping aspect ratio, then pad/crop to target_w_each
        oh, ow = img.shape[:2]
        scale  = target_h / oh
        nw     = int(ow * scale)
        resized = cv2.resize(img, (nw, target_h))

        if nw >= target_w_each:
            # centre-crop width
            x0 = (nw - target_w_each) // 2
            resized = resized[:, x0:x0 + target_w_each]
        else:
            # pad sides
            pad = np.full((target_h, target_w_each - nw, 3), BG_COLOR[0], dtype=np.uint8)
            resized = np.hstack([resized, pad])

        bar   = make_label_bar(label, target_w_each)
        strip = np.vstack([bar, resized])
        strips.append(strip)

    # Join panels with gap
    gap_col = np.full((target_h + LABEL_HEIGHT, PANEL_GAP, 3), BG_COLOR[0], dtype=np.uint8)
    joined  = strips[0]
    for s in strips[1:]:
        joined = np.hstack([joined, gap_col, s])

    # Status bar at bottom
    status_h = 38
    total_w  = joined.shape[1]
    status   = np.full((status_h, total_w, 3), BG_COLOR[0], dtype=np.uint8)

    folder_name = folder.name
    meta_path   = folder / "metadata.json"
    extra = ""
    if meta_path.exists():
        try:
            meta  = json.loads(meta_path.read_text())
            count = meta.get("detection_count", "?")
            cats  = list(set(meta.get("categories", [])))[:5]
            extra = f"  |  {count} detections"
            if cats:
                extra += f"  [{', '.join(cats)}]"
        except Exception:
            pass

    left_txt  = f"  {folder_name}{extra}"
    right_txt = f"{idx + 1} / {total}  [0]all [1-4]single [f]fullscreen [s]save [q]quit  "

    cv2.putText(status, left_txt,  (8, 26), FONT, 0.55, LABEL_FG,    1, cv2.LINE_AA)
    (rw, _), _ = cv2.getTextSize(right_txt, FONT, 0.55, 1)
    cv2.putText(status, right_txt, (total_w - rw - 4, 26), FONT, 0.55, COUNTER_FG, 1, cv2.LINE_AA)

    # separator line
    cv2.line(status, (0, 0), (total_w, 0), (60, 60, 60), 1)

    return np.vstack([joined, status])


def compute_layout(n_panels: int) -> tuple[int, int]:
    """Return (panel_image_height, panel_image_width) fitting MAX dimensions."""
    usable_w = MAX_WIDTH  - PANEL_GAP * (n_panels - 1)
    w_each   = usable_w  // n_panels
    # assume roughly 4:3 source images → derive height
    h_each   = int(w_each * 0.75)
    if h_each + LABEL_HEIGHT > MAX_HEIGHT - 38:
        h_each = MAX_HEIGHT - 38 - LABEL_HEIGHT
        w_each = int(h_each / 0.75)
    return h_each, w_each


# ── Main ──────────────────────────────────────────────────────────────────────
def run_viewer(root: Path) -> None:
    folders = collect_result_folders(root)
    if not folders:
        print(f"❌  No result folders found in '{root}'.")
        sys.exit(1)

    print(f"📂  Found {len(folders)} result folder(s) in '{root}'")
    print("     Controls: ←/→ navigate  |  1-4 single view  |  0 all  |  f fullscreen  |  s save  |  q quit")

    idx        = 0
    mode       = 0      # 0=all, 1=orig, 2=seg, 3=depth, 4=composite
    fullscreen = False

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, MAX_WIDTH, MAX_HEIGHT)

    # pre-compute layout for 3-panel and 1-panel modes
    layout_multi  = compute_layout(3)   # (h, w_each)
    layout_single = compute_layout(1)

    while True:
        n_panels  = 3 if mode == 0 else 1
        layout    = layout_multi if mode == 0 else layout_single
        frame     = build_frame(folders[idx], idx, len(folders), mode, *layout)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):          # q / Esc → quit
            break
        elif key in (ord('d'), ord(' '), 83, 0):   # → / space / right arrow
            idx = (idx + 1) % len(folders)
        elif key in (ord('a'), 81, 255):   # ← / left arrow
            idx = (idx - 1) % len(folders)
        elif key == ord('0'):
            mode = 0
        elif key == ord('1'):
            mode = 1
        elif key == ord('2'):
            mode = 2
        elif key == ord('3'):
            mode = 3
        elif key == ord('4'):
            mode = 4
        elif key == ord('f'):
            fullscreen = not fullscreen
            prop = cv2.WND_PROP_FULLSCREEN
            val  = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WINDOW_TITLE, prop, val)
        elif key == ord('s'):
            save_path = root / f"view_{folders[idx].name}_mode{mode}.png"
            cv2.imwrite(str(save_path), frame)
            print(f"💾  Saved → {save_path}")

    cv2.destroyAllWindows()
    print("👋  Viewer closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-by-side viewer for batch inference results (original / segmentation / depth)."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Root folder containing per-image result subfolders (output of infer_folder.py).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.results_dir.resolve()
    if not root.is_dir():
        print(f"❌  '{root}' is not a valid directory.")
        sys.exit(1)
    run_viewer(root)