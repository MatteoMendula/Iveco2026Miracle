# Inference Setup — Unified Multitask Models (DPT_Large)

This document describes the environment, versions, and steps required to replicate the TensorRT compilation and run the batch inference scripts on a new machine.

---

## Tested Hardware

| Component | Details |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (laptop) |
| CUDA Architecture | Blackwell SM_120 |
| NVIDIA Driver | CUDA 13.x (system) |

---

## Performance Results

| Backend | ms/image (GPU) |
|---|---|
| PyTorch plain FP32 | ~221 ms |
| PyTorch + channels_last + CUDA streams | ~211 ms |
| TensorRT FP16 mixed precision | **~55 ms** |

---

## Python Environment

### Conda env
```bash
conda create -n iveco2026 python=3.12
conda activate iveco2026
```

### PyTorch — MUST be nightly cu128 (Blackwell SM_120 support)
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

Tested version: `torch==2.12.0.dev20260407+cu128`

> ⚠️ Stable PyTorch builds (cu121, cu124) do not include kernels for SM_120 and will cause `CUDA error: no kernel image is available for execution on the device`.

### TensorRT Python bindings
```bash
pip install tensorrt==10.9.0.34
pip install tensorrt-lean tensorrt-dispatch
```

> ⚠️ Version **10.9.0.34** is required. Older versions (e.g. 10.3.0 installed by default) are not compatible with `.trt` files compiled with the Docker 25.04 container.

### Model dependencies
```bash
pip install timm opencv-python numpy
```

---

## TensorRT Engine Compilation

The `.trt` files need to be compiled **once** on a machine with the same target GPU (SM_120). TRT engines are **not portable across different GPU architectures**.

### Step 1 — Export models to ONNX

```bash
python export_onnx.py
```

Produces: `depth_teacher.onnx`, `seg_teacher.onnx`

### Step 2 — Compile with trtexec via Docker

Use the NVIDIA TensorRT 25.04 container which includes TRT 10.9.0 with Blackwell support:

```bash
docker pull nvcr.io/nvidia/tensorrt:25.04-py3
```

```bash
# Depth
docker run --gpus all --rm \
  -v /path/to/onnx/files:/workspace \
  nvcr.io/nvidia/tensorrt:25.04-py3 \
  trtexec --onnx=/workspace/depth_teacher.onnx \
          --saveEngine=/workspace/depth_teacher.trt \
          --fp16 \
          --precisionConstraints=prefer \
          --verbose

# Segmentation
docker run --gpus all --rm \
  -v /path/to/onnx/files:/workspace \
  nvcr.io/nvidia/tensorrt:25.04-py3 \
  trtexec --onnx=/workspace/seg_teacher.onnx \
          --saveEngine=/workspace/seg_teacher.trt \
          --fp16 \
          --precisionConstraints=prefer \
          --verbose
```

> `--precisionConstraints=prefer` = mixed precision: FP16 where safe, FP32 on critical layers (GroupNorm, sigmoid). Do not use `--fp16` alone to avoid depth quality degradation.

### Expected output
```
depth_teacher.trt
seg_teacher.trt
```

---

## Script Configuration

### `test_inference_bulk_trt.py`
Update the engine paths at the top of the file:
```python
DEPTH_TRT = "/path/to/depth_teacher.trt"
SEG_TRT   = "/path/to/seg_teacher.trt"
```

### `test_inference_bulk.py` (PyTorch version, fallback)
Update the checkpoint paths:
```python
DEPTH_CKPT = "/path/to/best_depth_teacher.pth"
SEG_CKPT   = "/path/to/best_seg_teacher.pth"
MODEL_TYPE = "DPT_Large"
```

---

## Usage

```bash
# TRT (recommended)
python test_inference_bulk_trt.py /path/to/images --output_dir /path/to/results

# PyTorch fallback
python test_inference_bulk.py /path/to/images --output_dir /path/to/results
```

### Per-image output
Each image produces a subfolder containing:
```
<image_stem>/
    original.jpg
    detection.jpg
    segmentation.jpg
    depth.jpg
    composite.jpg
    metadata.json
```

### Batch output
```
batch_summary.json   # aggregated stats + per-image latency
```

---

## Important Notes

### TRT engine portability
`.trt` files are **GPU architecture-specific**. An engine compiled for SM_120 (RTX 5090) will not run on SM_89 (RTX 4090) or any other architecture. Always recompile on the target machine.

### Supported model types
- `MODEL_TYPE = "DPT_Large"` — ViT-L/16 backbone, ~307M parameters (recommended)
- `MODEL_TYPE = "DPT_Hybrid"` — ResNet50+ViT hybrid backbone, ~123M parameters (lower quality)

### Teacher vs Split
All scripts exclusively use **non-split teacher models** (`use_split=False`, default). Do not change this without dedicated split checkpoints.

### max_depth_meters
The depth model was trained with `max_depth_meters=200` (new, DPT_Hybrid) or `max_depth_meters=255` (old, DPT_Large). Visualization uses per-frame min-max normalization — the absolute value does not affect visual output quality.