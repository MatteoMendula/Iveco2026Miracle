# export_onnx.py
import torch
from unified_depth_teacher import UnifiedDepthTeacher
from unified_seg_teacher import UnifiedSegmentationTeacher

DEVICE = torch.device("cuda")
MODEL_TYPE = "DPT_Large"
DEPTH_CKPT = "/home/albus/Documents/AureliusIndustries/demoNovember2025/centurion-aurelius-industries/ai-model/checkpoints/best_depth_teacher.pth"
SEG_CKPT = "/home/albus/Documents/AureliusIndustries/demoNovember2025/centurion-aurelius-industries/ai-model/checkpoints/best_seg_teacher.pth"

x = torch.randn(1, 3, 512, 512).cuda()

# Depth
depth_model = UnifiedDepthTeacher(model_type=MODEL_TYPE, hub_repo="intel-isl/MiDaS").cuda()
ckpt = torch.load(DEPTH_CKPT, map_location=DEVICE)
depth_model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)), strict=False)
depth_model.eval()
torch.onnx.export(depth_model, (x,), "depth_teacher.onnx", opset_version=18,
    input_names=["input"], output_names=["depth"], do_constant_folding=True)
print("✅ depth_teacher.onnx")

# Seg
from unified_seg_teacher import UnifiedSegmentationTeacher
def _infer_seg_classes(path):
    ckpt = torch.load(path, map_location="cpu")
    sd = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    for k, v in sd.items():
        if k.endswith("classifier.weight"):
            return int(v.shape[0])
    return 1

seg_model = UnifiedSegmentationTeacher(num_classes=_infer_seg_classes(SEG_CKPT),
    model_type=MODEL_TYPE, hub_repo="intel-isl/MiDaS").cuda()
ckpt = torch.load(SEG_CKPT, map_location=DEVICE)
seg_model.load_state_dict(ckpt.get("model_state", ckpt.get("state_dict", ckpt)), strict=False)
seg_model.eval()
torch.onnx.export(seg_model, (x,), "seg_teacher.onnx", opset_version=18,
    input_names=["input"], output_names=["seg"], do_constant_folding=True)
print("✅ seg_teacher.onnx")