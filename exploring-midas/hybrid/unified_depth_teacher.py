from __future__ import annotations

"""
unified_depth_teacher.py

Depth teacher / split-student wrapper using UnifiedMidasBackbone.

This version preserves compatibility with existing trained depth teacher checkpoints
that contain keys like:

    depth_head.proj.0.weight
    depth_head.proj.1.weight
    ...

Default behavior:
    whole non-split depth teacher model with custom depth_head.

Optional behavior:
    split-capable depth model when use_split=True.

Important:
    The new custom DPT_Hybrid split path lives in unified_midas_backbone.py.
    This wrapper only exposes/passes the split configuration arguments.
"""

from typing import Optional, Dict, Tuple

import math
import torch
from torch import nn
import torch.nn.functional as F

from unified_midas_backbone import (
    UnifiedMidasBackbone,
    MidasBackboneFeatures,
    SplitPacket,
    SplitBackboneFeatures,
)


class ResidualDepthBlock(nn.Module):
    """
    Small residual refinement block for depth-head features.

    This is intentionally lightweight: two 3x3 convolutions at a fixed channel
    count, with GroupNorm and SiLU. It adds local refinement capacity without
    turning the output head into a second decoder.
    """

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()

        g = min(int(groups), int(channels))
        while channels % g != 0 and g > 1:
            g -= 1

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g, channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g, channels)
        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.act2(x + skip)
        return x


class BalancedUpsampleBlock(nn.Module):
    """
    Controlled learned x2 upsampling block.

    Main path:
        ConvTranspose2d(kernel=4, stride=2, padding=1, output_padding=0)

    Stabilizer path:
        Bilinear resize + 1x1 projection, added at low weight.

    This avoids the old aggressive pattern of two back-to-back transposed
    convolutions followed by a stride-2 correction, while also avoiding the
    overly smooth interpolation-only replacement.
    """

    def __init__(self, in_ch: int, out_ch: int, skip_scale: float = 0.20) -> None:
        super().__init__()
        self.skip_scale = float(skip_scale)

        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            bias=False,
        )

        self.skip_proj = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, bias=False)

        g = min(8, out_ch)
        while out_ch % g != 0 and g > 1:
            g -= 1

        self.norm = nn.GroupNorm(g, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.refine = ResidualDepthBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        learned = self.deconv(x)

        smooth = F.interpolate(
            x,
            size=learned.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        smooth = self.skip_proj(smooth)

        x = learned + self.skip_scale * smooth
        x = self.act(self.norm(x))
        x = self.refine(x)
        return x


class DepthHead(nn.Module):
    """
    Lightweight balanced depth output head.

    Design intent:
        - keep this as a small output refinement head, not a second decoder
        - refine features before spatial expansion
        - use one controlled learned upsample, not stacked aggressive upsampling
        - use interpolation only as a small stabilizing skip branch
        - output normalized depth for the stable training contract

    Expected trainer setting for the current pipeline:
        --prediction-units normalized

    Metric depth is then interpreted as:
        depth_meters = raw_head_output * max_depth_meters
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        min_depth: float = 1e-4,
        use_positive_output: bool = True,
    ) -> None:
        super().__init__()

        # Keep the public signature compatible with existing UnifiedDepthTeacher
        # construction. mid_channels remains accepted but this head intentionally
        # stays lighter than a full 256-channel refinement decoder.
        hidden_channels = 96 if mid_channels is None else min(int(mid_channels), 128)
        up_channels = 64

        self.min_depth = float(min_depth)
        self.use_positive_output = bool(use_positive_output)

        def gn(num_channels: int, max_groups: int = 8) -> nn.GroupNorm:
            num_groups = min(max_groups, num_channels)
            while num_channels % num_groups != 0 and num_groups > 1:
                num_groups -= 1
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            gn(hidden_channels),
            nn.SiLU(inplace=True),
            ResidualDepthBlock(hidden_channels),
        )

        self.up = BalancedUpsampleBlock(
            in_ch=hidden_channels,
            out_ch=up_channels,
            skip_scale=0.20,
        )

        self.hr_refine = ResidualDepthBlock(up_channels)

        self.out = nn.Sequential(
            nn.Conv2d(up_channels, up_channels // 2, kernel_size=3, padding=1, bias=False),
            gn(up_channels // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(up_channels // 2, 1, kernel_size=1, bias=True),
        )

        self._init_output(init_depth_norm=0.35)


    def _init_output(self, init_depth_norm: float = 0.35) -> None:
        init_depth_norm = float(init_depth_norm)
        init_depth_norm = max(1e-3, min(1.0 - 1e-3, init_depth_norm))

        final = self.out[-1]
        if isinstance(final, nn.Conv2d):
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            bias = math.log(init_depth_norm / (1.0 - init_depth_norm))
            nn.init.constant_(final.bias, bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(feat)
        x = self.up(x)
        x = self.hr_refine(x)
        x = self.out(x)

        if self.use_positive_output:
            # Stable normalized depth convention. Use train_depth_teacher.py with
            # --prediction-units normalized so metrics convert by max_depth_meters.
            x = torch.sigmoid(x)
            x = self.min_depth + (1.0 - self.min_depth) * x

        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x


class UnifiedDepthTeacher(nn.Module):
    def __init__(
        self,
        model_type: str = "DPT_Hybrid",
        *,
        hub_repo: str = "intel-isl/MiDaS",
        hub_kwargs: Optional[dict] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,

        # If true, use original MiDaS scratch.output_conv.
        # For our trained depth teachers this should normally stay False.
        use_midas_head: bool = False,
        use_temporal_lstm: bool = False,

        # Legacy split options
        compressor_type: Optional[str] = None,
        spatial_use_vit: bool = False,

        # Common split switch
        use_split: bool = False,

        # New custom DPT_Hybrid split options
        split_frontend_type: str = "legacy",
        custom_spatial_ch: int = 128,
        custom_token_dim: int = 128,
        custom_patch_size: int = 4,
        custom_keep_ratio: float = 0.25,
        custom_vit_depth_enc: int = 2,
        custom_vit_depth_dec: int = 2,
        custom_vit_heads: int = 4,
        custom_freeze_first_conv: bool = False,

        # Custom head options
        depth_head_mid_channels: Optional[int] = None,
        min_depth: float = 0.0,
        positive_depth: bool = True,
    ) -> None:
        super().__init__()
        self.use_split = bool(use_split)
        self.use_midas_head = bool(use_midas_head)
        self.use_temporal_lstm = bool(use_temporal_lstm)

        if self.use_temporal_lstm:
            raise NotImplementedError(
                "Temporal LSTM is not supported in the cleaned depth wrapper."
            )

        self.backbone = UnifiedMidasBackbone(
            model_type=model_type,
            midas_model=None,
            hub_repo=hub_repo,
            hub_kwargs=hub_kwargs,
            freeze_encoder=freeze_encoder,
            freeze_decoder=freeze_decoder,
            compressor_type=compressor_type,
            spatial_use_vit=spatial_use_vit,
            use_split=use_split,

            split_frontend_type=split_frontend_type,
            custom_spatial_ch=custom_spatial_ch,
            custom_token_dim=custom_token_dim,
            custom_patch_size=custom_patch_size,
            custom_keep_ratio=custom_keep_ratio,
            custom_vit_depth_enc=custom_vit_depth_enc,
            custom_vit_depth_dec=custom_vit_depth_dec,
            custom_vit_heads=custom_vit_heads,
            custom_freeze_first_conv=custom_freeze_first_conv,
        )

        in_ch = self.backbone.get_feature_dim()
        self.depth_head = DepthHead(
            in_channels=in_ch,
            mid_channels=depth_head_mid_channels,
            min_depth=min_depth,
            use_positive_output=positive_depth,
        )

        if self.use_midas_head and self.backbone.depth_head is None:
            raise RuntimeError("use_midas_head=True but backbone has no scratch.output_conv.")

    @staticmethod
    def normalize_midas(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5

    def set_use_split(self, use_split: bool) -> None:
        self.use_split = bool(use_split)
        self.backbone.set_use_split(use_split)

    def set_custom_split_use_vit(self, use_vit: bool) -> None:
        if not hasattr(self.backbone, "set_custom_split_use_vit"):
            raise RuntimeError("Current backbone does not expose set_custom_split_use_vit().")
        self.backbone.set_custom_split_use_vit(use_vit)

    def forward_head(self, x: torch.Tensor) -> SplitPacket:
        return self.backbone.forward_head(x)

    def _depth_from_features(self, feats) -> torch.Tensor:
        if self.use_midas_head:
            # Rare fallback path; not recommended for our trained teacher checkpoints.
            if self.backbone.depth_head is None:
                raise RuntimeError("No MiDaS depth head available.")
            depth = self.backbone.depth_head(feats.feat)
        else:
            depth = self.depth_head(feats.feat)

        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        return depth

    def forward_tail(self, packet: SplitPacket, *, return_backbone: bool = False):
        feats: SplitBackboneFeatures = self.backbone.forward_tail(
            packet,
            return_depth=False,
            return_pyramid=False,
            return_all=False,
        )
        depth = self._depth_from_features(feats)
        if return_backbone:
            return depth, feats
        return depth, None

    def forward_student(self, x: torch.Tensor, *, return_backbone: bool = False):
        feats = self.backbone(
            x,
            return_depth=False,
            return_pyramid=False,
            return_all=False,
            use_split=self.use_split,
        )
        depth = self._depth_from_features(feats)
        if return_backbone:
            return depth, feats
        return depth, None

    def forward_teacher(self, x: torch.Tensor, *, return_backbone: bool = False):
        old = self.use_split
        self.set_use_split(False)
        try:
            return self.forward_student(x, return_backbone=return_backbone)
        finally:
            self.set_use_split(old)

    def forward(self, x: torch.Tensor, *, return_backbone: bool = False):
        return self.forward_student(x, return_backbone=return_backbone)

    def load_from_old_teacher_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        own = self.state_dict()
        copied = 0
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                copied += 1
        return copied

    def trainable_param_counts(self):
        counts = {
            "backbone_pretrained": sum(
                p.numel() for p in self.backbone.pretrained.parameters() if p.requires_grad
            ),
            "backbone_scratch": sum(
                p.numel() for p in self.backbone.scratch.parameters() if p.requires_grad
            ),
            "depth_head": sum(p.numel() for p in self.depth_head.parameters() if p.requires_grad),
        }

        if hasattr(self.backbone, "packet_bottleneck"):
            counts["packet_bottleneck"] = sum(
                p.numel() for p in self.backbone.packet_bottleneck.parameters()
                if p.requires_grad
            )

        if hasattr(self.backbone, "custom_split"):
            counts["custom_split"] = sum(
                p.numel() for p in self.backbone.custom_split.parameters()
                if p.requires_grad
            )
            counts["custom_encoder"] = sum(
                p.numel() for p in self.backbone.custom_split.encoder.parameters()
                if p.requires_grad
            )
            counts["custom_vit"] = sum(
                p.numel() for p in self.backbone.custom_split.vit.parameters()
                if p.requires_grad
            )
            counts["custom_decoder"] = sum(
                p.numel() for p in self.backbone.custom_split.decoder.parameters()
                if p.requires_grad
            )

        return counts
