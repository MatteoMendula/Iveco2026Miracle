from __future__ import annotations

"""
unified_seg_teacher.py

Segmentation teacher / split-student wrapper using UnifiedMidasBackbone.

Default behavior:
    whole non-split teacher model

Optional behavior:
    split-capable model when use_split=True

Important:
    The new custom DPT_Hybrid split path lives in unified_midas_backbone.py.
    This wrapper only exposes/passes the split configuration arguments.
"""

from typing import Optional, Tuple, Dict

import torch
from torch import nn

from unified_midas_backbone import (
    UnifiedMidasBackbone,
    MidasBackboneFeatures,
    SplitPacket,
    SplitBackboneFeatures,
)


class SegmentationHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = max(64, in_channels // 2)

        def gn(num_channels: int) -> nn.GroupNorm:
            num_groups = 32
            while num_groups > 1 and num_channels % num_groups != 0:
                num_groups //= 2
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        self.proj = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                bias=False,
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        # input_size is kept for compatibility with older code. The trainer
        # handles final resizing to mask size if needed.
        x = self.proj(feat)
        x = self.classifier(x)
        return x


class UnifiedSegmentationTeacher(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_type: str = "DPT_Large",
        *,
        hub_repo: str = "intel-isl/MiDaS",
        hub_kwargs: Optional[dict] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,

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
    ) -> None:
        super().__init__()
        self.use_split = bool(use_split)

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
        self.seg_head = SegmentationHead(in_channels=in_ch, num_classes=num_classes)

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

    def forward_tail(self, packet: SplitPacket, *, return_backbone: bool = False):
        feats: SplitBackboneFeatures = self.backbone.forward_tail(
            packet,
            return_pyramid=False,
            return_all=False,
        )
        logits = self.seg_head(feats.feat, input_size=(0, 0))
        if return_backbone:
            return logits, feats
        return logits, None

    def forward(self, x: torch.Tensor, *, return_backbone: bool = False):
        if self.use_split:
            feats: SplitBackboneFeatures = self.backbone(
                x,
                return_pyramid=False,
                return_all=False,
                use_split=True,
            )
            logits = self.seg_head(feats.feat, input_size=x.shape[-2:])
            if return_backbone:
                return logits, feats
            return logits, None

        feats: MidasBackboneFeatures = self.backbone(
            x,
            return_pyramid=False,
            return_all=False,
            use_split=False,
        )
        logits = self.seg_head(feats.feat, input_size=x.shape[-2:])
        if return_backbone:
            return logits, feats
        return logits, None

    def load_from_old_teacher_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> int:
        own = self.state_dict()
        copied = 0
        for k, v in state_dict.items():
            if k in own and own[k].shape == v.shape:
                own[k].copy_(v)
                copied += 1
        return copied

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def trainable_param_counts(self):
        counts = {
            "backbone_pretrained": sum(
                p.numel() for p in self.backbone.pretrained.parameters() if p.requires_grad
            ),
            "backbone_scratch": sum(
                p.numel() for p in self.backbone.scratch.parameters() if p.requires_grad
            ),
            "seg_head": sum(p.numel() for p in self.seg_head.parameters() if p.requires_grad),
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
