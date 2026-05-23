"""
unified_seg_teacher.py

MiDaS/DPT-based segmentation teacher:
- Shared backbone: UnifiedMidasBackbone
- Segmentation head: lightweight decoder that upsamples to input resolution.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from unified_midas_backbone import UnifiedMidasBackbone, MidasBackboneFeatures

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

        # Choose a reasonable number of groups for GroupNorm
        def gn(num_channels: int) -> nn.GroupNorm:
            # e.g., 32 groups if possible, else fall back to smaller
            num_groups = 32
            while num_groups > 1 and num_channels % num_groups != 0:
                num_groups //= 2
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)

        self.proj = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, mid_channels,
                kernel_size=3, stride=2, padding=1, bias=False
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                mid_channels, mid_channels,
                kernel_size=3, stride=2, bias=False
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=3, stride=2, padding=1, bias=False
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels, mid_channels,
                kernel_size=3, padding=1, bias=False
            ),
            gn(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        x = self.proj(feat)
        # x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.classifier(x)
        return x


class UnifiedSegmentationTeacher(nn.Module):
    """
    Full segmentation teacher: backbone + segmentation head.

    Expected input:
        - RGB in MiDaS-normalized space (roughly [-1, 1]).
          For your pipeline: images come in [0,1]; use:
              x = (images - 0.5) / 0.5
    """

    def __init__(
        self,
        num_classes: int,
        model_type: str = "DPT_Large",
        *,
        hub_repo: str = "intel-isl/MiDaS",
        hub_kwargs: Optional[dict] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = UnifiedMidasBackbone(
            model_type=model_type,
            midas_model=None,
            hub_repo=hub_repo,
            hub_kwargs=hub_kwargs,
            freeze_encoder=freeze_encoder,
            freeze_decoder=freeze_decoder,
        )

        in_ch = self.backbone.get_feature_dim()
        self.seg_head = SegmentationHead(in_channels=in_ch, num_classes=num_classes)

    @staticmethod
    def normalize_midas(x: torch.Tensor) -> torch.Tensor:
        """
        Match MiDaS DPT normalization.

        Your dataloader already returns images in [0,1].
        MiDaS DPT does (image/255 - 0.5)/0.5 on 0..255 inputs.
        Equivalent here: (x - 0.5)/0.5 -> roughly [-1,1].
        """
        return (x - 0.5) / 0.5

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_backbone: bool = False,
    ) -> Tuple[torch.Tensor, Optional[MidasBackboneFeatures]]:
        """
        x:
            [B, 3, H, W], assumed already normalized via `normalize_midas`.
        """
        H, W = x.shape[-2:]
        feats = self.backbone(x, return_pyramid=False, return_all=False)
        logits = self.seg_head(feats.feat, input_size=(H, W))

        # logits = torch.sigmoid(logits)

        if return_backbone:
            return logits, feats

        return logits, None

    # Convenience for freezing
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
