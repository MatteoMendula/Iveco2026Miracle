"""
unified_depth_teacher.py

MiDaS/DPT-based depth teacher:
- Shared backbone: UnifiedMidasBackbone
- Depth head: reuse MiDaS output_conv by default (scalar depth-like output)

IMPORTANT:
    We treat the head output as a generic "depth" map and NEVER invert it.
    No inverse-depth loss, no 1/x on GT or predictions.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from unified_midas_backbone import UnifiedMidasBackbone, MidasBackboneFeatures


class DepthHead(nn.Module):
    """
    Simple depth head (used only if we can't reuse MiDaS output_conv).

    Input:
        feat: [B, C, H_feat, W_feat]
        input_size: (H_in, W_in)
    Output:
        depth: [B, 1, H_in, W_in]  (scalar depth-like map)
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = max(64, in_channels // 2)

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)

    def forward(self, feat: torch.Tensor, input_size: Tuple[int, int]) -> torch.Tensor:
        x = self.proj(feat)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        x = self.out_conv(x)  # [B,1,H_in,W_in]
        return x


class UnifiedDepthTeacher(nn.Module):
    """
    Full depth teacher: backbone + depth head.

    By default, reuses MiDaS DPT output head (`scratch.output_conv`) so we
    start from the original MiDaS behaviour (no extra transforms).

    Expected input:
        - RGB in MiDaS-normalized space (roughly [-1, 1]):
              x_norm = (x - 0.5) / 0.5
          where x is [0,1] float from your dataloader.
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        *,
        hub_repo: str = "intel-isl/MiDaS",
        hub_kwargs: Optional[dict] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        use_midas_head: bool = True,
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

        # Preferred: reuse MiDaS depth head (already pre-trained).
        self.uses_midas_head = False
        if use_midas_head and self.backbone.depth_head is not None:
            self.depth_head = self.backbone.depth_head  # scratch.output_conv
            self.uses_midas_head = True
        else:
            # Fallback: custom head, randomly initialized.
            self.depth_head = DepthHead(in_channels=in_ch)

    @staticmethod
    def normalize_midas(x: torch.Tensor) -> torch.Tensor:
        """
        Match MiDaS DPT normalization on RGB in [0,1].
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
            [B, 3, H, W], MiDaS-normalized.

        Returns
        -------
        depth:
            [B, 1, H, W] scalar depth-like map (same orientation as MiDaS).
        feats (optional):
            Backbone features (if return_backbone=True), else None.
        """
        B, C, H, W = x.shape
        feats = self.backbone(x, return_pyramid=False, return_all=False)

        d = self.depth_head(feats.feat)  # [B,1,h,w] or [B,h,w]
        if d.dim() == 3:
            d = d.unsqueeze(1)

        if d.shape[-2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bicubic", align_corners=False)

        if return_backbone:
            return d, feats

        return d, None

    # Convenience for freezing
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
