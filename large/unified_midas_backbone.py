"""
NOT FULLY FINISHED FOR SPLITTING!

unified_midas_backbone.py

Unified MiDaS / DPT-style backbone.

This module exposes a high-level backbone that reuses the exact encoder/decoder
from a DPT-based MiDaS model (e.g. "DPT_Large" or "DPT_Hybrid" from
intel-isl/MiDaS) but returns a clean feature tensor that can be shared across
multiple tasks (depth, segmentation, detection) and later split across devices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class MidasBackboneFeatures:
    """
    Container for backbone outputs.

    Attributes
    ----------
    feat :
        High-resolution fused feature map (what we will usually feed into
        downstream task heads). Shape is typically [B, C, H, W] with stride 4.
    pyramid :
        Multi-scale feature maps in decoder order, from coarse to fine.
        Example for 4-layer DPT: [path_4, path_3, path_2, path_1].
        Elements that do not exist for a given backbone are omitted.
    encoder_layers :
        Raw encoder feature maps as returned by the MiDaS transformer wrapper.
        For a 4-layer DPT this is [layer_1, layer_2, layer_3, layer_4].
    extra :
        Any other intermediate tensors you might want for debugging / analysis.
    """

    feat: torch.Tensor
    pyramid: List[torch.Tensor]
    encoder_layers: List[torch.Tensor]
    extra: Dict[str, torch.Tensor]


class UnifiedMidasBackbone(nn.Module):
    """
    Thin wrapper around a DPT-based MiDaS model that exposes a reusable backbone.

    Key ideas
    ---------
    * We **reuse the exact MiDaS encoder/decoder modules** (`pretrained`,
      `scratch`, and `forward_transformer`) so that loading weights from
      `torch.hub.load("intel-isl/MiDaS", "DPT_Large")` is trivial.
    * We re-implement only the high-level forward pass, stopping **before**
      the original depth head (`scratch.output_conv`). That gives us a clean
      feature map `feat` that we can share between multiple task heads.
    * The code is structured into explicit **encode / decode stages**, which
      are natural split points for future split computing.

    Typical usage
    -------------
    >>> backbone = UnifiedMidasBackbone(model_type="DPT_Large")
    >>> out = backbone(images)
    >>> feat = out.feat            # [B, C, H, W] shared features
    >>> pyramid = out.pyramid      # multi-scale features if you need FPN-style heads

    If you already constructed a MiDaS model yourself, you can pass it
    directly instead of letting this class call `torch.hub.load`:

    >>> midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    >>> backbone = UnifiedMidasBackbone(midas_model=midas)
    """

    def __init__(
        self,
        model_type: str = "DPT_Large",
        *,
        midas_model: Optional[nn.Module] = None,
        hub_repo: str = "intel-isl/MiDaS",
        hub_kwargs: Optional[Dict[str, Any]] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        model_type:
            MiDaS model type string passed to `torch.hub.load`. For DPT-based
            models you typically want: "DPT_Large" or "DPT_Hybrid".
        midas_model:
            Optional, already-instantiated MiDaS model (e.g. from torch.hub).
            If provided, we wrap this instance instead of calling torch.hub.
        hub_repo:
            Torch hub repo string. Defaults to "intel-isl/MiDaS".
        hub_kwargs:
            Extra keyword arguments forwarded to `torch.hub.load`.
        freeze_encoder:
            If True, all parameters in the transformer backbone (`pretrained`)
            are frozen (no gradient).
        freeze_decoder:
            If True, all parameters in the refinenet decoder (`scratch.*`,
            except the depth head) are frozen.
        """
        super().__init__()

        if midas_model is None:
            if hub_kwargs is None:
                hub_kwargs = {}
            # NOTE: this assumes the environment has access to the MiDaS hub
            # repo. If you prefer to load weights from disk, construct the
            # MiDaS model yourself and pass it via `midas_model=...`.
            midas_model = torch.hub.load(hub_repo, model_type, **hub_kwargs)

        # Sanity check that we indeed have a DPT-style model.
        if not hasattr(midas_model, "pretrained") or not hasattr(
            midas_model, "scratch"
        ):
            raise ValueError(
                "The provided MiDaS model does not look like a DPT model "
                "(missing `pretrained` or `scratch` attributes). "
                "Make sure you use a DPT-based MiDaS variant such as "
                "\"DPT_Large\" or \"DPT_Hybrid\"."
            )

        # --- Copy the important pieces from the MiDaS model ---
        # These are registered as submodules of this backbone, so they will
        # participate in parameter iteration, .to(device), etc.
        self.pretrained: nn.Module = midas_model.pretrained
        self.scratch: nn.Module = midas_model.scratch

        # `forward_transformer` is a plain function (e.g. forward_vit / forward_beit)
        # that MiDaS attaches to the DPT instance; we just reuse it.
        self.forward_transformer = getattr(midas_model, "forward_transformer", None)
        if self.forward_transformer is None:
            raise ValueError(
                "MiDaS DPT model is missing `forward_transformer`. "
                "This wrapper currently expects the standard MiDaS DPT API."
            )

        # Meta flags copied from the original DPT implementation.
        self.channels_last: bool = getattr(midas_model, "channels_last", False)
        self.number_layers: int = int(getattr(midas_model, "number_layers", 4))

        # Store a reference to the original depth head (scratch.output_conv),
        # but we DO NOT call it inside the backbone. Later we can either:
        #   * reuse these convs as our depth head, or
        #   * drop them and attach custom multi-task heads.
        self.depth_head: Optional[nn.Module] = getattr(
            self.scratch, "output_conv", None
        )

        # Try to infer the final feature dimensionality from refinenet1.
        self._feature_dim: Optional[int] = None
        ref1 = getattr(self.scratch, "refinenet1", None)
        if ref1 is not None and hasattr(ref1, "out_conv"):
            try:
                self._feature_dim = int(ref1.out_conv.out_channels)
            except Exception:
                self._feature_dim = None

        # Freeze sub-parts if requested.
        if freeze_encoder:
            for p in self.pretrained.parameters():
                p.requires_grad = False
        if freeze_decoder:
            for name, module in self.scratch.named_children():
                # Do not accidentally freeze the depth head here; we want to
                # make that decision explicitly later.
                if name == "output_conv":
                    continue
                for p in module.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    #  Stage 1: encoder / transformer
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run the MiDaS transformer encoder and return the raw feature maps.

        This corresponds to the original DPT call:
            layers = forward_{vit/beit/...}(self.pretrained, x)

        Returns
        -------
        List[Tensor]
            List of encoder feature maps at multiple resolutions. For standard
            DPT backbones this list has length 4.
        """
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)

        # forward_transformer is a free function bound into the model
        # (e.g. forward_vit(pretrained, x) -> [layer_1, ..., layer_4]).
        layers = self.forward_transformer(self.pretrained, x)

        if not isinstance(layers, (list, tuple)):
            raise RuntimeError(
                "Expected MiDaS forward_transformer to return a list/tuple of "
                f"feature maps, but got type {type(layers)} instead."
            )

        return list(layers)

    # ------------------------------------------------------------------
    #  Stage 2: decoder / refinenet fusion
    # ------------------------------------------------------------------
    def decode(
        self, encoder_layers: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Run the DPT refinenet decoder given encoder features.

        Parameters
        ----------
        encoder_layers:
            List of feature maps from :meth:`encode`. For standard DPT models
            this is a list of length 4: [layer_1, layer_2, layer_3, layer_4].

        Returns
        -------
        feat :
            Highest-resolution fused feature map (stride 4 in the original
            DPT design). This is what we typically feed to downstream heads.
        intermediates :
            Dictionary with all intermediate decoder activations
            (layer_*_rn, path_*, etc.).
        """
        n = len(encoder_layers)
        if n not in (3, 4):
            raise ValueError(
                f"Expected 3 or 4 encoder layers, but got {n}. "
                "This wrapper is written for standard DPT configs."
            )

        # Unpack encoder outputs.
        if n == 3:
            layer_1, layer_2, layer_3 = encoder_layers
            layer_4 = None
        else:
            layer_1, layer_2, layer_3, layer_4 = encoder_layers

        # Adapt feature dimensions via the scratch conv layers.
        l1_rn = self.scratch.layer1_rn(layer_1)
        l2_rn = self.scratch.layer2_rn(layer_2)
        l3_rn = self.scratch.layer3_rn(layer_3)
        l4_rn = None

        if self.number_layers >= 4 and layer_4 is not None:
            # Some backbones (e.g. BEiT/Swin/ViT-L) use a 4th encoder stage.
            l4_rn = self.scratch.layer4_rn(layer_4)

        # Top-down fusion path, following the DPT refinenet design.
        path_4 = None
        if self.number_layers == 3 or not hasattr(self.scratch, "refinenet4"):
            # 3-stage variant: start fusion from the deepest feature.
            path_3 = self.scratch.refinenet3(
                l3_rn, size=l2_rn.shape[2:]
            )
        else:
            # 4-stage variant: fuse the deepest level into the next one.
            path_4 = self.scratch.refinenet4(
                l4_rn, size=l3_rn.shape[2:]
            )
            path_3 = self.scratch.refinenet3(
                path_4, l3_rn, size=l2_rn.shape[2:]
            )

        path_2 = self.scratch.refinenet2(
            path_3, l2_rn, size=l1_rn.shape[2:]
        )
        path_1 = self.scratch.refinenet1(path_2, l1_rn)

        # Optional extra upsampling step used for some LeViT backbones.
        stem_transpose = getattr(self.scratch, "stem_transpose", None)
        if stem_transpose is not None:
            path_1 = stem_transpose(path_1)

        intermediates: Dict[str, torch.Tensor] = {
            "layer_1": layer_1,
            "layer_2": layer_2,
            "layer_3": layer_3,
            "layer_4": layer_4 if layer_4 is not None else torch.empty(0),
            "layer_1_rn": l1_rn,
            "layer_2_rn": l2_rn,
            "layer_3_rn": l3_rn,
            "layer_4_rn": l4_rn if l4_rn is not None else torch.empty(0),
            "path_4": path_4 if path_4 is not None else torch.empty(0),
            "path_3": path_3,
            "path_2": path_2,
            "path_1": path_1,
        }

        return path_1, intermediates

    # ------------------------------------------------------------------
    #  Unified forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        *,
        return_depth: bool = False,
        return_pyramid: bool = True,
        return_all: bool = False,
    ) -> MidasBackboneFeatures | Tuple[MidasBackboneFeatures, torch.Tensor]:
        """
        Complete backbone forward pass.

        Parameters
        ----------
        x:
            Input RGB tensor of shape [B, 3, H, W]. **Expected normalized**
            like MiDaS DPT (roughly in [-1, 1] if you follow the MiDaS
            transform).
        return_depth:
            If True and a MiDaS depth head is available, also return the
            MiDaS depth prediction (using the original depth head). This is
            mainly useful for sanity-checking that the backbone behaves like
            the original MiDaS model.
        return_pyramid:
            If True, populate the `pyramid` field with multi-scale decoder
            features. If False, `pyramid` will contain only `[feat]`.
        return_all:
            If True, keep all intermediate tensors in `extra`. If False,
            only a minimal subset is stored.

        Returns
        -------
        MidasBackboneFeatures
            Backbone features; if `return_depth=True`, also returns a depth
            tensor as a second element of the tuple.
        """
        encoder_layers = self.encode(x)
        feat, intermediates = self.decode(encoder_layers)

        if return_pyramid:
            pyramid: List[torch.Tensor] = []
            # Coarse-to-fine order.
            if "path_4" in intermediates and intermediates["path_4"].numel() > 0:
                pyramid.append(intermediates["path_4"])
            pyramid.append(intermediates["path_3"])
            pyramid.append(intermediates["path_2"])
            pyramid.append(intermediates["path_1"])
        else:
            pyramid = [feat]

        if return_all:
            extra = intermediates
        else:
            extra = {"path_2": intermediates["path_2"], "path_3": intermediates["path_3"]}

        features = MidasBackboneFeatures(
            feat=feat,
            pyramid=pyramid,
            encoder_layers=encoder_layers,
            extra=extra,
        )

        if return_depth and self.depth_head is not None:
            depth = self.depth_head(feat)
            # Original DPTDepthModel squeezes the channel dimension; we keep
            # it as [B, 1, H, W] here so downstream code can treat it as a
            # regular feature map if desired.
            return features, depth

        return features

    # ------------------------------------------------------------------
    #  Convenience helpers
    # ------------------------------------------------------------------
    def get_feature_dim(self) -> int:
        """
        Return the channel dimension of the fused feature map (path_1).

        This is what segmentation / detection heads should use as their
        input channel count.
        """
        if self._feature_dim is not None:
            return self._feature_dim

        # Fallback: try to infer from refinenet1 on a dummy input.
        ref1 = getattr(self.scratch, "refinenet1", None)
        if ref1 is not None and hasattr(ref1, "out_conv"):
            try:
                self._feature_dim = int(ref1.out_conv.out_channels)
                return self._feature_dim
            except Exception:
                pass

        raise RuntimeError(
            "Could not infer backbone feature dimension; "
            "check the DPT scratch.refinenet1 implementation."
        )

    @property
    def out_channels(self) -> int:
        """Alias to `get_feature_dim()` for convenience."""
        return self.get_feature_dim()

    @property
    def split_points(self) -> Dict[str, str]:
        """
        Human-readable description of natural split points for split computing.

        Returns
        -------
        Dict[str, str]
            Keys are stage names; values describe the corresponding tensors.

        * "encode"  : input -> encoder feature list (Transformer backbone).
        * "decode"  : encoder features -> fused spatial feature map.
        * "head"    : fused feature -> task-specific predictions (not defined
                      in this file; will be added by future heads).
        """
        return {
            "encode": "RGB input -> [layer_1, layer_2, layer_3, layer_4]",
            "decode": "[layers] -> fused path_1 feature map (stride ~4)",
            "head": "path_1 feature -> task outputs (depth / seg / detection)",
        }


__all__ = ["UnifiedMidasBackbone", "MidasBackboneFeatures"]
