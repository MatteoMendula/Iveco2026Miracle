from __future__ import annotations

"""
unified_midas_backbone.py

Shared MiDaS/DPT backbone with TWO modes:

1) Non-split teacher mode (default)
   - preserves the original whole-model backbone behavior
   - intended for teacher training / eval / old checkpoint compatibility

2) Split mode (optional)
   - exposes forward_head / forward_tail
   - uses the shared split/compressor path

The default forward path remains NON-SPLIT so existing teacher checkpoints
can load and behave near-exactly like the older teacher models.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import copy

import torch
from torch import nn


# ---------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------

@dataclass
class MidasBackboneFeatures:
    feat: torch.Tensor
    pyramid: List[torch.Tensor]
    encoder_layers: List[torch.Tensor]
    extra: Dict[str, torch.Tensor]


@dataclass
class SplitPacket:
    model_type: str
    split_id: str
    compressed_payload: torch.Tensor
    raw_payload: Optional[torch.Tensor]
    cache: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


@dataclass
class SplitBackboneFeatures:
    feat: torch.Tensor
    pyramid: List[torch.Tensor]
    encoder_layers: List[torch.Tensor]
    extra: Dict[str, torch.Tensor]
    reconstructed_payload: Optional[torch.Tensor]


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _gn(ch: int) -> nn.GroupNorm:
    g = min(32, ch)
    while g > 1 and ch % g != 0:
        g //= 2
    return nn.GroupNorm(g, ch)


def fixed_random_keep_mask(
    batch_size: int,
    num_tokens: int,
    keep_count: int,
    *,
    device: torch.device,
    keep_prefix_tokens: int = 0,
) -> torch.Tensor:
    feature_tokens = num_tokens - keep_prefix_tokens
    keep_count = max(0, min(keep_count, feature_tokens))

    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool, device=device)
    if keep_prefix_tokens > 0:
        mask[:, :keep_prefix_tokens] = True
    if keep_count == 0:
        return mask

    for b in range(batch_size):
        perm = torch.randperm(feature_tokens, device=device)
        kept = perm[:keep_count] + keep_prefix_tokens
        mask[b, kept] = True
    return mask


# ---------------------------------------------------------------------
# Compressors used ONLY in split mode
# ---------------------------------------------------------------------

class TokenBottleneck(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 384, bottleneck_channels: int = 192) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, bottleneck_channels),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, channels),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class SpatialBottleneckEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, bottleneck_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            _gn(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, bottleneck_channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialBottleneckDecoder(nn.Module):
    def __init__(self, out_channels: int, hidden_channels: int = 256, bottleneck_channels: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(bottleneck_channels, hidden_channels, kernel_size=1, bias=False),
            _gn(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SpatialBottleneck(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 256, bottleneck_channels: int = 128) -> None:
        super().__init__()
        self.encoder = SpatialBottleneckEncoder(in_channels, hidden_channels, bottleneck_channels)
        self.decoder = SpatialBottleneckDecoder(in_channels, hidden_channels, bottleneck_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


class TinyTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class ViTTokenMaskingCore(nn.Module):
    def __init__(
        self,
        token_dim: int,
        depth_enc: int = 2,
        depth_dec: int = 2,
        num_heads: int = 4,
        keep_ratio: float = 0.25,
        keep_prefix_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.keep_ratio = keep_ratio
        self.keep_prefix_tokens = keep_prefix_tokens

        self.enc_blocks = nn.ModuleList([TinyTransformerBlock(token_dim, num_heads=num_heads) for _ in range(depth_enc)])
        self.dec_blocks = nn.ModuleList([TinyTransformerBlock(token_dim, num_heads=num_heads) for _ in range(depth_dec)])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed: Optional[nn.Parameter] = None
        self.num_tokens_cached: Optional[int] = None

    def _get_pos_embed(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        if self.pos_embed is None or self.num_tokens_cached != num_tokens:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.token_dim, device=device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.num_tokens_cached = num_tokens
        return self.pos_embed

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, N, _ = x.shape
        pos = self._get_pos_embed(N, x.device)
        t = x + pos
        for blk in self.enc_blocks:
            t = blk(t)

        feature_tokens = N - self.keep_prefix_tokens
        keep_count = max(1, int(math.floor(feature_tokens * self.keep_ratio)))
        keep_mask = fixed_random_keep_mask(B, N, keep_count, device=t.device, keep_prefix_tokens=self.keep_prefix_tokens)

        z = t.clone()
        z[~keep_mask] = 0.0
        return z, {"keep_mask": keep_mask, "num_tokens": N}

    def decode(self, z: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        keep_mask = meta["keep_mask"]
        N = meta["num_tokens"]
        pos = self._get_pos_embed(N, z.device)

        x = z.clone()
        x[~keep_mask] = 0.0
        x = x + (~keep_mask).unsqueeze(-1).float() * self.mask_token
        x = x + pos
        for blk in self.dec_blocks:
            x = blk(x)
        return x


class OptionalViTSpatialCompressor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        bottleneck_channels: int = 128,
        token_dim: int = 192,
        patch_size: int = 2,
        depth_enc: int = 2,
        depth_dec: int = 2,
        num_heads: int = 4,
        keep_ratio: float = 0.25,
        use_vit: bool = False,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.use_vit = bool(use_vit)

        self.spatial_encoder = SpatialBottleneckEncoder(in_channels, hidden_channels, bottleneck_channels)
        self.spatial_decoder = SpatialBottleneckDecoder(in_channels, hidden_channels, bottleneck_channels)

        self.tokenize = nn.Conv2d(bottleneck_channels, token_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.detokenize = nn.ConvTranspose2d(token_dim, bottleneck_channels, kernel_size=patch_size, stride=patch_size, bias=True)
        self.vit_core = ViTTokenMaskingCore(token_dim, depth_enc, depth_dec, num_heads, keep_ratio, 0)

    def set_use_vit(self, use_vit: bool) -> None:
        self.use_vit = bool(use_vit)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z_spatial = self.spatial_encoder(x)
        if not self.use_vit:
            return z_spatial, {"use_vit": False}

        t = self.tokenize(z_spatial)
        gh, gw = t.shape[-2:]
        tok = t.flatten(2).transpose(1, 2)
        tok_masked, vit_meta = self.vit_core.encode(tok)
        return tok_masked, {"use_vit": True, "compressed_grid_shape": (gh, gw), "vit_meta": vit_meta}

    def decode(self, z: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        if not meta.get("use_vit", False):
            z_spatial = z
        else:
            gh, gw = meta["compressed_grid_shape"]
            tok = self.vit_core.decode(z, meta["vit_meta"])
            x = tok.transpose(1, 2).reshape(tok.shape[0], self.token_dim, gh, gw)
            z_spatial = self.detokenize(x)
        return self.spatial_decoder(z_spatial)


class ViTTokenViaImageCompressor(nn.Module):
    def __init__(
        self,
        token_channels: int,
        image_token_dim: int = 192,
        patch_size: int = 2,
        depth_enc: int = 2,
        depth_dec: int = 2,
        num_heads: int = 4,
        keep_ratio: float = 0.25,
        keep_prefix_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.image_token_dim = image_token_dim
        self.keep_prefix_tokens = keep_prefix_tokens

        self.to_image = nn.Conv2d(token_channels, image_token_dim, kernel_size=1, bias=True)
        self.from_image = nn.Conv2d(image_token_dim, token_channels, kernel_size=1, bias=True)
        self.tokenize = nn.Conv2d(image_token_dim, image_token_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.detokenize = nn.ConvTranspose2d(image_token_dim, image_token_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.vit_core = ViTTokenMaskingCore(image_token_dim, depth_enc, depth_dec, num_heads, keep_ratio, 0)

    def encode(self, tokens: torch.Tensor, grid_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, _, C = tokens.shape
        gh, gw = grid_size
        kp = self.keep_prefix_tokens
        prefix = tokens[:, :kp] if kp > 0 else None
        feat_tokens = tokens[:, kp:] if kp > 0 else tokens

        x = feat_tokens.transpose(1, 2).reshape(B, C, gh, gw)
        x = self.to_image(x)
        t = self.tokenize(x)
        cgh, cgw = t.shape[-2:]
        tok = t.flatten(2).transpose(1, 2)
        z, vit_meta = self.vit_core.encode(tok)
        return z, {"grid_shape": (gh, gw), "compressed_grid_shape": (cgh, cgw), "prefix": prefix, "vit_meta": vit_meta}

    def decode(self, z: torch.Tensor, meta: Dict[str, Any]) -> torch.Tensor:
        cgh, cgw = meta["compressed_grid_shape"]
        prefix = meta["prefix"]
        tok = self.vit_core.decode(z, meta["vit_meta"])
        x = tok.transpose(1, 2).reshape(tok.shape[0], self.image_token_dim, cgh, cgw)
        x = self.detokenize(x)
        x = self.from_image(x)
        feat_tokens = x.flatten(2).transpose(1, 2)
        if prefix is not None:
            return torch.cat([prefix, feat_tokens], dim=1)
        return feat_tokens




# ---------------------------------------------------------------------
# Custom DPT_Hybrid pre-transformer split modules
# ---------------------------------------------------------------------

@dataclass
class CustomSplitPayloadInfo:
    token_grid: Tuple[int, int]
    token_dim: int
    total_tokens: int
    kept_tokens: int
    keep_ratio: float
    estimated_int8_bytes: int
    target_min_bytes: int
    target_max_bytes: int


@dataclass
class CustomDecodedHybridFeatures:
    stage0_hat: torch.Tensor
    stage1_hat: torch.Tensor
    pretransformer_hat: torch.Tensor


def _pick_groups(ch: int, max_groups: int = 32) -> int:
    g = min(max_groups, ch)
    while g > 1 and ch % g != 0:
        g //= 2
    return g


def _last_conv_out_channels(module: nn.Module, default: int) -> int:
    out_ch = None
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Conv2d):
            out_ch = int(m.out_channels)
            break
    return int(default if out_ch is None else out_ch)


def _conv_stride_tuple(conv: nn.Conv2d) -> Tuple[int, int]:
    s = conv.stride
    return (int(s[0]), int(s[1])) if isinstance(s, tuple) else (int(s), int(s))


def _conv_kernel_tuple(conv: nn.Conv2d) -> Tuple[int, int]:
    k = conv.kernel_size
    return (int(k[0]), int(k[1])) if isinstance(k, tuple) else (int(k), int(k))


class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, kernel_size: int = 3, stride: int = 1, groups: int = 1, act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups, bias=False)
        self.gn = nn.GroupNorm(_pick_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x)))


class DWPointwiseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.dw = ConvGNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch, act=True)
        self.pw = ConvGNAct(in_ch, out_ch, kernel_size=1, stride=1, groups=1, act=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class DSResidualBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.dw = ConvGNAct(ch, ch, kernel_size=3, stride=1, groups=ch, act=True)
        self.pw = nn.Conv2d(ch, ch, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(_pick_groups(ch), ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gn(self.pw(self.dw(x)))
        return self.act(x + y)


class CustomHybridHeadEncoder(nn.Module):
    """Copied 7x7 DPT conv output -> compact spatial latent Z."""
    def __init__(self, stem_ch: int = 64, spatial_ch: int = 128, ch1: int = 64, ch2: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(stem_ch, ch1, kernel_size=3, stride=2, act=True),       # 256 -> 128
            DWPointwiseBlock(ch1, ch2, stride=1),                             # 128 -> 128
            DWPointwiseBlock(ch2, spatial_ch, stride=2),                      # 128 -> 64
            DWPointwiseBlock(spatial_ch, spatial_ch, stride=1),               # refine
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedCustomViTBottleneck(nn.Module):
    """Old-code-exact ViT bottleneck for stage-3 latent reconstruction.

    Execution intentionally mirrors the user's VIT_Encoder.py MaskedAutoencoder
    non-quantized encoder+decoder path:

      encoder side:
        patch_embedding(x)
        flatten + transpose
        + positional_embedding
        sample keep_indices
        masked_full = tokens.clone()
        keep_mask.scatter_(...)
        masked_full = masked_full * keep_mask

      decoder side:
        full_tokens = masked_full
        transformer(full_tokens)
        decoder MLP per token
        reassemble patches into image/feature map

    Important differences from the previous implementation:
      - no encoder transformer before masking
      - no learned mask token
      - no second positional embedding before decoder transformer
      - no ConvTranspose depatchify
      - decoder is Linear -> GELU -> Linear to patch pixels, then exact patch reassembly
    """
    def __init__(
        self,
        spatial_ch: int = 128,
        token_dim: int = 128,
        patch_size: int = 4,
        keep_ratio: float = 0.25,
        depth_enc: int = 2,
        depth_dec: int = 2,
        num_heads: int = 4,
        target_min_bytes: int = 6144,
        target_max_bytes: int = 8192,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        self.spatial_ch = int(spatial_ch)
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.keep_ratio = float(keep_ratio)
        self.target_min_bytes = int(target_min_bytes)
        self.target_max_bytes = int(target_max_bytes)
        self.max_tokens = int(max_tokens)

        # Old code equivalent:
        # self.patch_embedding = nn.Conv2d(input_channels, hidden_dim,
        #                                  kernel_size=patch_size, stride=patch_size)
        self.patchify = nn.Conv2d(
            self.spatial_ch,
            self.token_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )

        # Old code positional_embedding, but allocated with max_tokens so it can
        # support the configured grid. Slice [:, :N, :] during forward.
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_tokens, self.token_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Old code decoder transformer only. depth_enc is accepted for CLI/checkpoint
        # compatibility but intentionally not used in the forward path.
        # This deeper variant keeps the same masked-token structure but uses
        # a larger FFN (4x token_dim) for better bottleneck capacity without
        # increasing payload bytes.
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=int(num_heads),
            dim_feedforward=int(self.token_dim * 4),
            batch_first=True,
            activation="gelu",
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=int(depth_dec))

        # Old code decoder:
        # nn.Linear(hidden_dim, hidden_dim), GELU, Linear(hidden_dim, ps*ps*C)
        self.decoder = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.patch_size * self.patch_size * self.spatial_ch),
        )

    def _make_keep_indices(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        K = max(1, int(round(N * self.keep_ratio)))
        keep = torch.empty(B, K, dtype=torch.long, device=device)
        for b in range(B):
            perm = torch.randperm(N, device=device)
            keep[b] = perm[:K]
        return keep

    def _unpatchify_old_style(self, decoded_patches: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        # decoded_patches: [B, N, C, ps, ps], N=th*tw
        B, N, C, ps, _ = decoded_patches.shape
        if N != th * tw:
            raise RuntimeError(f"Patch count mismatch: N={N}, th*tw={th*tw}")

        # Vectorized equivalent of the old explicit for-loop:
        # out[:, :, h:h+ps, w:w+ps] = decoded_patches[:, j]
        x = decoded_patches.view(B, th, tw, C, ps, ps)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x.view(B, C, th * ps, tw * ps)

    def forward(
        self,
        z: torch.Tensor,
        *,
        force_keep_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, CustomSplitPayloadInfo, torch.Tensor]:
        B, C, H, W = z.shape
        if C != self.spatial_ch:
            raise RuntimeError(f"Expected spatial_ch={self.spatial_ch}, got {C}")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise RuntimeError(f"Z spatial size {(H, W)} must be divisible by patch_size={self.patch_size}")

        th, tw = H // self.patch_size, W // self.patch_size
        N = th * tw
        if N > self.max_tokens:
            raise RuntimeError(f"Need {N} pos tokens, max_tokens={self.max_tokens}")

        # Old encoder path.
        tokens = self.patchify(z)                         # [B, D, th, tw]
        tokens = tokens.flatten(2).transpose(1, 2)         # [B, N, D]
        tokens = tokens + self.pos_embed[:, :N, :]         # add positional embedding ONCE

        keep_idx = (
            self._make_keep_indices(B, N, z.device)
            if force_keep_indices is None
            else force_keep_indices.to(device=z.device, dtype=torch.long)
        )
        K = int(keep_idx.shape[1])
        D = self.token_dim

        # Match old non-quantized path: full-size zero-masked token tensor.
        keep_mask = torch.zeros(B, N, 1, device=z.device, dtype=tokens.dtype)
        keep_mask.scatter_(1, keep_idx.unsqueeze(-1), 1.0)
        masked_full = tokens.clone() * keep_mask           # [B, N, D]

        # We still expose compact kept_tokens for payload accounting/debug,
        # but the actual decoder receives masked_full just like the old code.
        gather_idx = keep_idx.unsqueeze(-1).expand(B, K, D)
        kept_tokens = torch.gather(tokens, dim=1, index=gather_idx)

        # Old decoder path.
        encoded = self.transformer(masked_full)            # [B, N, D]
        decoded = self.decoder(encoded)                    # [B, N, ps*ps*C]
        decoded = decoded.view(B, N, self.spatial_ch, self.patch_size, self.patch_size)
        z_hat = self._unpatchify_old_style(decoded, th, tw)[..., :H, :W]

        info = CustomSplitPayloadInfo(
            token_grid=(th, tw),
            token_dim=D,
            total_tokens=N,
            kept_tokens=K,
            keep_ratio=self.keep_ratio,
            estimated_int8_bytes=int(K * D),
            target_min_bytes=self.target_min_bytes,
            target_max_bytes=self.target_max_bytes,
        )
        return z_hat, info, kept_tokens


class CustomHybridFeatureDecoder(nn.Module):
    """Decode Z/Z_hat into stage0-like, stage1-like, and pre-transformer CNN features."""
    def __init__(self, spatial_ch: int, stage0_ch: int, stage1_ch: int, pre_ch: int) -> None:
        super().__init__()
        self.stage0_decoder = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvGNAct(spatial_ch, 192, kernel_size=3, stride=1, act=True),
            DWPointwiseBlock(192, 256, stride=1),
            nn.Conv2d(256, stage0_ch, kernel_size=1, bias=True),
        )
        self.stage1_decoder = nn.Sequential(
            ConvGNAct(spatial_ch, 256, kernel_size=1, stride=1, act=True),
            DSResidualBlock(256),
            DSResidualBlock(256),
            nn.Conv2d(256, stage1_ch, kernel_size=1, bias=True),
        )
        self.pre_decoder = nn.Sequential(
            ConvGNAct(spatial_ch, 256, kernel_size=1, stride=1, act=True),
            DSResidualBlock(256),
            ConvGNAct(256, 256, kernel_size=3, stride=2, groups=256, act=True),
            ConvGNAct(256, 512, kernel_size=1, stride=1, act=True),
            DSResidualBlock(512),
            nn.Conv2d(512, pre_ch, kernel_size=1, bias=True),
        )

    def forward(self, z: torch.Tensor) -> CustomDecodedHybridFeatures:
        return CustomDecodedHybridFeatures(
            stage0_hat=self.stage0_decoder(z),
            stage1_hat=self.stage1_decoder(z),
            pretransformer_hat=self.pre_decoder(z),
        )


class CustomHybridPreTransformerSplit(nn.Module):
    """Custom DPT_Hybrid split frontend integrated inside UnifiedMidasBackbone."""
    def __init__(self, first_conv: nn.Conv2d, *, stage0_ch: int, stage1_ch: int, pre_ch: int, spatial_ch: int = 128, token_dim: int = 128, patch_size: int = 4, keep_ratio: float = 0.25, vit_depth_enc: int = 2, vit_depth_dec: int = 2, vit_heads: int = 4, freeze_first_conv: bool = False) -> None:
        super().__init__()
        self.first_conv = copy.deepcopy(first_conv)
        if freeze_first_conv:
            for p in self.first_conv.parameters():
                p.requires_grad = False
        stem_ch = int(first_conv.out_channels)
        self.stem_norm = nn.GroupNorm(_pick_groups(stem_ch), stem_ch)
        self.stem_act = nn.SiLU(inplace=True)
        self.encoder = CustomHybridHeadEncoder(stem_ch=stem_ch, spatial_ch=spatial_ch)
        self.vit = MaskedCustomViTBottleneck(spatial_ch=spatial_ch, token_dim=token_dim, patch_size=patch_size, keep_ratio=keep_ratio, depth_enc=vit_depth_enc, depth_dec=vit_depth_dec, num_heads=vit_heads)
        self.decoder = CustomHybridFeatureDecoder(spatial_ch=spatial_ch, stage0_ch=stage0_ch, stage1_ch=stage1_ch, pre_ch=pre_ch)
        self.use_vit = False
        self.fixed_keep_enabled = False
        self.fixed_keep_seed = 12345
        self.spatial_ch = int(spatial_ch)
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.keep_ratio = float(keep_ratio)

    def set_use_vit(self, use_vit: bool) -> None:
        self.use_vit = bool(use_vit)

    def set_fixed_keep(self, enabled: bool, seed: int = 12345) -> None:
        self.fixed_keep_enabled = bool(enabled)
        self.fixed_keep_seed = int(seed)

    def _make_fixed_keep_indices(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        K = max(1, int(round(N * self.keep_ratio)))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.fixed_keep_seed))
        idx = torch.randperm(N, generator=g)[:K].to(device=device, dtype=torch.long)
        return idx.unsqueeze(0).expand(B, -1).contiguous()

    def encode(self, x: torch.Tensor, *, use_vit: Optional[bool] = None, force_keep_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        stem = self.stem_act(self.stem_norm(self.first_conv(x)))
        z = self.encoder(stem)
        active_vit = self.use_vit if use_vit is None else bool(use_vit)
        meta: Dict[str, Any] = {"custom_split": True, "use_vit": active_vit, "stem_shape": tuple(stem.shape), "z_spatial_shape": tuple(z.shape)}
        if not active_vit:
            return z, meta
        if force_keep_indices is None and self.fixed_keep_enabled:
            H, W = int(z.shape[-2]), int(z.shape[-1])
            N = (H // self.patch_size) * (W // self.patch_size)
            force_keep_indices = self._make_fixed_keep_indices(int(z.shape[0]), N, z.device)
            meta["fixed_keep_seed"] = int(self.fixed_keep_seed)
        z_hat, info, kept_tokens = self.vit(z, force_keep_indices=force_keep_indices)
        meta.update({
            "payload_info": {
                "token_grid": info.token_grid,
                "token_dim": info.token_dim,
                "total_tokens": info.total_tokens,
                "kept_tokens": info.kept_tokens,
                "keep_ratio": info.keep_ratio,
                "estimated_int8_bytes": info.estimated_int8_bytes,
                "target_min_bytes": info.target_min_bytes,
                "target_max_bytes": info.target_max_bytes,
            },
            "kept_tokens": kept_tokens,
        })
        return z_hat, meta

    def decode(self, z_or_zhat: torch.Tensor) -> CustomDecodedHybridFeatures:
        return self.decoder(z_or_zhat)



class CustomHybridStage0Decoder(nn.Module):
    """Decode Z/Z_hat into one DPT stage0-compatible feature map only."""
    def __init__(self, spatial_ch: int, stage0_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            ConvGNAct(spatial_ch, 192, kernel_size=3, stride=1, act=True),
            DWPointwiseBlock(192, 256, stride=1),
            nn.Conv2d(256, stage0_ch, kernel_size=1, bias=True),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CustomHybridStage0Split(nn.Module):
    """Custom DPT_Hybrid split that reconstructs only stage0.

    Head side:
        x -> copied first image conv -> light encoder -> z -> optional ViT -> z_hat

    Tail side:
        z/z_hat -> stage0_hat, then the original DPT_Hybrid backbone continues
        from backbone.stages[1:] naturally.
    """
    def __init__(
        self,
        first_conv: nn.Conv2d,
        *,
        stage0_ch: int,
        spatial_ch: int = 64,
        token_dim: int = 128,
        patch_size: int = 4,
        keep_ratio: float = 0.25,
        vit_depth_enc: int = 2,
        vit_depth_dec: int = 2,
        vit_heads: int = 4,
        freeze_first_conv: bool = False,
    ) -> None:
        super().__init__()
        self.first_conv = copy.deepcopy(first_conv)
        if freeze_first_conv:
            for p in self.first_conv.parameters():
                p.requires_grad = False
        stem_ch = int(first_conv.out_channels)
        self.stem_norm = nn.GroupNorm(_pick_groups(stem_ch), stem_ch)
        self.stem_act = nn.SiLU(inplace=True)
        self.encoder = CustomHybridHeadEncoder(stem_ch=stem_ch, spatial_ch=spatial_ch)
        self.vit = MaskedCustomViTBottleneck(
            spatial_ch=spatial_ch,
            token_dim=token_dim,
            patch_size=patch_size,
            keep_ratio=keep_ratio,
            depth_enc=vit_depth_enc,
            depth_dec=vit_depth_dec,
            num_heads=vit_heads,
        )
        self.decoder = CustomHybridStage0Decoder(spatial_ch=spatial_ch, stage0_ch=stage0_ch)
        self.use_vit = False
        self.fixed_keep_enabled = False
        self.fixed_keep_seed = 12345
        self.spatial_ch = int(spatial_ch)
        self.token_dim = int(token_dim)
        self.patch_size = int(patch_size)
        self.keep_ratio = float(keep_ratio)

    def set_use_vit(self, use_vit: bool) -> None:
        self.use_vit = bool(use_vit)

    def set_fixed_keep(self, enabled: bool, seed: int = 12345) -> None:
        self.fixed_keep_enabled = bool(enabled)
        self.fixed_keep_seed = int(seed)

    def _make_fixed_keep_indices(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        K = max(1, int(round(N * self.keep_ratio)))
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.fixed_keep_seed))
        idx = torch.randperm(N, generator=g)[:K].to(device=device, dtype=torch.long)
        return idx.unsqueeze(0).expand(B, -1).contiguous()

    def encode(
        self,
        x: torch.Tensor,
        *,
        use_vit: Optional[bool] = None,
        force_keep_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        stem = self.stem_act(self.stem_norm(self.first_conv(x)))
        z = self.encoder(stem)
        active_vit = self.use_vit if use_vit is None else bool(use_vit)
        meta: Dict[str, Any] = {
            "custom_split": True,
            "custom_stage0": True,
            "use_vit": active_vit,
            "stem_shape": tuple(stem.shape),
            "z_spatial_shape": tuple(z.shape),
        }
        if not active_vit:
            return z, meta
        if force_keep_indices is None and self.fixed_keep_enabled:
            H, W = int(z.shape[-2]), int(z.shape[-1])
            N = (H // self.patch_size) * (W // self.patch_size)
            force_keep_indices = self._make_fixed_keep_indices(int(z.shape[0]), N, z.device)
            meta["fixed_keep_seed"] = int(self.fixed_keep_seed)
        z_hat, info, kept_tokens = self.vit(z, force_keep_indices=force_keep_indices)
        meta.update({
            "payload_info": {
                "token_grid": info.token_grid,
                "token_dim": info.token_dim,
                "total_tokens": info.total_tokens,
                "kept_tokens": info.kept_tokens,
                "keep_ratio": info.keep_ratio,
                "estimated_int8_bytes": info.estimated_int8_bytes,
                "target_min_bytes": info.target_min_bytes,
                "target_max_bytes": info.target_max_bytes,
            },
            "kept_tokens": kept_tokens,
        })
        return z_hat, meta

    def decode(self, z_or_zhat: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_or_zhat)

# ---------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------

class UnifiedMidasBackbone(nn.Module):
    def __init__(
        self,
        model_type: str = "DPT_Large",
        *,
        midas_model: Optional[nn.Module] = None,
        hub_repo: str = "intel-isl/MiDaS",
        hub_kwargs: Optional[Dict[str, Any]] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
        keep_raw_payload: bool = True,
        compressor_type: Optional[str] = None,
        spatial_use_vit: bool = False,
        use_split: bool = False,
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

        if midas_model is None:
            hub_kwargs = {} if hub_kwargs is None else hub_kwargs
            midas_model = torch.hub.load(hub_repo, model_type, **hub_kwargs)

        for m in midas_model.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        if not hasattr(midas_model, "pretrained") or not hasattr(midas_model, "scratch"):
            raise ValueError("Expected DPT-style MiDaS model with pretrained and scratch modules.")

        self.model_type = str(model_type)
        self.pretrained: nn.Module = midas_model.pretrained
        self.scratch: nn.Module = midas_model.scratch
        self.forward_transformer = getattr(midas_model, "forward_transformer", None)
        if self.forward_transformer is None:
            raise ValueError("MiDaS DPT model missing `forward_transformer`.")

        self.channels_last = bool(getattr(midas_model, "channels_last", False))
        self.number_layers = int(getattr(midas_model, "number_layers", 4))
        self.depth_head: Optional[nn.Module] = getattr(self.scratch, "output_conv", None)
        self.keep_raw_payload = bool(keep_raw_payload)
        self.use_split = bool(use_split)
        self.split_frontend_type = str(split_frontend_type)

        ref1 = getattr(self.scratch, "refinenet1", None)
        self._feature_dim = None
        if ref1 is not None and hasattr(ref1, "out_conv"):
            self._feature_dim = int(ref1.out_conv.out_channels)

        if self.model_type == "DPT_Large":
            self.split_id = "large_block5"
            self.hook_block_indices = (5, 11, 17, 23)
            if compressor_type is None:
                compressor_type = "linear_token"
            if compressor_type == "linear_token":
                self.packet_bottleneck = TokenBottleneck(1024, 384, 192)
            elif compressor_type == "vit_token_via_image":
                self.packet_bottleneck = ViTTokenViaImageCompressor(1024, 192, 2, 2, 2, 4, 0.25, 1)
            else:
                raise ValueError(f"Unsupported compressor_type for DPT_Large: {compressor_type}")
            self.compressor_type = compressor_type
        elif self.model_type == "DPT_Hybrid":
            self.hook_block_indices = (0, 1, 8, 11)
            backbone = self.pretrained.model.patch_embed.backbone
            stage0_ch = _last_conv_out_channels(backbone.stages[0], 256)
            stage1_ch = _last_conv_out_channels(backbone.stages[1], 512)
            pre_ch = _last_conv_out_channels(backbone.stages[-1], 1024)
            self.stage0_channels = int(stage0_ch)
            self.stage1_channels = int(stage1_ch)
            self.pretransformer_channels = int(pre_ch)
            self.packet_channels = int(stage1_ch)

            if self.split_frontend_type in ("custom_stage0", "custom_light_stage0"):
                self.split_id = "hybrid_custom_stage0"
                self.compressor_type = "custom_stage0"
                first_conv = self._find_first_dpt_image_conv()
                self.custom_split = CustomHybridStage0Split(
                    first_conv,
                    stage0_ch=self.stage0_channels,
                    spatial_ch=custom_spatial_ch,
                    token_dim=custom_token_dim,
                    patch_size=custom_patch_size,
                    keep_ratio=custom_keep_ratio,
                    vit_depth_enc=custom_vit_depth_enc,
                    vit_depth_dec=custom_vit_depth_dec,
                    vit_heads=custom_vit_heads,
                    freeze_first_conv=custom_freeze_first_conv,
                )
            elif self.split_frontend_type in ("custom_pretransformer", "custom_light_pretransformer"):
                self.split_id = "hybrid_custom_pretransformer"
                self.compressor_type = "custom_pretransformer"
                first_conv = self._find_first_dpt_image_conv()
                self.custom_split = CustomHybridPreTransformerSplit(
                    first_conv,
                    stage0_ch=self.stage0_channels,
                    stage1_ch=self.stage1_channels,
                    pre_ch=self.pretransformer_channels,
                    spatial_ch=custom_spatial_ch,
                    token_dim=custom_token_dim,
                    patch_size=custom_patch_size,
                    keep_ratio=custom_keep_ratio,
                    vit_depth_enc=custom_vit_depth_enc,
                    vit_depth_dec=custom_vit_depth_dec,
                    vit_heads=custom_vit_heads,
                    freeze_first_conv=custom_freeze_first_conv,
                )
            else:
                self.split_id = "hybrid_stage1"
                if compressor_type is None:
                    compressor_type = "conv_spatial"
                if compressor_type == "conv_spatial":
                    self.packet_bottleneck = SpatialBottleneck(stage1_ch, max(128, stage1_ch // 2), max(64, stage1_ch // 4))
                elif compressor_type == "vit_spatial":
                    self.packet_bottleneck = OptionalViTSpatialCompressor(
                        in_channels=stage1_ch,
                        hidden_channels=max(128, stage1_ch // 2),
                        bottleneck_channels=max(64, stage1_ch // 4),
                        token_dim=192,
                        patch_size=2,
                        depth_enc=2,
                        depth_dec=2,
                        num_heads=4,
                        keep_ratio=0.25,
                        use_vit=spatial_use_vit,
                    )
                else:
                    raise ValueError(f"Unsupported compressor_type for DPT_Hybrid: {compressor_type}")
                self.compressor_type = compressor_type
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        if freeze_encoder:
            for p in self.pretrained.parameters():
                p.requires_grad = False
        if freeze_decoder:
            for name, module in self.scratch.named_children():
                if name == "output_conv":
                    continue
                for p in module.parameters():
                    p.requires_grad = False

    # -------------------------
    # Whole-model teacher path
    # -------------------------

    def set_use_split(self, use_split: bool) -> None:
        self.use_split = bool(use_split)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        layers = self.forward_transformer(self.pretrained, x)
        if not isinstance(layers, (list, tuple)):
            raise RuntimeError(f"Expected list/tuple from forward_transformer, got {type(layers)}")
        return list(layers)

    def decode(self, encoder_layers: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        n = len(encoder_layers)
        if n not in (3, 4):
            raise ValueError(f"Expected 3 or 4 encoder layers, got {n}")

        if n == 3:
            layer_1, layer_2, layer_3 = encoder_layers
            layer_4 = None
        else:
            layer_1, layer_2, layer_3, layer_4 = encoder_layers

        l1_rn = self.scratch.layer1_rn(layer_1)
        l2_rn = self.scratch.layer2_rn(layer_2)
        l3_rn = self.scratch.layer3_rn(layer_3)
        l4_rn = None
        if self.number_layers >= 4 and layer_4 is not None:
            l4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = None
        if self.number_layers == 3 or not hasattr(self.scratch, "refinenet4"):
            path_3 = self.scratch.refinenet3(l3_rn, size=l2_rn.shape[2:])
        else:
            path_4 = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
            path_3 = self.scratch.refinenet3(path_4, l3_rn, size=l2_rn.shape[2:])

        path_2 = self.scratch.refinenet2(path_3, l2_rn, size=l1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, l1_rn)

        stem_transpose = getattr(self.scratch, "stem_transpose", None)
        if stem_transpose is not None:
            path_1 = stem_transpose(path_1)

        return path_1, {
            "layer_1": layer_1,
            "layer_2": layer_2,
            "layer_3": layer_3,
            "layer_4": layer_4 if layer_4 is not None else torch.empty(0, device=path_1.device),
            "layer_1_rn": l1_rn,
            "layer_2_rn": l2_rn,
            "layer_3_rn": l3_rn,
            "layer_4_rn": l4_rn if l4_rn is not None else torch.empty(0, device=path_1.device),
            "path_4": path_4 if path_4 is not None else torch.empty(0, device=path_1.device),
            "path_3": path_3,
            "path_2": path_2,
            "path_1": path_1,
        }

    def forward_whole(
        self,
        x: torch.Tensor,
        *,
        return_depth: bool = False,
        return_pyramid: bool = True,
        return_all: bool = False,
    ):
        encoder_layers = self.encode(x)
        feat, intermediates = self.decode(encoder_layers)

        if return_pyramid:
            pyramid: List[torch.Tensor] = []
            if intermediates["path_4"].numel() > 0:
                pyramid.append(intermediates["path_4"])
            pyramid.extend([intermediates["path_3"], intermediates["path_2"], intermediates["path_1"]])
        else:
            pyramid = [feat]

        extra = intermediates if return_all else {
            "path_2": intermediates["path_2"],
            "path_3": intermediates["path_3"],
        }

        features = MidasBackboneFeatures(feat=feat, pyramid=pyramid, encoder_layers=encoder_layers, extra=extra)
        if return_depth and self.depth_head is not None:
            depth = self.depth_head(feat)
            return features, depth
        return features

    # -------------------------
    # Split helpers
    # -------------------------

    def set_spatial_use_vit(self, use_vit: bool) -> None:
        if self.model_type == "DPT_Hybrid" and self.compressor_type == "vit_spatial":
            self.packet_bottleneck.set_use_vit(use_vit)
        if self.model_type == "DPT_Hybrid" and self.compressor_type in ("custom_pretransformer", "custom_stage0"):
            self.custom_split.set_use_vit(use_vit)

    def set_custom_split_use_vit(self, use_vit: bool) -> None:
        if not hasattr(self, "custom_split"):
            raise RuntimeError("No custom_split module exists on this backbone.")
        self.custom_split.set_use_vit(use_vit)

    def set_custom_split_fixed_keep(self, enabled: bool, seed: int = 12345) -> None:
        if not hasattr(self, "custom_split"):
            raise RuntimeError("No custom_split module exists on this backbone.")
        if not hasattr(self.custom_split, "set_fixed_keep"):
            raise RuntimeError("custom_split does not support fixed keep pattern.")
        self.custom_split.set_fixed_keep(enabled, seed)

    def _package_split_backbone_features(
        self,
        encoder_layers: List[torch.Tensor],
        feat: torch.Tensor,
        intermediates: Dict[str, torch.Tensor],
        reconstructed_payload: Optional[torch.Tensor],
        *,
        return_pyramid: bool,
        return_all: bool,
    ) -> SplitBackboneFeatures:
        if return_pyramid:
            pyramid: List[torch.Tensor] = []
            if intermediates["path_4"].numel() > 0:
                pyramid.append(intermediates["path_4"])
            pyramid.extend([intermediates["path_3"], intermediates["path_2"], intermediates["path_1"]])
        else:
            pyramid = [feat]
        extra = intermediates if return_all else {"path_2": intermediates["path_2"], "path_3": intermediates["path_3"]}
        return SplitBackboneFeatures(feat=feat, pyramid=pyramid, encoder_layers=encoder_layers, extra=extra, reconstructed_payload=reconstructed_payload)

    def _large_prepare_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        model = self.pretrained.model
        B = x.shape[0]
        pos_embed = model._resize_pos_embed(
            model.pos_embed,
            x.shape[-2] // model.patch_size[1],
            x.shape[-1] // model.patch_size[0],
        )
        tokens = model.patch_embed(x)
        if getattr(model, "dist_token", None) is not None:
            cls_tokens = model.cls_token.expand(B, -1, -1)
            dist_token = model.dist_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, dist_token, tokens), dim=1)
        else:
            if getattr(model, "no_embed_class", False):
                tokens = tokens + pos_embed
            cls_tokens = model.cls_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
            if not getattr(model, "no_embed_class", False):
                tokens = tokens + pos_embed
        tokens = model.pos_drop(tokens)
        return tokens, {"grid_size": (x.shape[-2] // model.patch_size[1], x.shape[-1] // model.patch_size[0])}

    def _large_postprocess_tokens(self, tokens: torch.Tensor, act_idx: int, grid_size: Tuple[int, int]) -> torch.Tensor:
        post = getattr(self.pretrained, f"act_postprocess{act_idx}")
        x = tokens
        for mod in post:
            if isinstance(mod, nn.Unflatten):
                x = x.contiguous().unflatten(2, grid_size)
            else:
                x = mod(x)
        return x

    def _hybrid_stage01(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        model = self.pretrained.model
        backbone = model.patch_embed.backbone
        y = backbone.stem(x) if hasattr(backbone, "stem") else backbone.root(x)
        stage0 = backbone.stages[0](y)
        stage1 = backbone.stages[1](stage0)
        return stage0, stage1

    def _hybrid_continue_to_tokens(self, stage1: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        model = self.pretrained.model
        backbone = model.patch_embed.backbone
        y = stage1
        for s in range(2, len(backbone.stages)):
            y = backbone.stages[s](y)
        return self._hybrid_pretransformer_to_tokens(y)

    def _hybrid_postprocess_tokens(self, tokens: torch.Tensor, act_idx: int, grid_size: Tuple[int, int]) -> torch.Tensor:
        post = getattr(self.pretrained, f"act_postprocess{act_idx}")
        x = tokens
        for mod in post:
            if isinstance(mod, nn.Unflatten):
                x = x.contiguous().unflatten(2, grid_size)
            else:
                x = mod(x)
        return x


    def _find_first_dpt_image_conv(self) -> nn.Conv2d:
        backbone = self.pretrained.model.patch_embed.backbone
        roots = []
        if hasattr(backbone, "stem"):
            roots.append(backbone.stem)
        if hasattr(backbone, "root"):
            roots.append(backbone.root)
        roots.append(backbone)
        fallback = None
        for root in roots:
            for m in root.modules():
                if isinstance(m, nn.Conv2d) and int(m.in_channels) == 3:
                    if fallback is None:
                        fallback = m
                    kh, kw = _conv_kernel_tuple(m)
                    sh, sw = _conv_stride_tuple(m)
                    if kh == 7 and kw == 7 and sh == 2 and sw == 2:
                        return m
        if fallback is not None:
            return fallback
        raise RuntimeError("Could not find first DPT_Hybrid image Conv2d.")

    def extract_hybrid_teacher_pretransformer(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs the original DPT_Hybrid CNN head up to the pre-transformer CNN feature.
        Use this in split stage-1 training as the feature reconstruction target.
        """
        if self.model_type != "DPT_Hybrid":
            raise RuntimeError("extract_hybrid_teacher_pretransformer is DPT_Hybrid only.")
        model = self.pretrained.model
        backbone = model.patch_embed.backbone
        y = backbone.stem(x) if hasattr(backbone, "stem") else backbone.root(x)
        stage0 = backbone.stages[0](y)
        stage1 = backbone.stages[1](stage0)
        y = stage1
        for sidx in range(2, len(backbone.stages)):
            y = backbone.stages[sidx](y)
        return {"stage0": stage0, "stage1": stage1, "pretransformer": y}

    def _hybrid_pretransformer_to_tokens(self, pre_feat: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        model = self.pretrained.model
        tokens = model.patch_embed.proj(pre_feat).flatten(2).transpose(1, 2)
        B = tokens.shape[0]
        grid_h = int(pre_feat.shape[-2])
        grid_w = int(pre_feat.shape[-1])
        pos_embed = model._resize_pos_embed(model.pos_embed, grid_h, grid_w)
        if getattr(model, "dist_token", None) is not None:
            cls_tokens = model.cls_token.expand(B, -1, -1)
            dist_token = model.dist_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, dist_token, tokens), dim=1)
        else:
            if getattr(model, "no_embed_class", False):
                tokens = tokens + pos_embed
            cls_tokens = model.cls_token.expand(B, -1, -1)
            tokens = torch.cat((cls_tokens, tokens), dim=1)
            if not getattr(model, "no_embed_class", False):
                tokens = tokens + pos_embed
        tokens = model.pos_drop(tokens)
        return tokens, (grid_h, grid_w)

    def forward_head(self, x: torch.Tensor) -> SplitPacket:
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)

        if self.model_type == "DPT_Large":
            tokens, meta = self._large_prepare_tokens(x)
            model = self.pretrained.model
            for idx in range(6):
                tokens = model.blocks[idx](tokens)
            raw_payload = tokens if self.keep_raw_payload else None

            if self.compressor_type == "linear_token":
                compressed_payload = self.packet_bottleneck.encode(tokens)
            elif self.compressor_type == "vit_token_via_image":
                compressed_payload, comp_meta = self.packet_bottleneck.encode(tokens, meta["grid_size"])
                meta = dict(meta); meta.update(comp_meta)
            else:
                raise RuntimeError("Invalid compressor_type for DPT_Large")

            return SplitPacket(self.model_type, self.split_id, compressed_payload, raw_payload, {"layer1_tokens": tokens}, meta)

        if self.model_type == "DPT_Hybrid" and self.compressor_type in ("custom_pretransformer", "custom_stage0"):
            compressed_payload, meta = self.custom_split.encode(x)
            raw_payload = compressed_payload if self.keep_raw_payload else None
            return SplitPacket(self.model_type, self.split_id, compressed_payload, raw_payload, {}, meta)

        stage0, stage1 = self._hybrid_stage01(x)
        raw_payload = stage1 if self.keep_raw_payload else None

        if self.compressor_type == "conv_spatial":
            compressed_payload = self.packet_bottleneck.encode(stage1)
            meta = {}
        elif self.compressor_type == "vit_spatial":
            compressed_payload, meta = self.packet_bottleneck.encode(stage1)
        else:
            raise RuntimeError("Invalid compressor_type for DPT_Hybrid")

        return SplitPacket(self.model_type, self.split_id, compressed_payload, raw_payload, {"stage0": stage0}, meta)


    def _forward_tail_hybrid_from_layers(
        self,
        *,
        layer_1: torch.Tensor,
        layer_2: torch.Tensor,
        tokens: torch.Tensor,
        grid_size: Tuple[int, int],
        reconstructed_payload: Optional[torch.Tensor],
        return_depth: bool,
        return_pyramid: bool,
        return_all: bool,
    ):
        model = self.pretrained.model
        hooks = self.hook_block_indices
        captured: Dict[int, torch.Tensor] = {}
        for idx, blk in enumerate(model.blocks):
            tokens = blk(tokens)
            if idx in (hooks[2], hooks[3]):
                captured[idx] = tokens
        tokens = model.norm(tokens)
        captured[hooks[3]] = tokens

        layer_3 = self._hybrid_postprocess_tokens(captured[hooks[2]], 3, grid_size)
        layer_4 = self._hybrid_postprocess_tokens(captured[hooks[3]], 4, grid_size)
        encoder_layers = [layer_1, layer_2, layer_3, layer_4]
        feat, intermediates = self.decode(encoder_layers)
        features = self._package_split_backbone_features(
            encoder_layers, feat, intermediates, reconstructed_payload,
            return_pyramid=return_pyramid, return_all=return_all,
        )
        if return_depth and self.depth_head is not None:
            depth = self.depth_head(feat)
            return features, depth
        return features

    def forward_tail(
        self,
        packet: SplitPacket,
        *,
        return_depth: bool = False,
        return_pyramid: bool = False,
        return_all: bool = False,
    ):
        if packet.model_type != self.model_type or packet.split_id != self.split_id:
            raise ValueError("Split packet does not match this backbone.")

        if self.model_type == "DPT_Large":
            if self.compressor_type == "linear_token":
                tokens = self.packet_bottleneck.decode(packet.compressed_payload)
            elif self.compressor_type == "vit_token_via_image":
                tokens = self.packet_bottleneck.decode(packet.compressed_payload, packet.meta)
            else:
                raise RuntimeError("Invalid compressor_type for DPT_Large")

            reconstructed_payload = tokens
            model = self.pretrained.model
            hooks = self.hook_block_indices
            captured: Dict[int, torch.Tensor] = {hooks[0]: packet.cache["layer1_tokens"]}

            for idx in range(6, len(model.blocks)):
                tokens = model.blocks[idx](tokens)
                if idx in hooks[1:]:
                    captured[idx] = tokens

            tokens = model.norm(tokens)
            captured[hooks[3]] = tokens
            grid_size = tuple(packet.meta["grid_size"])

            layer_1 = self._large_postprocess_tokens(captured[hooks[0]], 1, grid_size)
            layer_2 = self._large_postprocess_tokens(captured[hooks[1]], 2, grid_size)
            layer_3 = self._large_postprocess_tokens(captured[hooks[2]], 3, grid_size)
            layer_4 = self._large_postprocess_tokens(captured[hooks[3]], 4, grid_size)
        else:
            if self.compressor_type == "custom_stage0":
                stage0 = self.custom_split.decode(packet.compressed_payload)
                model = self.pretrained.model
                backbone = model.patch_embed.backbone
                stage1 = backbone.stages[1](stage0)
                tokens, grid_size = self._hybrid_continue_to_tokens(stage1)
                layer_1 = getattr(self.pretrained, "act_postprocess1")(stage0)
                layer_2 = getattr(self.pretrained, "act_postprocess2")(stage1)
                return self._forward_tail_hybrid_from_layers(
                    layer_1=layer_1,
                    layer_2=layer_2,
                    tokens=tokens,
                    grid_size=grid_size,
                    reconstructed_payload=stage0,
                    return_depth=return_depth,
                    return_pyramid=return_pyramid,
                    return_all=return_all,
                )

            if self.compressor_type == "custom_pretransformer":
                decoded = self.custom_split.decode(packet.compressed_payload)
                layer_1 = getattr(self.pretrained, "act_postprocess1")(decoded.stage0_hat)
                layer_2 = getattr(self.pretrained, "act_postprocess2")(decoded.stage1_hat)
                tokens, grid_size = self._hybrid_pretransformer_to_tokens(decoded.pretransformer_hat)
                return self._forward_tail_hybrid_from_layers(
                    layer_1=layer_1,
                    layer_2=layer_2,
                    tokens=tokens,
                    grid_size=grid_size,
                    reconstructed_payload=decoded.pretransformer_hat,
                    return_depth=return_depth,
                    return_pyramid=return_pyramid,
                    return_all=return_all,
                )

            if self.compressor_type == "conv_spatial":
                stage1 = self.packet_bottleneck.decode(packet.compressed_payload)
            elif self.compressor_type == "vit_spatial":
                stage1 = self.packet_bottleneck.decode(packet.compressed_payload, packet.meta)
            else:
                raise RuntimeError("Invalid compressor_type for DPT_Hybrid")

            reconstructed_payload = stage1
            stage0 = packet.cache["stage0"]
            tokens, grid_size = self._hybrid_continue_to_tokens(stage1)
            layer_1 = getattr(self.pretrained, "act_postprocess1")(stage0)
            layer_2 = getattr(self.pretrained, "act_postprocess2")(stage1)
            return self._forward_tail_hybrid_from_layers(
                layer_1=layer_1,
                layer_2=layer_2,
                tokens=tokens,
                grid_size=grid_size,
                reconstructed_payload=reconstructed_payload,
                return_depth=return_depth,
                return_pyramid=return_pyramid,
                return_all=return_all,
            )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_depth: bool = False,
        return_pyramid: bool = True,
        return_all: bool = False,
        use_split: Optional[bool] = None,
    ):
        active_split = self.use_split if use_split is None else bool(use_split)
        if not active_split:
            return self.forward_whole(x, return_depth=return_depth, return_pyramid=return_pyramid, return_all=return_all)
        return self.forward_tail(self.forward_head(x), return_depth=return_depth, return_pyramid=return_pyramid, return_all=return_all)

    def get_feature_dim(self) -> int:
        if self._feature_dim is not None:
            return self._feature_dim
        ref1 = getattr(self.scratch, "refinenet1", None)
        if ref1 is not None and hasattr(ref1, "out_conv"):
            self._feature_dim = int(ref1.out_conv.out_channels)
            return self._feature_dim
        raise RuntimeError("Could not infer backbone feature dimension.")

    @property
    def out_channels(self) -> int:
        return self.get_feature_dim()

    @property
    def split_points(self) -> Dict[str, str]:
        if self.model_type == "DPT_Hybrid" and self.compressor_type == "custom_pretransformer":
            return {
                "whole": "RGB input -> original full teacher backbone path",
                "split_head": "RGB input -> copied 7x7 conv -> custom DSConv encoder -> optional ViT bottleneck",
                "split_tail": "custom payload -> custom decoder -> pre-transformer feature + DPT skips -> original transformer/DPT decoder",
                "head": "path_1 feature -> task outputs",
            }
        return {
            "whole": "RGB input -> original full teacher backbone path",
            "split_head": "RGB input -> split packet",
            "split_tail": "split packet -> fused path_1 feature map",
            "head": "path_1 feature -> task outputs",
        }

    def split_param_counts(self) -> Dict[str, int]:
        def ct(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters())
        def tr(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        out: Dict[str, int] = {}
        if hasattr(self, "custom_split"):
            out["custom_split_total"] = ct(self.custom_split)
            out["custom_split_trainable"] = tr(self.custom_split)
            out["custom_first_conv"] = ct(self.custom_split.first_conv)
            out["custom_encoder"] = ct(self.custom_split.encoder)
            out["custom_vit"] = ct(self.custom_split.vit)
            out["custom_decoder"] = ct(self.custom_split.decoder)
        if hasattr(self, "packet_bottleneck"):
            out["packet_bottleneck_total"] = ct(self.packet_bottleneck)
            out["packet_bottleneck_trainable"] = tr(self.packet_bottleneck)
        return out

    @staticmethod
    def normalize_midas(x: torch.Tensor) -> torch.Tensor:
        return (x - 0.5) / 0.5


__all__ = [
    "UnifiedMidasBackbone",
    "MidasBackboneFeatures",
    "SplitPacket",
    "SplitBackboneFeatures",
    "TokenBottleneck",
    "SpatialBottleneck",
    "SpatialBottleneckEncoder",
    "SpatialBottleneckDecoder",
    "ViTTokenMaskingCore",
    "OptionalViTSpatialCompressor",
    "ViTTokenViaImageCompressor",
    "CustomHybridPreTransformerSplit",
    "CustomHybridHeadEncoder",
    "MaskedCustomViTBottleneck",
    "CustomHybridFeatureDecoder",
    "CustomSplitPayloadInfo",
    "CustomDecodedHybridFeatures",
]
