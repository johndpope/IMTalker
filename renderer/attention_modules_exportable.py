"""
Export-Friendly Attention Modules for IMTalker
===============================================

This module provides ONNX/TF.js-exportable replacements for the attention
mechanisms in attention_modules.py.

Key Changes:
1. ExportFriendlyMultiheadAttention - Uses nn.MultiheadAttention with fixed sequence lengths
2. ExportFriendlyUpsampler - Replaces GuidedResampler with learnable upsampling + attention
3. ExportFriendlySwinAttention - Pre-computes relative position bias, uses fixed window partitioning
4. Export-friendly transformer blocks with static shapes

The main issue with GuidedResampler is its use of:
- torch.topk() with dynamic k values
- torch.gather() with dynamic indices
- Complex index arithmetic that creates data-dependent control flow

Our solution replaces this with:
- Learnable upsampling (ConvTranspose2d)
- Standard cross-attention between upsampled features and high-res features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ExportFriendlyMultiheadAttention(nn.Module):
    """
    Export-friendly multihead attention using PyTorch's nn.MultiheadAttention.

    This is a drop-in replacement for StandardUnifiedAttention that:
    1. Uses the optimized nn.MultiheadAttention implementation
    2. Supports fixed maximum sequence lengths for ONNX export
    3. Returns compatible output format (output, attention_weights)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Use nn.MultiheadAttention for better export compatibility
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            bias=qkv_bias,
            batch_first=True,
        )

        # Additional projection dropout (MHA has internal dropout)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N_q, C]
            key: [B, N_k, C]
            value: [B, N_k, C]
            mask: Optional attention mask

        Returns:
            output: [B, N_q, C]
            attn_weights: [B, num_heads, N_q, N_k]
        """
        # Convert mask format if needed
        attn_mask = None
        if mask is not None:
            # nn.MultiheadAttention expects additive mask where -inf means "don't attend"
            # Our mask is 0 for "don't attend", so we convert
            attn_mask = mask.float().masked_fill(mask == 0, float('-inf'))

        # nn.MultiheadAttention with batch_first=True
        output, attn_weights = self.mha(
            query, key, value,
            attn_mask=attn_mask,
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights
        )

        output = self.proj_drop(output)

        return output, attn_weights

    @classmethod
    def from_original(cls, original: 'StandardUnifiedAttention') -> 'ExportFriendlyMultiheadAttention':
        """Convert from original StandardUnifiedAttention."""
        exportable = cls(
            dim=original.q_proj.in_features,
            num_heads=original.num_heads,
            qkv_bias=original.q_proj.bias is not None,
            attn_drop=original.attn_drop.p if hasattr(original.attn_drop, 'p') else 0.0,
            proj_drop=original.proj_drop.p if hasattr(original.proj_drop, 'p') else 0.0,
        )

        # Transfer weights to nn.MultiheadAttention format
        # MHA packs Q, K, V into in_proj_weight
        with torch.no_grad():
            # in_proj_weight is [3*dim, dim] for Q, K, V stacked
            exportable.mha.in_proj_weight.copy_(torch.cat([
                original.q_proj.weight,
                original.k_proj.weight,
                original.v_proj.weight,
            ], dim=0))

            if original.q_proj.bias is not None:
                exportable.mha.in_proj_bias.copy_(torch.cat([
                    original.q_proj.bias,
                    original.k_proj.bias,
                    original.v_proj.bias,
                ], dim=0))

            # Output projection
            exportable.mha.out_proj.weight.copy_(original.proj.weight)
            if original.proj.bias is not None:
                exportable.mha.out_proj.bias.copy_(original.proj.bias)

        return exportable


class ExportFriendlyUpsampler(nn.Module):
    """
    Export-friendly replacement for GuidedResampler.

    Instead of using dynamic top-k selection and sparse indexing,
    we use learnable upsampling followed by cross-attention.

    The key insight is that GuidedResampler essentially:
    1. Uses coarse attention map to guide selection from high-res features
    2. Warps/resamples high-res features based on coarse attention

    Our replacement:
    1. Upsample low-res features to high-res using ConvTranspose2d
    2. Apply cross-attention between upsampled features and high-res features
    3. This achieves similar "guided" behavior but is fully exportable
    """

    def __init__(
        self,
        dim: int,
        upsample_ratio: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.ratio = upsample_ratio
        self.num_heads = num_heads

        # Learnable upsampler
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=upsample_ratio, stride=upsample_ratio),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        )

        # Cross-attention to blend with high-res features
        self.cross_attn = ExportFriendlyMultiheadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=True,
            attn_drop=0.0,
            proj_drop=0.0,
        )

        # Layer norms
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Output projection
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(
        self,
        v_high_feat: torch.Tensor,
        coarse_attn_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            v_high_feat: High resolution features [B, C, H, W]
            coarse_attn_map: Attention map from coarse level [B, N_low, N_low]
                            (Note: We don't actually use this for dynamic indexing,
                             but keep the signature for compatibility)

        Returns:
            warped_feat: Warped high-res features [B, C, H, W]
        """
        B, C, H, W = v_high_feat.shape
        H_low, W_low = H // self.ratio, W // self.ratio

        # Create low-res query from attention map (learnable projection)
        # We interpret the attention map as a soft selection over spatial locations
        # Shape: [B, N_low, N_low] -> [B, C, H_low, W_low]

        # Use the attention map to weight the high-res features at coarse level
        # First, pool high-res to low-res
        v_low = F.adaptive_avg_pool2d(v_high_feat, (H_low, W_low))

        # Upsample to high resolution
        v_upsampled = self.upsample(v_low)  # [B, C, H, W]

        # Flatten for attention
        q = v_upsampled.flatten(2).transpose(1, 2)  # [B, H*W, C]
        kv = v_high_feat.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Normalize
        q = self.norm_q(q)
        kv = self.norm_kv(kv)

        # Cross-attention: upsampled queries attending to high-res key/values
        out, _ = self.cross_attn(q, kv, kv)

        # Reshape back to spatial
        out = out.transpose(1, 2).view(B, C, H, W)

        # Final projection
        out = self.out_proj(out)

        return out

    @classmethod
    def from_original(
        cls,
        original: 'GuidedResampler',
        num_heads: int = 8,
    ) -> 'ExportFriendlyUpsampler':
        """
        Create from original GuidedResampler.

        Note: GuidedResampler has no learnable parameters, so this
        creates a new module that needs to be trained/fine-tuned.
        """
        return cls(
            dim=original.dim,
            upsample_ratio=original.ratio,
            num_heads=num_heads,
        )


class ExportFriendlySwinAttention(nn.Module):
    """
    Export-friendly Swin Transformer attention.

    Key changes for exportability:
    1. Pre-compute relative position bias as a fixed buffer
    2. Use standard attention operations without dynamic indexing
    3. Fixed window size with pre-computed masks
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        # Relative position bias - pre-expanded to [num_heads, ws*ws, ws*ws]
        self.relative_position_bias = nn.Parameter(
            torch.zeros(num_heads, window_size * window_size, window_size * window_size)
        )
        nn.init.trunc_normal_(self.relative_position_bias, std=0.02)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B*num_windows, window_size*window_size, C]
            key: [B*num_windows, window_size*window_size, C]
            value: [B*num_windows, window_size*window_size, C]
            mask: Optional shift mask [num_windows, ws*ws, ws*ws]

        Returns:
            output: [B*num_windows, window_size*window_size, C]
        """
        B_, N, C = query.shape

        # Project Q, K, V
        q = self.q(query).view(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key).view(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(value).view(B_, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B_, num_heads, N, N]

        # Add pre-computed relative position bias
        attn = attn + self.relative_position_bias.unsqueeze(0)

        # Apply shift mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    @classmethod
    def from_original(cls, original: 'SwinUnifiedAttention') -> 'ExportFriendlySwinAttention':
        """Convert from original SwinUnifiedAttention."""
        # Infer window size from relative_position_bias_table
        table_size = original.relative_position_bias_table.shape[0]
        # table_size = (2*ws-1)^2, solve for ws
        ws = (int(math.sqrt(table_size)) + 1) // 2

        exportable = cls(
            dim=original.q.in_features,
            num_heads=original.num_heads,
            window_size=ws,
            qkv_bias=original.q.bias is not None,
            attn_drop=original.attn_drop.p if hasattr(original.attn_drop, 'p') else 0.0,
            proj_drop=original.proj_drop.p if hasattr(original.proj_drop, 'p') else 0.0,
        )

        with torch.no_grad():
            # Copy Q, K, V projections
            exportable.q.weight.copy_(original.q.weight)
            exportable.k.weight.copy_(original.k.weight)
            exportable.v.weight.copy_(original.v.weight)

            if original.q.bias is not None:
                exportable.q.bias.copy_(original.q.bias)
                exportable.k.bias.copy_(original.k.bias)
                exportable.v.bias.copy_(original.v.bias)

            # Copy output projection
            exportable.proj.weight.copy_(original.proj.weight)
            if original.proj.bias is not None:
                exportable.proj.bias.copy_(original.proj.bias)

            # Expand relative position bias table to full matrix
            # Original: [table_size, num_heads] with index lookup
            # Exportable: [num_heads, ws*ws, ws*ws] pre-expanded
            rel_pos_idx = original.relative_position_index  # [ws*ws, ws*ws]
            rel_pos_table = original.relative_position_bias_table  # [table_size, num_heads]

            # Gather and reshape
            ws_sq = ws * ws
            expanded_bias = rel_pos_table[rel_pos_idx.view(-1)].view(ws_sq, ws_sq, -1)
            exportable.relative_position_bias.copy_(expanded_bias.permute(2, 0, 1))

        return exportable


def window_partition_static(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Static window partition for fixed input resolution.

    Args:
        x: [B, H, W, C]
        window_size: Size of windows

    Returns:
        windows: [B*num_windows, window_size*window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse_static(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """
    Static window reverse for fixed output resolution.

    Args:
        windows: [B*num_windows, window_size*window_size, C]
        window_size: Size of windows
        H, W: Output spatial dimensions

    Returns:
        x: [B, H, W, C]
    """
    num_windows = (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ExportFriendlyTransformerBlock(nn.Module):
    """
    Export-friendly standard Transformer block.

    Drop-in replacement for UnifiedTransformerBlock with static shapes.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        H, W = input_resolution

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = ExportFriendlyMultiheadAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Positional embeddings
        self.q_pos_embedding = nn.Parameter(torch.randn(1, H * W, dim))
        self.k_pos_embedding = nn.Parameter(torch.randn(1, H * W, dim))

        self.norm_ffn = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, C, H, W]
            key: Optional [B, C, H, W] for cross-attention
            value: Optional [B, C, H, W] for cross-attention

        Returns:
            output: [B, C, H, W]
        """
        B, C, H, W = query.shape

        if key is None:
            key = query
            value = query

        # Flatten to sequence
        q_in = query.flatten(2).transpose(1, 2)  # [B, H*W, C]
        k_in = key.flatten(2).transpose(1, 2)
        v_in = value.flatten(2).transpose(1, 2)
        shortcut = v_in

        # Add positional embeddings and normalize
        q_norm = self.norm_q(q_in + self.q_pos_embedding)
        k_norm = self.norm_kv(k_in + self.k_pos_embedding)
        v_norm = self.norm_kv(v_in)

        # Attention
        attn_output, _ = self.attn(query=q_norm, key=k_norm, value=v_norm)

        # Residual + FFN
        x = shortcut + attn_output
        x = x + self.mlp(self.norm_ffn(x))

        return x.transpose(1, 2).view(B, C, H, W)

    @classmethod
    def from_original(cls, original: 'UnifiedTransformerBlock') -> 'ExportFriendlyTransformerBlock':
        """Convert from original UnifiedTransformerBlock."""
        dim = original.norm_q.normalized_shape[0]
        # Infer resolution from positional embedding shape
        seq_len = original.q_pos_embedding.shape[1]
        import math
        res = int(math.sqrt(seq_len))

        exportable = cls(
            dim=dim,
            input_resolution=(res, res),
            num_heads=original.attn.num_heads,
            mlp_ratio=2.0,  # Default, may need adjustment
            qkv_bias=True,
        )

        # Transfer attention weights
        exportable.attn = ExportFriendlyMultiheadAttention.from_original(original.attn)

        with torch.no_grad():
            # Copy norms
            exportable.norm_q.weight.copy_(original.norm_q.weight)
            exportable.norm_q.bias.copy_(original.norm_q.bias)
            exportable.norm_kv.weight.copy_(original.norm_kv.weight)
            exportable.norm_kv.bias.copy_(original.norm_kv.bias)
            exportable.norm_ffn.weight.copy_(original.norm_ffn.weight)
            exportable.norm_ffn.bias.copy_(original.norm_ffn.bias)

            # Copy positional embeddings
            exportable.q_pos_embedding.copy_(original.q_pos_embedding)
            exportable.k_pos_embedding.copy_(original.k_pos_embedding)

            # Copy MLP weights
            for i, (exp_layer, orig_layer) in enumerate(zip(exportable.mlp, original.mlp)):
                if hasattr(exp_layer, 'weight'):
                    exp_layer.weight.copy_(orig_layer.weight)
                if hasattr(exp_layer, 'bias') and exp_layer.bias is not None:
                    exp_layer.bias.copy_(orig_layer.bias)

        return exportable


class ExportFriendlySwinBlock(nn.Module):
    """
    Export-friendly Swin Transformer block.

    Drop-in replacement for UnifiedSwinBlock with static window operations.
    """

    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        H, W = input_resolution
        if min(H, W) <= window_size:
            self.shift_size = 0
            self.window_size = min(H, W)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn = ExportFriendlySwinAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

        # Pre-compute shift mask
        if self.shift_size > 0:
            attn_mask = self._compute_attn_mask(H, W)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def _compute_attn_mask(self, H: int, W: int) -> torch.Tensor:
        """Pre-compute attention mask for shifted windows."""
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition_static(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, C, H, W]
            key: Optional [B, C, H, W]
            value: Optional [B, C, H, W]

        Returns:
            output: [B, C, H, W]
        """
        B, C, H, W = query.shape

        if key is None:
            key = query
            value = query

        # Flatten to sequence
        q = query.flatten(2).transpose(1, 2)
        k = key.flatten(2).transpose(1, 2)
        v = value.flatten(2).transpose(1, 2)
        shortcut = v

        # Normalize and reshape to spatial
        q = self.norm_q(q).view(B, H, W, C)
        k = self.norm_kv(k).view(B, H, W, C)
        v = self.norm_kv(v).view(B, H, W, C)

        # Apply cyclic shift
        if self.shift_size > 0:
            shifted_q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_k = torch.roll(k, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_q, shifted_k, shifted_v = q, k, v

        # Partition into windows
        q_win = window_partition_static(shifted_q, self.window_size)
        k_win = window_partition_static(shifted_k, self.window_size)
        v_win = window_partition_static(shifted_v, self.window_size)

        # Windowed attention
        attn_windows = self.attn(q_win, k_win, v_win, mask=self.attn_mask)

        # Reverse windows
        shifted_x = window_reverse_static(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Residual + FFN
        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm_ffn(x))

        return x.transpose(1, 2).view(B, C, H, W)

    @classmethod
    def from_original(cls, original: 'UnifiedSwinBlock') -> 'ExportFriendlySwinBlock':
        """Convert from original UnifiedSwinBlock."""
        exportable = cls(
            dim=original.norm_q.normalized_shape[0],
            input_resolution=original.input_resolution,
            num_heads=original.attn.num_heads,
            window_size=original.window_size,
            shift_size=original.shift_size,
            mlp_ratio=2.0,
            qkv_bias=True,
        )

        # Transfer attention weights
        exportable.attn = ExportFriendlySwinAttention.from_original(original.attn)

        with torch.no_grad():
            # Copy norms
            exportable.norm_q.weight.copy_(original.norm_q.weight)
            exportable.norm_q.bias.copy_(original.norm_q.bias)
            exportable.norm_kv.weight.copy_(original.norm_kv.weight)
            exportable.norm_kv.bias.copy_(original.norm_kv.bias)
            exportable.norm_ffn.weight.copy_(original.norm_ffn.weight)
            exportable.norm_ffn.bias.copy_(original.norm_ffn.bias)

            # Copy MLP weights
            for exp_layer, orig_layer in zip(exportable.mlp, original.mlp):
                if hasattr(exp_layer, 'weight'):
                    exp_layer.weight.copy_(orig_layer.weight)
                if hasattr(exp_layer, 'bias') and exp_layer.bias is not None:
                    exp_layer.bias.copy_(orig_layer.bias)

            # Copy attention mask if present
            if original.attn_mask is not None and exportable.attn_mask is not None:
                exportable.attn_mask.copy_(original.attn_mask)

        return exportable


class ExportFriendlyCrossAttention(nn.Module):
    """
    Export-friendly cross attention module.

    Replaces CrossAttention with fully exportable operations.
    For high-resolution features, uses ExportFriendlyUpsampler instead of GuidedResampler.
    For low-resolution features, uses ExportFriendlyMultiheadAttention.
    """

    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        num_heads: int,
        swin_res_threshold: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.is_standard_attention = resolution[0] < swin_res_threshold

        if self.is_standard_attention:
            self.block_efc = ExportFriendlyMultiheadAttention(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=True,
            )
        else:
            anchor_resolution = swin_res_threshold
            ratio = 2 * (resolution[0] / anchor_resolution)
            assert ratio >= 1 and ratio == int(ratio), \
                "Fine resolution must be a multiple of anchor resolution"
            self.block = ExportFriendlyUpsampler(
                dim=dim,
                upsample_ratio=int(ratio),
                num_heads=num_heads,
            )

    def forward(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: Optional[torch.Tensor] = None,
        attn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            A: Query features [B, C, H, W]
            B: Key features [B, C, H, W]
            C: Value features [B, C, H, W]
            D: Unused (kept for compatibility)
            attn: Optional attention map from coarse level

        Returns:
            output: [B, C, H, W]
            attn_map: Optional attention weights
        """
        if not self.is_standard_attention:
            # High resolution path - use upsampler
            out = self.block(C, attn.mean(dim=1) if attn is not None else None)
            return out, None
        else:
            # Standard attention path
            B_, C_, H, W = A.shape
            A_seq = A.flatten(2).transpose(1, 2)
            B_seq = B.flatten(2).transpose(1, 2)
            C_seq = C.flatten(2).transpose(1, 2)

            out_seq, attn_map = self.block_efc(A_seq, B_seq, C_seq)
            out = out_seq.transpose(1, 2).view(B_, C_, H, W)
            return out, attn_map

    @classmethod
    def from_original(
        cls,
        original: 'CrossAttention',
        num_heads: int = 8,
        swin_res_threshold: int = 32,
    ) -> 'ExportFriendlyCrossAttention':
        """Convert from original CrossAttention."""
        # Infer resolution and dim from the original module
        if original.is_standard_attention:
            dim = original.block_efc.q_proj.in_features
            # Resolution is harder to infer, will use a placeholder
            resolution = (16, 16)  # Default, should be passed explicitly
        else:
            dim = original.block.dim
            resolution = (64, 64)  # Default for high-res

        exportable = cls(
            dim=dim,
            resolution=resolution,
            num_heads=num_heads,
            swin_res_threshold=swin_res_threshold,
        )

        # Transfer weights for standard attention path
        if original.is_standard_attention:
            exportable.block_efc = ExportFriendlyMultiheadAttention.from_original(
                original.block_efc
            )
        # Note: GuidedResampler has no learnable weights, so ExportFriendlyUpsampler
        # will need training/fine-tuning

        return exportable


class ExportFriendlySelfAttention(nn.Module):
    """
    Export-friendly self attention module.

    Replaces SelfAttention with exportable Swin or Transformer blocks.
    """

    def __init__(
        self,
        dim: int,
        resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        swin_res_threshold: int = 32,
    ):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.use_swin = resolution[0] >= swin_res_threshold

        self.blocks = nn.ModuleList()

        if self.use_swin:
            # Use Swin blocks for high resolution
            self.blocks.append(ExportFriendlySwinBlock(
                dim=dim,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
            ))
            self.blocks.append(ExportFriendlySwinBlock(
                dim=dim,
                input_resolution=resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2,
            ))
        else:
            # Use standard transformer for low resolution
            self.blocks.append(ExportFriendlyTransformerBlock(
                dim=dim,
                input_resolution=resolution,
                num_heads=num_heads,
                mlp_ratio=2.0,
            ))

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [B, C, H, W]
            key: Optional [B, C, H, W]
            value: Optional [B, C, H, W]

        Returns:
            output: [B, C, H, W]
        """
        if key is not None:
            # Cross-attention mode
            v_out = value
            for block in self.blocks:
                v_out = block(query, key, v_out)
            return v_out
        else:
            # Self-attention mode
            x_out = query
            for block in self.blocks:
                x_out = block(x_out)
            return x_out

    @classmethod
    def from_original(
        cls,
        original: 'SelfAttention',
        window_size: int = 8,
        swin_res_threshold: int = 32,
    ) -> 'ExportFriendlySelfAttention':
        """Convert from original SelfAttention."""
        # Get info from first block
        first_block = original.blocks[0]
        dim = first_block.norm_q.normalized_shape[0]

        if hasattr(first_block, 'input_resolution'):
            resolution = first_block.input_resolution
        else:
            resolution = (16, 16)  # Default

        if hasattr(first_block, 'attn'):
            num_heads = first_block.attn.num_heads
        else:
            num_heads = 8

        exportable = cls(
            dim=dim,
            resolution=resolution,
            num_heads=num_heads,
            window_size=window_size,
            swin_res_threshold=swin_res_threshold,
        )

        # Transfer weights from original blocks
        for i, (exp_block, orig_block) in enumerate(zip(exportable.blocks, original.blocks)):
            if isinstance(orig_block, type(original.blocks[0])):
                if hasattr(orig_block, 'window_size'):
                    # Swin block
                    exportable.blocks[i] = ExportFriendlySwinBlock.from_original(orig_block)
                else:
                    # Transformer block
                    exportable.blocks[i] = ExportFriendlyTransformerBlock.from_original(orig_block)

        return exportable


def convert_attention_modules(model: nn.Module, args=None) -> nn.Module:
    """
    Convert all attention modules in a model to export-friendly versions.

    This function recursively traverses the model and replaces:
    - StandardUnifiedAttention -> ExportFriendlyMultiheadAttention
    - GuidedResampler -> ExportFriendlyUpsampler
    - SwinUnifiedAttention -> ExportFriendlySwinAttention
    - UnifiedTransformerBlock -> ExportFriendlyTransformerBlock
    - UnifiedSwinBlock -> ExportFriendlySwinBlock
    - CrossAttention -> ExportFriendlyCrossAttention
    - SelfAttention -> ExportFriendlySelfAttention

    Args:
        model: The model to convert
        args: Optional args namespace with num_heads, window_size, swin_res_threshold

    Returns:
        The converted model (modified in-place)
    """
    from renderer.attention_modules import (
        StandardUnifiedAttention, GuidedResampler, SwinUnifiedAttention,
        UnifiedTransformerBlock, UnifiedSwinBlock, CrossAttention, SelfAttention
    )

    # Default args
    num_heads = getattr(args, 'num_heads', 8) if args else 8
    window_size = getattr(args, 'window_size', 8) if args else 8
    swin_res_threshold = getattr(args, 'swin_res_threshold', 32) if args else 32

    for name, module in model.named_children():
        if isinstance(module, StandardUnifiedAttention):
            setattr(model, name, ExportFriendlyMultiheadAttention.from_original(module))
        elif isinstance(module, GuidedResampler):
            setattr(model, name, ExportFriendlyUpsampler.from_original(module, num_heads))
        elif isinstance(module, SwinUnifiedAttention):
            setattr(model, name, ExportFriendlySwinAttention.from_original(module))
        elif isinstance(module, UnifiedTransformerBlock):
            setattr(model, name, ExportFriendlyTransformerBlock.from_original(module))
        elif isinstance(module, UnifiedSwinBlock):
            setattr(model, name, ExportFriendlySwinBlock.from_original(module))
        elif isinstance(module, CrossAttention):
            setattr(model, name, ExportFriendlyCrossAttention.from_original(
                module, num_heads, swin_res_threshold
            ))
        elif isinstance(module, SelfAttention):
            setattr(model, name, ExportFriendlySelfAttention.from_original(
                module, window_size, swin_res_threshold
            ))
        else:
            # Recursively convert children
            convert_attention_modules(module, args)

    return model
