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

    The original GuidedResampler uses dynamic top-k selection with modulo operations
    (topk_indices % W_low) which ONNX cannot export. This replacement uses:

    1. Bilinear upsampling of coarse attention map to guide high-res features
    2. Learned convolutional refinement (no attention at high-res!)
    3. All operations are fully static and ONNX-exportable

    Key insight: The coarse attention map already captures spatial relationships.
    We can upsample it and use it as soft weights without dynamic indexing.

    TF.js FIX: Pre-computed spatial dimensions eliminate Shape ops.
    """

    def __init__(
        self,
        dim: int,
        upsample_ratio: int = 4,
        num_heads: int = 8,  # kept for API compatibility, not used
        high_res: Tuple[int, int] = (64, 64),  # Pre-defined output resolution
        low_res: Tuple[int, int] = (16, 16),   # Pre-defined coarse resolution
    ):
        super().__init__()
        self.dim = dim
        self.ratio = upsample_ratio

        # TF.js FIX: Store spatial dimensions as constants to avoid Shape ops
        self.H_high, self.W_high = high_res
        self.H_low, self.W_low = low_res

        # Learnable refinement after attention-guided upsampling
        # This replaces the sparse sampling with learned local processing
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=min(dim, 32)),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

        # Attention map processing - converts attention weights to spatial guidance
        # Input: [B, N_low, N_low] -> Output: [B, 1, H_low, W_low]
        self.attn_proj = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        v_high_feat: torch.Tensor,
        coarse_attn_map: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            v_high_feat: High resolution features [B, C, H, W]
            coarse_attn_map: Attention map from coarse level [B, N_low, N_low]
                            Used to guide upsampling. If None, uses uniform weights.

        Returns:
            warped_feat: Refined high-res features [B, C, H, W]
        """
        # NOTE: For ExportFriendlyUpsampler, we need to extract actual spatial dims
        # because the input sizes vary at different decoder stages.
        # The Shape ops overhead is acceptable for high-res features.
        _, _, H, W = v_high_feat.shape

        if coarse_attn_map is not None:
            # Compute low-res spatial dims from attention map sequence length
            # attn_map: [B, N_low, N_low] where N_low = H_low * W_low
            N_low = coarse_attn_map.shape[1]
            # Assume square spatial dimensions
            import math
            H_low = int(math.sqrt(N_low))
            W_low = H_low

            # Sum attention received by each position as importance score
            attn_importance = coarse_attn_map.sum(dim=-1)  # [B, N_low]

            # Reshape to spatial: [B, N_low] -> [B, 1, H_low, W_low]
            attn_importance = attn_importance.view(-1, H_low, W_low).unsqueeze(1)

            # Upsample attention importance to high-res using bilinear interpolation
            attn_upsampled = F.interpolate(
                attn_importance,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )  # [B, 1, H, W]

            # Normalize to [0, 1] range for soft gating
            attn_upsampled = torch.sigmoid(attn_upsampled - attn_upsampled.mean())

            # Apply attention-guided weighting
            guided_feat = v_high_feat * (1.0 + attn_upsampled)
        else:
            guided_feat = v_high_feat

        # Apply learned local refinement
        out = self.refine(guided_feat)

        # Residual connection
        out = out + v_high_feat

        return out

    @classmethod
    def from_original(
        cls,
        original: 'GuidedResampler',
        num_heads: int = 8,
        high_res: Tuple[int, int] = (64, 64),
        low_res: Tuple[int, int] = (16, 16),
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
            high_res=high_res,
            low_res=low_res,
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
        batch_size: int = 1,
        num_windows: int = 1,
    ) -> torch.Tensor:
        """
        Args:
            query: [B*num_windows, window_size*window_size, C]
            key: [B*num_windows, window_size*window_size, C]
            value: [B*num_windows, window_size*window_size, C]
            mask: Optional shift mask [num_windows, ws*ws, ws*ws]
            batch_size: Explicit batch size (REQUIRED for TF.js - no dynamic inference)
            num_windows: Explicit number of windows (REQUIRED for TF.js)

        Returns:
            output: [B*num_windows, window_size*window_size, C]
        """
        # TF.js FIX: Use pre-computed dimensions instead of extracting from shape
        # query.shape[0] would create a Shape op
        N = self.window_size * self.window_size
        C = self.dim

        # Project Q, K, V
        # TF.js FIX: Use -1 for first dim to let PyTorch infer it without Shape op
        q = self.q(query).view(-1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key).view(-1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(value).view(-1, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention scores
        attn = (q * self.scale) @ k.transpose(-2, -1)  # [B_, num_heads, N, N]

        # Add pre-computed relative position bias
        attn = attn + self.relative_position_bias.unsqueeze(0)

        # Apply shift mask if provided
        if mask is not None:
            # TF.js FIX: Use explicitly passed values - no dynamic shape computation
            nW = num_windows
            B = batch_size
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(B * nW, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        # TF.js FIX: Use -1 for batch dim to avoid Shape op
        x = (attn @ v).transpose(1, 2).reshape(-1, N, C)

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


class StaticWindowPartition(nn.Module):
    """
    Static window partition as a module with pre-computed dimensions.
    This avoids dynamic shape computations during ONNX export.

    TF.js FIX: Uses -1 in reshape to avoid Shape ops.
    """

    def __init__(self, H: int, W: int, window_size: int, dim: int = 256):
        super().__init__()
        self.H = H
        self.W = W
        self.window_size = window_size
        self.num_h = H // window_size
        self.num_w = W // window_size
        self.num_windows = self.num_h * self.num_w
        self.window_area = window_size * window_size
        self.dim = dim  # Store channel dimension for static reshapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C]
        Returns:
            windows: [B*num_windows, window_size*window_size, C]
        """
        # TF.js FIX: Use -1 to avoid extracting batch and channel from shape
        # This prevents Shape ops in the ONNX graph
        # Reshape: [B, H, W, C] -> [B, num_h, ws, num_w, ws, C]
        x = x.view(-1, self.num_h, self.window_size, self.num_w, self.window_size, self.dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        # Reshape to [B*num_windows, ws*ws, C]
        windows = x.view(-1, self.window_area, self.dim)
        return windows


class StaticWindowReverse(nn.Module):
    """
    Static window reverse as a module with pre-computed dimensions.
    This avoids dynamic shape computations during ONNX export.

    TF.js FIX: Uses explicit batch_size and pre-stored dim to avoid Shape ops.
    """

    def __init__(self, H: int, W: int, window_size: int, dim: int = 256):
        super().__init__()
        self.H = H
        self.W = W
        self.window_size = window_size
        self.num_h = H // window_size
        self.num_w = W // window_size
        self.num_windows = self.num_h * self.num_w
        self.dim = dim  # Store channel dimension for static reshapes

    def forward(self, windows: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """
        Args:
            windows: [B*num_windows, window_size*window_size, C]
            batch_size: The batch size B (REQUIRED for TF.js - default=1 for inference)
        Returns:
            x: [B, H, W, C]
        """
        # TF.js FIX: Use pre-stored dim instead of windows.shape[2]
        C = self.dim
        x = windows.view(batch_size, self.num_h, self.num_w, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(batch_size, self.H, self.W, C)
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
        # TF.js FIX: Use pre-stored dimensions instead of extracting from shape
        # query.shape would create Shape ops in ONNX
        H, W = self.input_resolution
        C = self.dim

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

        # TF.js FIX: Use unflatten to reshape without needing channel dim
        # x: [B, H*W, C] -> transpose -> [B, C, H*W] -> unflatten -> [B, C, H, W]
        return x.transpose(1, 2).unflatten(2, (H, W))

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

    Drop-in replacement for UnifiedSwinBlock with FULLY STATIC window operations.
    All dimensions are pre-computed at init time to avoid dynamic shape inference.

    TF.js FIX: Uses batch_size=1 assumption and pre-stored dimensions.
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
        batch_size: int = 1,  # TF.js FIX: Default batch size for static export
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.batch_size = batch_size  # TF.js FIX: Store for static reshapes

        H, W = input_resolution
        if min(H, W) <= window_size:
            self.shift_size = 0
            self.window_size = min(H, W)

        # Store as Python ints for static computation
        self.H = H
        self.W = W

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

        # TF.js FIX: Static window modules with pre-computed dimensions INCLUDING dim
        self.window_partition = StaticWindowPartition(H, W, self.window_size, dim=dim)
        self.window_reverse = StaticWindowReverse(H, W, self.window_size, dim=dim)

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
        # TF.js FIX: Use pre-stored dimensions instead of extracting from shape
        # query.shape[0] creates a Shape op which onnx2tf can't handle
        B = self.batch_size  # Use stored batch size (default=1 for inference)
        C = self.dim
        H = self.H
        W = self.W

        if key is None:
            key = query
            value = query

        # Flatten to sequence
        q = query.flatten(2).transpose(1, 2)
        k = key.flatten(2).transpose(1, 2)
        v = value.flatten(2).transpose(1, 2)
        shortcut = v

        # TF.js FIX: Normalize and reshape to spatial using -1 for batch dim
        # view(B, H, W, C) with dynamic B creates Shape op, but -1 doesn't
        q = self.norm_q(q).view(-1, H, W, C)
        k = self.norm_kv(k).view(-1, H, W, C)
        v = self.norm_kv(v).view(-1, H, W, C)

        # Apply cyclic shift
        if self.shift_size > 0:
            shifted_q = torch.roll(q, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_k = torch.roll(k, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_q, shifted_k, shifted_v = q, k, v

        # Partition into windows using static module
        q_win = self.window_partition(shifted_q)
        k_win = self.window_partition(shifted_k)
        v_win = self.window_partition(shifted_v)

        # Windowed attention - pass batch_size and num_windows explicitly for static export
        attn_windows = self.attn(
            q_win, k_win, v_win,
            mask=self.attn_mask,
            batch_size=B,
            num_windows=self.window_partition.num_windows
        )

        # Reverse windows using static module (pass batch size explicitly)
        shifted_x = self.window_reverse(attn_windows, B)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # Residual + FFN
        # TF.js FIX: Use flatten/unflatten to reshape without Shape ops
        x = x.flatten(1, 2)  # [B, H, W, C] -> [B, H*W, C]
        x = shortcut + x
        x = x + self.mlp(self.norm_ffn(x))

        # TF.js FIX: Use unflatten to reshape without needing channel dim
        return x.transpose(1, 2).unflatten(2, (H, W))

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

    def coarse_stage(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Coarse stage attention (standard attention path).

        Args:
            A: Query features [B, C, H, W]
            B: Key features [B, C, H, W]
            C: Value features [B, C, H, W]
            attn: Unused

        Returns:
            output: [B, C, H, W]
            attn_map: [B, num_heads, N, N]
        """
        # NOTE: For coarse_stage (low-res path), we extract shape from tensor
        # This creates Shape ops, but coarse_stage is only used for small tensors
        # where the overhead is minimal. The alternative (incorrect resolution) breaks.
        _, _, H, W = A.shape

        # A: [B, C, H, W] -> [B, H*W, C]
        A_seq = A.flatten(2).transpose(1, 2)
        B_seq = B.flatten(2).transpose(1, 2)
        C_seq = C.flatten(2).transpose(1, 2)

        # Attention: [B, H*W, C] -> [B, H*W, C]
        out_seq, attn_map = self.block_efc(A_seq, B_seq, C_seq)

        # Reshape back to spatial: [B, H*W, C] -> [B, C, H, W]
        out = out_seq.transpose(1, 2).unflatten(2, (H, W))
        return out, attn_map

    def fine_stage(
        self,
        C: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fine stage attention (high-res upsampler path).

        Args:
            C: Value features [B, C, H, W]
            attn: Attention map from coarse level

        Returns:
            output: [B, C, H, W]
        """
        attn_input = attn.mean(dim=1) if attn is not None else None
        out = self.block(C, attn_input)
        return out

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
            out = self.fine_stage(C, attn)
            return out
        else:
            # Standard attention path
            return self.coarse_stage(A, B, C, attn)

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
        elif isinstance(module, nn.ModuleList):
            # Handle ModuleList - convert each element
            for i, submodule in enumerate(module):
                if isinstance(submodule, StandardUnifiedAttention):
                    module[i] = ExportFriendlyMultiheadAttention.from_original(submodule)
                elif isinstance(submodule, GuidedResampler):
                    module[i] = ExportFriendlyUpsampler.from_original(submodule, num_heads)
                elif isinstance(submodule, SwinUnifiedAttention):
                    module[i] = ExportFriendlySwinAttention.from_original(submodule)
                elif isinstance(submodule, UnifiedTransformerBlock):
                    module[i] = ExportFriendlyTransformerBlock.from_original(submodule)
                elif isinstance(submodule, UnifiedSwinBlock):
                    module[i] = ExportFriendlySwinBlock.from_original(submodule)
                elif isinstance(submodule, CrossAttention):
                    module[i] = ExportFriendlyCrossAttention.from_original(
                        submodule, num_heads, swin_res_threshold
                    )
                elif isinstance(submodule, SelfAttention):
                    module[i] = ExportFriendlySelfAttention.from_original(
                        submodule, window_size, swin_res_threshold
                    )
                else:
                    # Recursively convert children of this submodule
                    convert_attention_modules(submodule, args)
        else:
            # Recursively convert children
            convert_attention_modules(module, args)

    return model
