"""
Export-Friendly LIA ResBlocks for ONNX/TF.js Compatibility
==========================================================

This module provides ONNX-exportable replacements for the StyleGAN2-based
blocks in lia_resblocks.py. The key changes:

1. EqualLinear/EqualConv2d: Bake scale into weights (no runtime multiplication)
2. ModulatedConv2d: Replace per-sample weight modulation with equivalent
   standard operations (feature modulation instead of weight modulation)

The export-friendly versions load weights from the original checkpoint and
produce numerically equivalent (within FP32 tolerance) outputs.

Usage:
    # Load original model
    original = IMTRenderer(args)
    original.load_state_dict(torch.load('renderer.ckpt')['state_dict'])

    # Convert to export-friendly version
    from renderer.lia_resblocks_exportable import convert_to_exportable
    exportable = convert_to_exportable(original)

    # Verify parity
    with torch.no_grad():
        out_orig = original(x_current, x_reference)
        out_export = exportable(x_current, x_reference)
        print(f"Parity: {torch.allclose(out_orig[0], out_export[0], atol=1e-5)}")

    # Export to ONNX
    torch.onnx.export(exportable, ...)
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Optional, Tuple


# =============================================================================
# EXPORT-FRIENDLY BASIC BLOCKS
# =============================================================================

class ExportableEqualLinear(nn.Module):
    """
    Export-friendly version of EqualLinear.

    Original EqualLinear multiplies weights by scale at runtime:
        out = F.linear(input, self.weight * self.scale, bias)

    This version bakes the scale into the weights during construction,
    making it a standard nn.Linear that ONNX can handle.

    Weight Transfer:
        new_weight = old_weight * old_scale
        new_bias = old_bias * old_lr_mul (if activation) else old_bias
    """
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, activation: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        self.scale = 2 ** 0.5  # For fused_leaky_relu scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.activation:
            # Fused leaky relu with scaling: leaky_relu(x) * sqrt(2)
            out = F.leaky_relu(out, negative_slope=0.2) * self.scale
        return out

    @classmethod
    def from_original(cls, original_module) -> 'ExportableEqualLinear':
        """
        Create from original EqualLinear, baking in the scale.
        """
        in_dim = original_module.weight.shape[1]
        out_dim = original_module.weight.shape[0]
        has_bias = original_module.bias is not None
        has_activation = original_module.activation is not None

        new_module = cls(in_dim, out_dim, bias=has_bias, activation=has_activation)

        with torch.no_grad():
            # Bake scale into weights
            scale = original_module.scale
            new_module.linear.weight.copy_(original_module.weight * scale)

            if has_bias:
                # Bias scaling depends on whether activation is used
                lr_mul = original_module.lr_mul
                if has_activation:
                    # For fused_leaky_relu, bias is added before activation
                    new_module.linear.bias.copy_(original_module.bias * lr_mul)
                else:
                    new_module.linear.bias.copy_(original_module.bias * lr_mul)

        return new_module


class ExportableEqualConv2d(nn.Module):
    """
    Export-friendly version of EqualConv2d.

    Original EqualConv2d multiplies weights by scale at runtime:
        out = F.conv2d(input, self.weight * self.scale, bias, ...)

    This version bakes the scale into weights, making it a standard Conv2d.
    """
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size,
                              stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    @classmethod
    def from_original(cls, original_module) -> 'ExportableEqualConv2d':
        """
        Create from original EqualConv2d, baking in the scale.
        """
        out_ch, in_ch, k, _ = original_module.weight.shape
        has_bias = original_module.bias is not None

        new_module = cls(
            in_ch, out_ch, k,
            stride=original_module.stride,
            padding=original_module.padding,
            bias=has_bias
        )

        with torch.no_grad():
            # Bake scale into weights
            scale = original_module.scale
            new_module.conv.weight.copy_(original_module.weight * scale)
            if has_bias:
                new_module.conv.bias.copy_(original_module.bias)

        return new_module


class ExportableFusedLeakyReLU(nn.Module):
    """
    Export-friendly version of FusedLeakyReLU.

    Original adds a learnable bias then applies scaled leaky relu.
    This version does the same but with explicit operations.
    """
    def __init__(self, channel: int, negative_slope: float = 0.2, scale: float = 2**0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x + self.bias, self.negative_slope) * self.scale

    @classmethod
    def from_original(cls, original_module) -> 'ExportableFusedLeakyReLU':
        new_module = cls(
            original_module.bias.shape[1],
            original_module.negative_slope,
            original_module.scale
        )
        with torch.no_grad():
            new_module.bias.copy_(original_module.bias)
        return new_module


# =============================================================================
# EXPORT-FRIENDLY MODULATED CONV2D - THE KEY CHALLENGE
# =============================================================================

class ExportableModulatedConv2d(nn.Module):
    """
    Export-friendly version of ModulatedConv2d.

    PROBLEM WITH ORIGINAL:
    ---------------------
    The original ModulatedConv2d does per-sample weight modulation:
        1. style = modulation(style_input)  # [B, in_ch]
        2. weight = base_weight * style.view(B, 1, in_ch, 1, 1)  # Per-sample!
        3. weight = weight.view(B * out_ch, in_ch, k, k)
        4. F.conv2d(input.view(1, B*in_ch, H, W), weight, groups=B)

    This uses groups=batch which creates dynamic graph structures that
    ONNX/TF.js cannot handle.

    SOLUTION - FEATURE MODULATION:
    -----------------------------
    Instead of modulating weights, we modulate the INPUT features:
        1. style = modulation(style_input)  # [B, in_ch]
        2. x_mod = x * style.view(B, in_ch, 1, 1)  # Modulate input
        3. out = conv(x_mod)  # Standard convolution
        4. if demodulate: out = out / demod_factor

    This is mathematically equivalent for the forward pass because:
        conv(x * s, w) = conv(x, w * s)  (when s is per-channel)

    The demodulation step is slightly different but produces equivalent
    results in practice (we compute it based on style rather than weights).
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        downsample: bool = False,
        blur_kernel: list = [1, 3, 3, 1]
    ):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.demodulate = demodulate
        self.upsample = upsample
        self.downsample = downsample
        self.padding = kernel_size // 2

        # Scale factor - will be baked into weights
        fan_in = in_channel * kernel_size ** 2
        self.register_buffer('scale', torch.tensor(1 / math.sqrt(fan_in)))

        # Style modulation: maps style_dim -> in_channel (for input modulation)
        self.modulation = ExportableEqualLinear(style_dim, in_channel, bias=True)

        # Main convolution weight - stored PRE-SCALED for ONNX compatibility
        # During from_original(), we multiply by scale so forward() uses static weights
        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))

        # For upsample: pre-transposed weight to avoid runtime transpose
        # This is set during from_original() or bake_weights()
        if upsample:
            self.weight_t = nn.Parameter(torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        else:
            self.weight_t = None

        # Pre-computed weight squared sum for demodulation (set during bake_weights)
        # This avoids runtime ReduceSum which onnx2tf can't convert properly
        if demodulate:
            self.register_buffer('w_sq_per_ch', torch.zeros(out_channel, in_channel))
        else:
            self.w_sq_per_ch = None

        # Track if weights have been baked (scale applied)
        self._weights_baked = False

        # Blur for up/downsampling
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = ExportableBlur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blur = ExportableBlur(blur_kernel, pad=(pad0, pad1))

    def bake_weights(self):
        """
        Bake scale into weights for ONNX export compatibility.
        This makes weights static tensors instead of runtime-computed.
        Also pre-computes w_sq_per_ch to eliminate runtime ReduceSum.
        """
        if self._weights_baked:
            return

        with torch.no_grad():
            # Apply scale to weight
            self.weight.mul_(self.scale.item())

            # For upsample, create pre-transposed weight
            if self.upsample and self.weight_t is not None:
                self.weight_t.copy_(self.weight.transpose(0, 1).contiguous())

            # Pre-compute weight squared sum for demodulation
            # This eliminates runtime ReduceSum which onnx2tf can't handle
            if self.demodulate and self.w_sq_per_ch is not None:
                # [out_ch, in_ch, k, k] -> sum over spatial dims -> [out_ch, in_ch]
                self.w_sq_per_ch.copy_((self.weight ** 2).sum(dim=[2, 3]))

        self._weights_baked = True

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using feature modulation instead of weight modulation.

        For ONNX/TF.js compatibility:
        - Weights are PRE-SCALED (scale baked in during from_original)
        - For upsample, we use PRE-TRANSPOSED weights (weight_t)
        - No runtime weight computation = static graph = clean export

        Args:
            x: Input features [B, in_channel, H, W]
            style: Style vector [B, style_dim]

        Returns:
            out: Modulated output [B, out_channel, H', W']
        """
        # Note: We avoid extracting batch dimension (x.shape[0]) to prevent
        # ONNX Shape ops which onnx2tf can't handle properly

        # Get modulation factors: [B, in_channel]
        style_mod = self.modulation(style)  # [B, in_channel]

        # === FEATURE MODULATION (replaces weight modulation) ===
        # Modulate input features by style
        # x_mod[b, c, h, w] = x[b, c, h, w] * style_mod[b, c]
        # Use unsqueeze instead of view(batch, ...) to avoid Shape ops
        x_mod = x * style_mod.unsqueeze(-1).unsqueeze(-1)  # [B, in_ch, 1, 1]

        # Use pre-baked weights (scale already applied)
        # If not baked yet (e.g., fresh init), apply scale at runtime
        if self._weights_baked:
            weight = self.weight  # Already scaled
        else:
            weight = self.weight * self.scale  # Runtime scaling (for training)

        # === DEMODULATION FACTOR ===
        if self.demodulate:
            # Original demodulation: normalize by sqrt(sum(w^2 * s^2))
            # Since we modulated x instead of w, we compute:
            # demod = 1 / sqrt(sum_ic(w_oc_ic^2 * s_ic^2)) per output channel

            # Use pre-computed weight squared sum (baked during export)
            # This avoids runtime ReduceSum which onnx2tf can't convert
            if self._weights_baked and self.w_sq_per_ch is not None:
                w_sq_per_ch = self.w_sq_per_ch  # [out_ch, in_ch] - static buffer
            else:
                # Runtime computation (for training/non-baked inference)
                w_sq_per_ch = (weight ** 2).sum(dim=[2, 3])  # [out_ch, in_ch]

            # Style squared: [B, in_channel]
            s_sq = style_mod ** 2

            # Compute demod factor: [B, out_ch]
            demod_sq = torch.einsum('oi,bi->bo', w_sq_per_ch, s_sq) + 1e-8
            demod = torch.rsqrt(demod_sq)  # [B, out_ch]

        # === CONVOLUTION ===
        if self.upsample:
            # Use pre-transposed weight for ONNX compatibility
            if self._weights_baked and self.weight_t is not None:
                weight_t = self.weight_t  # Pre-transposed, static
            else:
                weight_t = weight.transpose(0, 1).contiguous()  # Runtime transpose

            out = F.conv_transpose2d(x_mod, weight_t, padding=0, stride=2)
            out = self.blur(out)

        elif self.downsample:
            x_blurred = self.blur(x_mod)
            out = F.conv2d(x_blurred, weight, padding=0, stride=2)

        else:
            out = F.conv2d(x_mod, weight, padding=self.padding)

        # Apply demodulation (per-sample output scaling)
        if self.demodulate:
            # Use unsqueeze instead of view(batch, ...) to avoid Shape ops
            out = out * demod.unsqueeze(-1).unsqueeze(-1)  # [B, out_ch, 1, 1]

        return out

    @classmethod
    def from_original(cls, original_module) -> 'ExportableModulatedConv2d':
        """
        Create from original ModulatedConv2d, transferring weights.

        IMPORTANT: This method also BAKES the scale into weights for ONNX export.
        After conversion, weights are pre-scaled and (for upsample) pre-transposed.
        """
        new_module = cls(
            in_channel=original_module.in_channel,
            out_channel=original_module.out_channel,
            kernel_size=original_module.kernel_size,
            style_dim=original_module.modulation.weight.shape[1],  # Get from modulation layer
            demodulate=original_module.demodulate,
            upsample=original_module.upsample,
            downsample=original_module.downsample,
        )

        with torch.no_grad():
            # Transfer main weight (remove leading dim if present)
            orig_weight = original_module.weight
            if orig_weight.dim() == 5:
                orig_weight = orig_weight.squeeze(0)
            new_module.weight.copy_(orig_weight)

            # Transfer modulation layer
            new_module.modulation = ExportableEqualLinear.from_original(original_module.modulation)

            # Note: blur kernels don't need explicit transfer - they're computed from
            # the same [1,3,3,1] specification in both original and exportable.
            # The ExportableBlur constructs its diagonal kernels automatically.

            # BAKE weights for ONNX export (scale applied, transpose pre-computed)
            new_module.bake_weights()

        return new_module


class ExportableBlur(nn.Module):
    """
    Export-friendly blur using explicit per-channel convolution.

    WHY THIS APPROACH:
    ------------------
    Previous approaches failed because:
    1. Depthwise conv (groups=channels) - onnx2tf can't transpose [C,1,kH,kW] weights
    2. Unfold+matmul - legacy ONNX exporter doesn't support Unfold with dynamic sizes
    3. Reshape to [B*C, 1, H, W] - requires dynamic batch*channels computation

    SOLUTION - EXPAND KERNEL TO FULL CONV:
    --------------------------------------
    We pre-expand the blur kernel to a full [C, C, kH, kW] weight tensor where:
    - It's diagonal: channel i only reads from channel i
    - Each diagonal has the same blur kernel

    This is mathematically equivalent to depthwise conv but uses a standard
    conv weight format that both ONNX legacy exporter and onnx2tf can handle.

    Trade-off: Larger weight tensor (C*C*kH*kW instead of C*kH*kW), but
    this is acceptable for the blur kernel sizes (4x4) we use.
    """
    def __init__(self, kernel: list, pad: tuple, upsample_factor: int = 1, max_channels: int = 512):
        super().__init__()

        # Create blur kernel (same as before)
        k = torch.tensor(kernel, dtype=torch.float32)
        if k.ndim == 1:
            k = k[None, :] * k[:, None]
        k = k / k.sum()

        if upsample_factor > 1:
            k = k * (upsample_factor ** 2)

        self.pad = pad
        self.kernel_size = k.shape[0]

        # Store the base kernel for reference
        self.register_buffer('base_kernel', k)

        # Pre-build expanded kernels for common channel counts
        # This avoids runtime expansion which causes ONNX issues
        for c in [128, 256, 512]:
            expanded = self._build_diagonal_kernel(k, c)
            self.register_buffer(f'kernel_{c}', expanded)

    def _build_diagonal_kernel(self, k: torch.Tensor, channels: int) -> torch.Tensor:
        """
        Build a diagonal convolution kernel [C, C, kH, kW] where each
        output channel only depends on the same input channel.
        """
        kh, kw = k.shape
        # Create [C, C, kH, kW] with k on the diagonal
        weight = torch.zeros(channels, channels, kh, kw)
        for i in range(channels):
            weight[i, i] = k
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply blur using pre-expanded diagonal kernel.
        """
        batch, channels, height, width = x.shape

        # Pad input
        x_padded = F.pad(x, [self.pad[0], self.pad[1], self.pad[0], self.pad[1]])

        # Get the appropriate pre-built kernel
        if channels == 512:
            kernel = self.kernel_512
        elif channels == 256:
            kernel = self.kernel_256
        elif channels == 128:
            kernel = self.kernel_128
        else:
            # Fallback: build kernel at runtime (not ideal for export)
            kernel = self._build_diagonal_kernel(self.base_kernel, channels).to(x.device)

        # Apply standard conv2d with diagonal kernel
        out = F.conv2d(x_padded, kernel, padding=0)

        return out


# =============================================================================
# EXPORT-FRIENDLY STYLED CONV
# =============================================================================

class ExportableStyledConv(nn.Module):
    """
    Export-friendly version of StyledConv.

    Combines:
    - ModulatedConv2d (now using feature modulation)
    - NoiseInjection (optional, often disabled during inference)
    - FusedLeakyReLU activation
    """
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        blur_kernel: list = [1, 3, 3, 1],
        demodulate: bool = True
    ):
        super().__init__()

        self.conv = ExportableModulatedConv2d(
            in_channel, out_channel, kernel_size, style_dim,
            upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate
        )

        # Noise weight (usually not used during inference)
        self.noise_weight = nn.Parameter(torch.zeros(1))

        # Activation
        self.activate = ExportableFusedLeakyReLU(out_channel)

    def forward(self, x: torch.Tensor, style: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.conv(x, style)

        if noise is not None:
            out = out + self.noise_weight * noise

        out = self.activate(out)
        return out

    @classmethod
    def from_original(cls, original_module) -> 'ExportableStyledConv':
        """Create from original StyledConv."""
        orig_conv = original_module.conv

        new_module = cls(
            in_channel=orig_conv.in_channel,
            out_channel=orig_conv.out_channel,
            kernel_size=orig_conv.kernel_size,
            style_dim=orig_conv.modulation.weight.shape[1],
            upsample=orig_conv.upsample,
            demodulate=orig_conv.demodulate
        )

        with torch.no_grad():
            # Transfer conv
            new_module.conv = ExportableModulatedConv2d.from_original(orig_conv)

            # Transfer noise weight
            new_module.noise_weight.copy_(original_module.noise.weight)

            # Transfer activation
            new_module.activate = ExportableFusedLeakyReLU.from_original(original_module.activate)

        return new_module


# =============================================================================
# CONVERSION UTILITIES
# =============================================================================

def convert_equal_linear(module: nn.Module) -> nn.Module:
    """Recursively convert EqualLinear to ExportableEqualLinear."""
    from renderer.lia_resblocks import EqualLinear

    if isinstance(module, EqualLinear):
        return ExportableEqualLinear.from_original(module)

    # Handle nn.ModuleList specially
    if isinstance(module, nn.ModuleList):
        for i, child in enumerate(module):
            module[i] = convert_equal_linear(child)
        return module

    # Handle nn.Sequential
    if isinstance(module, nn.Sequential):
        for i, child in enumerate(module):
            module[i] = convert_equal_linear(child)
        return module

    for name, child in module.named_children():
        setattr(module, name, convert_equal_linear(child))

    return module


def convert_equal_conv(module: nn.Module) -> nn.Module:
    """Recursively convert EqualConv2d to ExportableEqualConv2d."""
    from renderer.lia_resblocks import EqualConv2d

    if isinstance(module, EqualConv2d):
        return ExportableEqualConv2d.from_original(module)

    # Handle nn.ModuleList specially
    if isinstance(module, nn.ModuleList):
        for i, child in enumerate(module):
            module[i] = convert_equal_conv(child)
        return module

    # Handle nn.Sequential
    if isinstance(module, nn.Sequential):
        for i, child in enumerate(module):
            module[i] = convert_equal_conv(child)
        return module

    for name, child in module.named_children():
        setattr(module, name, convert_equal_conv(child))

    return module


def convert_styled_conv(module: nn.Module) -> nn.Module:
    """Recursively convert StyledConv to ExportableStyledConv."""
    from renderer.lia_resblocks import StyledConv

    if isinstance(module, StyledConv):
        return ExportableStyledConv.from_original(module)

    # Handle nn.ModuleList specially
    if isinstance(module, nn.ModuleList):
        for i, child in enumerate(module):
            module[i] = convert_styled_conv(child)
        return module

    # Handle nn.Sequential
    if isinstance(module, nn.Sequential):
        for i, child in enumerate(module):
            module[i] = convert_styled_conv(child)
        return module

    for name, child in module.named_children():
        setattr(module, name, convert_styled_conv(child))

    return module


def convert_to_exportable(model: nn.Module, convert_attention: bool = False, args=None) -> nn.Module:
    """
    Convert a model to use export-friendly blocks.

    This recursively replaces:
    - EqualLinear -> ExportableEqualLinear
    - EqualConv2d -> ExportableEqualConv2d
    - StyledConv -> ExportableStyledConv

    And optionally (with convert_attention=True):
    - StandardUnifiedAttention -> ExportFriendlyMultiheadAttention
    - GuidedResampler -> ExportFriendlyUpsampler
    - SwinUnifiedAttention -> ExportFriendlySwinAttention
    - CrossAttention -> ExportFriendlyCrossAttention
    - SelfAttention -> ExportFriendlySelfAttention

    The converted model should produce numerically equivalent outputs
    and be exportable to ONNX/TF.js.

    Args:
        model: Original model (e.g., IMTRenderer)
        convert_attention: Whether to also convert attention modules
        args: Optional args namespace with num_heads, window_size, swin_res_threshold

    Returns:
        Converted model with export-friendly blocks
    """
    import copy
    model = copy.deepcopy(model)

    # Convert in order: styled_conv first (contains modulated_conv), then others
    model = convert_styled_conv(model)
    model = convert_equal_conv(model)
    model = convert_equal_linear(model)

    # Optionally convert attention modules
    if convert_attention:
        from renderer.attention_modules_exportable import convert_attention_modules
        model = convert_attention_modules(model, args)

    return model


# =============================================================================
# PARITY TESTING
# =============================================================================

def test_equal_linear_parity():
    """Test ExportableEqualLinear produces same output as EqualLinear."""
    from renderer.lia_resblocks import EqualLinear

    # Create original
    original = EqualLinear(512, 256, activation='fused_lrelu')
    original.eval()

    # Convert to exportable
    exportable = ExportableEqualLinear.from_original(original)
    exportable.eval()

    # Test
    x = torch.randn(4, 512)
    with torch.no_grad():
        out_orig = original(x)
        out_export = exportable(x)

    max_diff = (out_orig - out_export).abs().max().item()
    print(f"EqualLinear parity - Max diff: {max_diff:.2e}")
    return max_diff < 1e-5


def test_modulated_conv_parity():
    """Test ExportableModulatedConv2d produces same output as ModulatedConv2d."""
    from renderer.lia_resblocks import ModulatedConv2d

    # Create original
    original = ModulatedConv2d(
        in_channel=256, out_channel=256, kernel_size=3,
        style_dim=32, demodulate=True
    )
    original.eval()

    # Convert to exportable
    exportable = ExportableModulatedConv2d.from_original(original)
    exportable.eval()

    # Test
    x = torch.randn(2, 256, 16, 16)
    style = torch.randn(2, 32)

    with torch.no_grad():
        out_orig = original(x, style)
        out_export = exportable(x, style)

    max_diff = (out_orig - out_export).abs().max().item()
    mean_diff = (out_orig - out_export).abs().mean().item()

    print(f"ModulatedConv2d parity - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

    # Check relative error
    rel_error = (out_orig - out_export).abs() / (out_orig.abs() + 1e-8)
    print(f"  Relative error - Max: {rel_error.max().item():.2e}, Mean: {rel_error.mean().item():.2e}")

    return max_diff < 1e-4


def test_styled_conv_parity():
    """Test ExportableStyledConv produces same output as StyledConv."""
    from renderer.lia_resblocks import StyledConv

    # Create original
    original = StyledConv(
        in_channel=256, out_channel=256, kernel_size=3,
        style_dim=32, upsample=False
    )
    original.eval()

    # Convert to exportable
    exportable = ExportableStyledConv.from_original(original)
    exportable.eval()

    # Test
    x = torch.randn(2, 256, 16, 16)
    style = torch.randn(2, 32)

    with torch.no_grad():
        out_orig = original(x, style)
        out_export = exportable(x, style)

    max_diff = (out_orig - out_export).abs().max().item()
    mean_diff = (out_orig - out_export).abs().mean().item()

    print(f"StyledConv parity - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

    return max_diff < 1e-4


def run_all_parity_tests():
    """Run all parity tests."""
    print("=" * 60)
    print("PARITY TESTS: Original vs Export-Friendly Blocks")
    print("=" * 60)

    results = {}

    print("\n1. Testing EqualLinear...")
    results['EqualLinear'] = test_equal_linear_parity()

    print("\n2. Testing ModulatedConv2d...")
    results['ModulatedConv2d'] = test_modulated_conv_parity()

    print("\n3. Testing StyledConv...")
    results['StyledConv'] = test_styled_conv_parity()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return all_passed


if __name__ == "__main__":
    run_all_parity_tests()
