"""
==============================================================================
IMTalker Renderer Models
==============================================================================

This module implements the Identity-Adaptive Motion Transfer Renderer from the
IMTalker paper (arXiv:2511.22167). It builds upon the Implicit Motion Function
(IMF) framework with key enhancements for identity preservation.

ARCHITECTURE OVERVIEW (from IMTalker paper Section 3.2):
--------------------------------------------------------
The renderer G_render consists of three sub-modules:
  1. Identity-Adaptive (IA) Module (Φ) - personalizes motion latents
  2. Implicit Motion Transfer (IMT) Module - aligns motion with identity features
  3. Synthesis Network - renders aligned features to output image

TOKEN/LATENT FLOW:
-----------------
  Source Image (I_S) ──┬── IdentityEncoder ──► f_id (dense features) + global_id (512-dim)
                       │
                       └── MotionEncoder ────► z_motion_S (32-dim motion latent)
                                                    │
  Driving Signal ─────── MotionEncoder ────► z_motion_D (32-dim motion latent)
                                                    │
                              ┌──────────────────────┘
                              ▼
                    IdentityAdaptive(z_motion, global_id)
                              │
                              ▼
                    z_motion' (personalized 32-dim latent)
                              │
                              ▼
                    MotionDecoder(z_motion')
                              │
                              ▼
                    Multi-scale motion maps (m1, m2, m3, m4)
                              │
                              ▼
                    CrossAttention(m_D, m_S, f_id)
                              │
                              ▼
                    Aligned features ──► SynthesisNetwork ──► Output Frame

COMPARISON WITH ORIGINAL IMF (from /media/2TB/IMF/model.py):
-----------------------------------------------------------
Original IMF:
  - LatentTokenEncoder: outputs dm=32 motion latent
  - LatentTokenDecoder: StyleGAN2-based, produces multi-scale motion features
  - ImplicitMotionAlignment: Cross-attention Q(m_c), K(m_r), V(f_r)
  - No identity adaptation - motion and identity can leak

IMTalker Enhancements:
  - IdentityEncoder: Also outputs global identity embedding (dm=512)
  - IdentityAdaptive Module (NEW): Projects motion latent into identity-specific
    space via MLP: Φ(z_motion, f_global) → z_motion'
  - Motion Distance Consistency Loss: Ensures equal adaptation "strength" for
    all identities, preventing identity leakage
  - Coarse-to-fine attention: Standard attention at low-res, guided sparse
    resampling at high-res for efficiency (40 FPS @ 512x512)

INTEGRATION NOTES:
-----------------
To integrate IMF tokens with IMTalker:
  1. IMF's LatentTokenEncoder ≈ IMTalker's MotionEncoder (both output 32-dim)
  2. IMF's motion latent can be passed through IdentityAdaptive before decoding
  3. The key insight: personalization happens BEFORE the motion decoder, not after
  4. The 32-dim latent is the "implicit motion function" - compact but expressive
==============================================================================
"""

import torch
import torch.nn as nn

from renderer.modules import DownConvResBlock, ResBlock, UpConvResBlock, ConvResBlock
from renderer.attention_modules import CrossAttention, SelfAttention
from renderer.lia_resblocks import StyledConv, EqualConv2d, EqualLinear


class IdentityEncoder(nn.Module):
    """
    Dense Feature Encoder (E_id) from IMTalker Section 3.2.

    Extracts two types of representations from the source image:
      1. Multi-scale dense features (f_dense) - spatial feature pyramids for rendering
      2. Global identity embedding (f_global) - compact identity vector for IA module

    Architecture (similar to IMF's DenseFeatureEncoder but with identity branch):
      - Initial 7x7 conv → 32 channels
      - 6 DownConvResBlocks: 32 → 64 → 128 → 256 → 512 → 512 → 512
      - Global average pooling + 4x EqualLinear layers → 512-dim identity vector

    Args:
        in_channels: Input image channels (default: 3 for RGB)
        output_channels: Channel progression through encoder [64, 128, 256, 512, 512, 512]
        initial_channels: First conv output channels (default: 32)
        dm: Dimension of global identity embedding (default: 512)
             ★ This is DIFFERENT from motion latent dim (32) - identity needs more capacity

    Returns:
        features: List of dense feature maps [f_1, f_2, ..., f_6] in REVERSE order
                  (coarsest first for decoder processing)
        global_id: 512-dim identity embedding for IdentityAdaptive module

    IMF Equivalent: DenseFeatureEncoder (but IMF doesn't output global_id)
    """
    def __init__(self, in_channels=3, output_channels=[64, 128, 256, 512, 512, 512], initial_channels=32, dm=512):
        super(IdentityEncoder, self).__init__()

        # Initial convolution: 3 → 32 channels, preserves spatial size
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, initial_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(initial_channels),
            nn.ReLU(inplace=True)
        )

        # Downsampling blocks: each halves spatial resolution
        # 256→128→64→32→16→8→4 (for 256x256 input)
        self.down_blocks = nn.ModuleList()
        current_channels = initial_channels
        for out_channels in output_channels:
            if out_channels == 32:
                continue  # Skip if same as initial (no-op)
            self.down_blocks.append(DownConvResBlock(current_channels, out_channels))
            current_channels = out_channels

        # Identity embedding branch (NEW in IMTalker vs IMF)
        # Processes the final feature map into a global identity vector
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])
        self.final_linear = EqualLinear(output_channels[-1], dm)  # dm=512 for identity
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Forward pass extracts both spatial features and global identity.

        Args:
            x: Input image tensor [B, 3, H, W]

        Returns:
            features: Multi-scale features [f_6, f_5, f_4, f_3, f_2, f_1] (reversed)
            global_id: Identity embedding [B, 512]
        """
        features = []
        x = self.initial_conv(x)
        features.append(x)

        # Build feature pyramid
        for block in self.down_blocks:
            x = block(x)
            features.append(x)

        # Global identity embedding via global average pooling + MLP
        # x is now [B, 512, 4, 4] → flatten → [B, 512]
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))
        x = self.final_linear(x)  # [B, 512] identity embedding

        # Return features in reverse order (coarsest first) + identity
        return features[::-1], x

class MotionEncoder(nn.Module):
    """
    Motion Encoder (E_motion) from IMTalker Section 3.1.

    Encodes a frame into a compact motion latent vector (z_motion).
    This is THE KEY TOKEN in the IMF/IMTalker architecture!

    ★ THE MOTION LATENT (z_motion) IS THE "IMPLICIT MOTION FUNCTION" ★
    - Dimension: 32 (dm=32) - highly compressed representation
    - Contains: head pose, expression, eye gaze, lip shape - ALL motion info
    - Identity-agnostic by design (but can leak identity without IA module)
    - Can be generated from audio via Flow-Matching Motion Generator

    Architecture (identical to IMF's LatentTokenEncoder):
      - Initial 3x3 conv → 64 channels
      - ResBlocks with downsampling: 64 → 128 → 256 → 512 → 512 → 512
      - EqualConv2d for style consistency
      - Global average pooling → [B, 512]
      - 4x EqualLinear layers (512 → 512) with LeakyReLU
      - Final linear: 512 → 32 (the motion latent!)

    Args:
        initial_channels: First conv output channels (default: 64)
        output_channels: Channel progression [64, 128, 256, 512, 512, 512]
        dm: Dimension of motion latent (default: 32)
            ★ This is THE critical hyperparameter - 32 is the "magic number"
            ★ Small enough for efficient generation, large enough for expressiveness

    Returns:
        z_motion: Motion latent tensor [B, 32]
                  This 32-dim vector captures ALL motion information!

    IMF Equivalent: LatentTokenEncoder (identical architecture)

    IMPORTANT FOR INTEGRATION:
    -------------------------
    The output of this encoder is what gets passed to:
      1. IdentityAdaptive module (for personalization)
      2. MotionDecoder (to produce multi-scale motion maps)

    For audio-driven generation, a Flow-Matching generator predicts z_motion
    directly from audio features, bypassing this encoder entirely.
    """
    def __init__(self, initial_channels=64, output_channels=[64, 128, 256, 512, 512, 512], dm=32):
        super(MotionEncoder, self).__init__()

        # Initial convolution: 3 → 64 channels
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(0.2)

        # ResBlocks with downsampling (from LIA/StyleGAN2 codebase)
        # Each ResBlock halves spatial resolution
        self.res_blocks = nn.ModuleList()
        in_channels = initial_channels
        for out_channels in output_channels:
            self.res_blocks.append(ResBlock(in_channels, out_channels))
            in_channels = out_channels

        # EqualConv2d for weight equalization (StyleGAN2 technique)
        self.equalconv = EqualConv2d(output_channels[-1], output_channels[-1], kernel_size=3, stride=1, padding=1)

        # MLP to produce final motion latent
        # 4x EqualLinear: maintains expressiveness while compressing
        self.linear_layers = nn.ModuleList([EqualLinear(output_channels[-1], output_channels[-1]) for _ in range(4)])

        # Final projection: 512 → dm (32)
        # This is where the "implicit motion function" becomes a compact token!
        self.final_linear = EqualLinear(output_channels[-1], dm)

    def forward(self, x):
        """
        Encode frame to motion latent.

        Args:
            x: Input frame [B, 3, H, W]

        Returns:
            z_motion: 32-dim motion latent [B, 32]
                      ★ This is the token that represents ALL motion!
        """
        x = self.activation(self.conv1(x))

        # Encode through ResBlocks
        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.equalconv(x)

        # Global average pooling: [B, 512, H, W] → [B, 512]
        x = x.view(x.size(0), x.size(1), -1).mean(dim=2)

        # MLP processing
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))

        # Final projection to motion latent
        x = self.final_linear(x)  # [B, 32] - the motion token!
        return x

class MotionDecoder(nn.Module):
    """
    Motion Decoder (D_motion / IMF_D) from IMTalker Section 3.2.2.

    Transforms the compact 32-dim motion latent into multi-scale 2D motion maps.
    Uses StyleGAN2's style modulation to inject motion information at each layer.

    ★ THIS IS WHERE THE 32-DIM TOKEN BECOMES SPATIAL MOTION FEATURES ★

    The decoder uses a "learned constant" as starting point (like StyleGAN2),
    then progressively upsamples while injecting motion via style modulation.
    This allows the compact token to control spatial motion at multiple scales.

    Architecture (StyleGAN2-inspired, from LIA codebase):
      - Learned constant: [1, 32, 4, 4] - starting spatial canvas
      - 13 StyledConv layers with progressive upsampling:
        * Layers 0-3:   4x4  → 8x8   (512 channels) → outputs m1 at 8x8
        * Layers 4-6:   8x8  → 16x16 (512 channels) → outputs m2 at 16x16
        * Layers 7-9:  16x16 → 32x32 (256 channels) → outputs m3 at 32x32
        * Layers 10-12: 32x32 → 64x64 (128 channels) → outputs m4 at 64x64

    Style Modulation (from StyleGAN2):
      - Weight demodulation normalizes conv weights based on style
      - Allows motion latent to control "what motion happens where"
      - Key insight: style injection = motion injection in this context

    Args:
        latent_dim: Dimension of input motion latent (default: 32)
                    ★ Must match MotionEncoder output dim!
        const_dim: Channels of learned constant (default: 32)

    Returns:
        (m1, m2, m3, m4): Multi-scale motion maps at resolutions 8x8, 16x16, 32x32, 64x64
                         Each map encodes "how to move" at that spatial scale
                         Used as Q and K in cross-attention with identity features

    IMF Equivalent: LatentTokenDecoder (identical architecture)

    MOTION MAP USAGE IN CROSS-ATTENTION:
    -----------------------------------
    m_c = MotionDecoder(z_motion_current)  # "where we want to go"
    m_r = MotionDecoder(z_motion_reference)  # "where we came from"
    f_r = IdentityEncoder(reference)  # identity features

    aligned = CrossAttention(Q=m_c, K=m_r, V=f_r)
    # Q·K^T computes motion correspondence, then samples from V (identity features)
    """
    def __init__(self, latent_dim=32, const_dim=32):
        super().__init__()

        # Learned constant - the "canvas" we paint motion onto
        # Shape: [1, 32, 4, 4] - starts at 4x4 resolution
        self.const = nn.Parameter(torch.randn(1, const_dim, 4, 4))

        # StyleGAN2-style convolutions with style modulation
        # The motion latent (t) is injected into each layer via style modulation
        self.style_conv_layers = nn.ModuleList([
            # 4x4 → 8x8 block (outputs m1 at layer 3)
            StyledConv(const_dim, 512, 3, latent_dim),           # 0: 4x4, 32→512
            StyledConv(512, 512, 3, latent_dim, upsample=True),  # 1: 4x4→8x8
            StyledConv(512, 512, 3, latent_dim),                 # 2: 8x8
            StyledConv(512, 512, 3, latent_dim),                 # 3: 8x8 → m1 (512ch, 8x8)

            # 8x8 → 16x16 block (outputs m2 at layer 6)
            StyledConv(512, 512, 3, latent_dim, upsample=True),  # 4: 8x8→16x16
            StyledConv(512, 512, 3, latent_dim),                 # 5: 16x16
            StyledConv(512, 512, 3, latent_dim),                 # 6: 16x16 → m2 (512ch, 16x16)

            # 16x16 → 32x32 block (outputs m3 at layer 9)
            StyledConv(512, 256, 3, latent_dim, upsample=True),  # 7: 16x16→32x32, 512→256
            StyledConv(256, 256, 3, latent_dim),                 # 8: 32x32
            StyledConv(256, 256, 3, latent_dim),                 # 9: 32x32 → m3 (256ch, 32x32)

            # 32x32 → 64x64 block (outputs m4 at layer 12)
            StyledConv(256, 128, 3, latent_dim, upsample=True),  # 10: 32x32→64x64, 256→128
            StyledConv(128, 128, 3, latent_dim),                 # 11: 64x64
            StyledConv(128, 128, 3, latent_dim)                  # 12: 64x64 → m4 (128ch, 64x64)
        ])

    def forward(self, t):
        """
        Decode motion latent to multi-scale motion maps.

        Args:
            t: Motion latent tensor [B, 32]
               ★ This is either from MotionEncoder or after IdentityAdaptive!

        Returns:
            (m1, m2, m3, m4): Motion maps at 8x8, 16x16, 32x32, 64x64
                             Channels: 512, 512, 256, 128 respectively
        """
        # Replicate constant for batch
        x = self.const.repeat(t.shape[0], 1, 1, 1)  # [B, 32, 4, 4]

        m1, m2, m3, m4 = None, None, None, None

        for i, layer in enumerate(self.style_conv_layers):
            # Style modulation injects motion latent `t` into convolution
            x = layer(x, t)

            # Capture multi-scale motion maps at specific layers
            if i == 3:
                m1 = x  # 8x8 resolution, 512 channels - coarsest motion
            elif i == 6:
                m2 = x  # 16x16 resolution, 512 channels
            elif i == 9:
                m3 = x  # 32x32 resolution, 256 channels
            elif i == 12:
                m4 = x  # 64x64 resolution, 128 channels - finest motion

        return m1, m2, m3, m4
    
class SynthesisNetwork(nn.Module):
    """
    Synthesis Network from IMTalker Section 3.2.3.

    Fuses aligned features from the IMT module into the final output image.
    Uses a hybrid architecture of Transformer blocks and ResConv blocks.

    The synthesis network receives features that have already been "motion-aligned"
    by the cross-attention mechanism. Its job is to:
      1. Progressively upsample from coarse to fine
      2. Fuse multi-scale aligned features
      3. Apply self-attention for global coherence
      4. Output the final RGB image

    Architecture:
      - UpConvResBlocks for spatial upsampling (2x each)
      - Skip connections concatenating aligned features at each scale
      - ConvResBlocks to process concatenated features
      - SelfAttention (Transformer) blocks for global consistency
      - Final conv + PixelShuffle for 2x upscale to output resolution

    Args:
        args: Configuration arguments (contains attention params)
        feature_dims: Channel dimensions [32, 64, 128, 256, 512, 512]
        spatial_dims: Spatial resolutions [256, 128, 64, 32, 16, 8]

    Returns:
        output: Final RGB image [B, 3, H, W] with Sigmoid activation

    Design Notes:
      - Features are processed coarse-to-fine (reversed order)
      - Self-attention adds global coherence (important for faces)
      - PixelShuffle in final layer is more efficient than transposed conv
    """
    def __init__(self, args, feature_dims, spatial_dims):
        super().__init__()
        self.args = args

        # Reverse dimensions: we process coarse → fine
        feature_dims_rev = feature_dims[::-1]  # [512, 512, 256, 128, 64, 32]
        spatial_dims_rev = spatial_dims[::-1]  # [8, 16, 32, 64, 128, 256]

        # Upsampling blocks: each doubles spatial resolution
        self.upconv_blocks = nn.ModuleList([
            UpConvResBlock(feature_dims_rev[i], feature_dims_rev[i+1])
            for i in range(len(feature_dims_rev) - 1)
        ])

        # ResBlocks to process concatenated features (x2 channels due to skip connection)
        self.resblocks = nn.ModuleList([
            ConvResBlock(feature_dims_rev[i+1] * 2, feature_dims_rev[i+1])
            for i in range(len(feature_dims_rev) - 1)
        ])

        # Transformer blocks for global self-attention at each scale
        # This is crucial for face coherence - ensures eyes/mouth/etc are consistent
        self.transformer_blocks = nn.ModuleList()
        for i in range(len(spatial_dims_rev) - 1):
            s_dim = spatial_dims_rev[i+1]
            f_dim = feature_dims_rev[i+1]
            self.transformer_blocks.append(
                SelfAttention(args=args, dim=f_dim, resolution=(s_dim, s_dim))
            )

        # Final output layer: conv + PixelShuffle for 2x upscale + Sigmoid
        self.final_conv = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_dims_rev[-1], 3 * 4, kernel_size=3, padding=1),  # 32 → 12 for PixelShuffle
            nn.PixelShuffle(upscale_factor=2),  # 12 → 3 channels, 2x spatial
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, features_align):
        """
        Synthesize output image from aligned features.

        Args:
            features_align: List of aligned feature maps [f_coarse, ..., f_fine]
                           These come from CrossAttention(motion_D, motion_S, identity)

        Returns:
            output: RGB image tensor [B, 3, H, W]
        """
        x = features_align[0]  # Start with coarsest aligned features

        for i in range(len(self.upconv_blocks)):
            x = self.upconv_blocks[i](x)  # Upsample 2x
            x = torch.cat([x, features_align[i + 1]], dim=1)  # Skip connection
            x = self.resblocks[i](x)  # Process concatenated features
            x = self.transformer_blocks[i](x)  # Global self-attention

        return self.final_conv(x)

class IdentidyAdaptive(nn.Module):
    """
    Identity-Adaptive (IA) Module (Φ) from IMTalker Section 3.2.1.

    ★★★ THIS IS THE KEY INNOVATION IN IMTALKER VS ORIGINAL IMF ★★★

    The IA module solves the "identity leakage" problem that plagued IMF:
    - In IMF, motion latents can accidentally encode identity information
    - During cross-identity transfer, this causes the output to drift toward
      the driving video's identity instead of preserving the source

    SOLUTION: Project the motion latent into a PERSONALIZED space for each identity!

    How it works:
      1. Take the generic motion latent z_motion (32-dim)
      2. Concatenate with identity embedding f_global (512-dim)
      3. MLP processes this to output z_motion' (32-dim, personalized)
      4. z_motion' is now "customized" for this specific identity

    Intuition:
      - Different people express the same emotion differently
      - "Happy" for person A might involve more eye crinkle
      - "Happy" for person B might involve more mouth movement
      - The IA module learns these identity-specific "dialects" of motion

    Training Objective (Motion Distance Consistency Loss):
      For the same motion z_motion applied to identities A and B:
        L_dist = |d(z_motion, z_A') - d(z_motion, z_B')|
      This ensures equal adaptation "strength" for all identities,
      preventing the module from adapting some identities more than others.

    Args:
        dim_mot: Motion latent dimension (default: 32, must match MotionEncoder)
        dim_app: Identity embedding dimension (default: 512, from IdentityEncoder)
        depth: Number of MLP layers (default: 4)

    Input:
        mot: Motion latent [B, 32] - generic, identity-agnostic motion
        app: Identity embedding [B, 512] - from IdentityEncoder global branch

    Output:
        z_motion': Personalized motion latent [B, 32]
                   Same dimension as input, but now identity-aware!

    IMF Equivalent: NONE - this is NEW in IMTalker!

    INTEGRATION WITH IMF:
    --------------------
    To add identity adaptation to IMF codebase:
        # Original IMF flow:
        t_c = latent_token_encoder(current_frame)  # [B, 32]
        m_c = latent_token_decoder(t_c)  # multi-scale motion maps

        # IMTalker flow with IA module:
        t_c = latent_token_encoder(current_frame)  # [B, 32]
        f_r, global_id = identity_encoder(reference)  # global_id: [B, 512]
        t_c_adapted = identity_adaptive(t_c, global_id)  # [B, 32] personalized!
        m_c = latent_token_decoder(t_c_adapted)  # motion maps now identity-aware
    """
    def __init__(self, dim_mot=32, dim_app=512, depth=4):
        super().__init__()

        # Input layer: concatenated motion + identity → identity dimension
        # (32 + 512) → 512
        self.in_layer = EqualLinear(dim_app + dim_mot, dim_app)

        # MLP backbone: 4 layers of 512 → 512
        self.linear_layers = nn.ModuleList([
            EqualLinear(dim_app, dim_app) for _ in range(depth)
        ])

        # Output projection: 512 → 32 (back to motion latent dimension)
        self.final_linear = EqualLinear(dim_app, dim_mot)

        self.activation = nn.LeakyReLU(0.2)
        self.scale_activation = nn.Sigmoid()  # Not currently used, but available

    def forward(self, mot, app):
        """
        Adapt motion latent to specific identity.

        Args:
            mot: Generic motion latent [B, 32] from MotionEncoder
            app: Identity embedding [B, 512] from IdentityEncoder

        Returns:
            z_motion': Personalized motion latent [B, 32]

        The output has the same dimension as input motion, but is now
        "personalized" to express motion in an identity-specific way.
        """
        # Concatenate motion and identity: [B, 32] + [B, 512] → [B, 544]
        x = torch.cat((mot, app), dim=-1)

        # Project to identity space
        x = self.in_layer(x)  # [B, 544] → [B, 512]

        # MLP processing
        for linear_layer in self.linear_layers:
            x = self.activation(linear_layer(x))

        # Project back to motion latent space
        out = self.final_linear(x)  # [B, 512] → [B, 32]

        return out

class IMTRenderer(nn.Module):
    """
    Identity-Adaptive Motion Transfer Renderer (G_render) from IMTalker.

    This is the main renderer class that combines all components to perform
    motion transfer while preserving identity. It implements the complete
    pipeline from the IMTalker paper Section 3.2.

    ★★★ COMPLETE TOKEN FLOW (DETAILED) ★★★

    ENCODING PHASE:
    ──────────────
    Reference Frame (x_reference):
      └── IdentityEncoder (dense_feature_encoder)
          ├── f_r: Multi-scale dense features [f_1..f_6] for rendering
          └── i_r: Global identity embedding [B, 512] for IA module

      └── MotionEncoder (latent_token_encoder)
          └── t_r: Reference motion latent [B, 32]

    Current/Driving Frame (x_current):
      └── MotionEncoder (latent_token_encoder)
          └── t_c: Current motion latent [B, 32]
              (For audio-driven: t_c comes from Flow-Matching generator)

    IDENTITY ADAPTATION PHASE:
    ─────────────────────────
    t_r + i_r ──► IdentityAdaptive ──► ta_r: Adapted reference motion [B, 32]
    t_c + i_r ──► IdentityAdaptive ──► ta_c: Adapted current motion [B, 32]

    ★ Key insight: Both motions are adapted to the SAME identity (i_r)
      This ensures the output preserves the reference identity!

    MOTION DECODING PHASE:
    ─────────────────────
    ta_r ──► MotionDecoder ──► ma_r: Reference motion maps (m1, m2, m3, m4)
    ta_c ──► MotionDecoder ──► ma_c: Current motion maps (m1, m2, m3, m4)

    IMPLICIT MOTION TRANSFER (IMT) PHASE:
    ────────────────────────────────────
    At each spatial scale i:
      CrossAttention(Q=ma_c[i], K=ma_r[i], V=f_r[i])

    Where:
      Q (Query) = Current motion map "where we want features to appear"
      K (Key)   = Reference motion map "where features currently are"
      V (Value) = Reference identity features "what the identity looks like"

    The attention computes: softmax(Q·K^T / √d) · V
      - Q·K^T finds motion correspondence (current↔reference positions)
      - softmax normalizes to attention weights
      - Multiplying by V samples identity features at corresponding positions

    This is IMPLICIT motion transfer because we never compute explicit optical flow!
    The cross-attention implicitly learns the motion correspondence.

    SYNTHESIS PHASE:
    ───────────────
    aligned_features ──► SynthesisNetwork ──► output_frame [B, 3, H, W]

    ARCHITECTURE DIMENSIONS:
    ──────────────────────
    feature_dims = [32, 64, 128, 256, 512, 512]  (channel dimensions)
    spatial_dims = [256, 128, 64, 32, 16, 8]      (spatial resolutions)
    motion_latent_dim = 32                        (the compact token!)
    identity_dim = 512                            (global identity embedding)

    IMF EQUIVALENT MAPPING:
    ─────────────────────
    IMTalker                    │ IMF (original)
    ───────────────────────────────────────────────────
    IdentityEncoder             │ DenseFeatureEncoder (no identity branch)
    MotionEncoder               │ LatentTokenEncoder
    MotionDecoder               │ LatentTokenDecoder
    IdentityAdaptive            │ ✗ (NOT IN IMF - key innovation!)
    CrossAttention (imt)        │ ImplicitMotionAlignment
    SynthesisNetwork            │ FrameDecoder

    COARSE-TO-FINE ATTENTION:
    ────────────────────────
    For efficiency (40 FPS!), IMTalker uses hierarchical attention:
    - Coarse levels (8x8, 16x16): Full standard attention O(N²)
    - Fine levels (32x32, 64x64): Guided sparse resampling using coarse attention
      This avoids computing full attention at high resolution!
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Multi-scale dimensions for feature pyramid
        self.feature_dims = [32, 64, 128, 256, 512, 512]   # Channels at each scale
        self.motion_dims = self.feature_dims               # Motion map channels (same)
        self.spatial_dims = [256, 128, 64, 32, 16, 8]      # Spatial resolutions

        # ============ ENCODERS ============
        # Identity Encoder: extracts dense features + global identity embedding
        # Input: [B, 3, 256, 256] → Output: (features, identity [B, 512])
        self.dense_feature_encoder = IdentityEncoder(output_channels=self.feature_dims)

        # Motion Encoder: extracts compact motion latent
        # Input: [B, 3, 256, 256] → Output: motion latent [B, 32]
        self.latent_token_encoder = MotionEncoder(
            initial_channels=64,
            output_channels=[128, 256, 512, 512, 512],
            # Note: dm=32 by default - THE motion token dimension
        )

        # ============ DECODER ============
        # Motion Decoder: expands motion latent to multi-scale motion maps
        # Input: [B, 32] → Output: (m1, m2, m3, m4) at 8x8, 16x16, 32x32, 64x64
        self.latent_token_decoder = MotionDecoder()

        # Frame Decoder / Synthesis Network: renders aligned features to image
        self.frame_decoder = SynthesisNetwork(args, self.feature_dims, self.spatial_dims)

        # ============ IDENTITY ADAPTIVE MODULE ============
        # ★ THE KEY INNOVATION! Personalizes motion for each identity
        # Input: (motion [B, 32], identity [B, 512]) → Output: adapted motion [B, 32]
        self.adapt = IdentidyAdaptive()

        # ============ IMPLICIT MOTION TRANSFER (CROSS-ATTENTION) ============
        # One cross-attention block per spatial scale
        # Coarse scales: standard attention (is_standard_attention=True)
        # Fine scales: guided sparse resampling (is_standard_attention=False)
        self.imt = nn.ModuleList()
        for dim, s_dim in zip(self.feature_dims[::-1], self.spatial_dims[::-1]):
            self.imt.append(CrossAttention(args=args, dim=dim, resolution=(s_dim, s_dim)))

    def decode(self, ma_c, ma_r, f_r):
        """
        Implicit Motion Transfer via hierarchical cross-attention.

        This is the core of IMTalker's motion transfer mechanism!
        Uses cross-attention to implicitly compute motion correspondence
        and sample identity features at corresponding positions.

        Args:
            ma_c: Current motion maps - "where we want features" (Query)
            ma_r: Reference motion maps - "where features are" (Key)
            f_r: Reference identity features - "what features look like" (Value)

        Returns:
            output_frame: Rendered frame [B, 3, H, W]

        Attention Mechanism:
            aligned = softmax(Q·K^T / √d) · V
            Where Q from ma_c, K from ma_r, V from f_r

        Coarse-to-Fine Strategy:
            - Coarse levels: Compute full attention, save attention map
            - Fine levels: Use upsampled coarse attention for guided sampling
              (Avoids O(N²) at high resolution!)
        """
        num_levels = len(self.spatial_dims)
        aligned_features = [None] * num_levels
        attention_map = None  # Reused across fine levels

        for i in range(num_levels):
            attention_block = self.imt[i]

            if attention_block.is_standard_attention:
                # COARSE STAGE: Full cross-attention
                # Computes attention map from motion features
                aligned_feature, attention_map = attention_block.coarse_stage(
                    ma_c[i],  # Query: current motion
                    ma_r[i],  # Key: reference motion
                    f_r[i]    # Value: identity features
                )
                aligned_features[i] = aligned_feature
            else:
                # FINE STAGE: Guided sparse resampling
                # Uses upsampled coarse attention map (no Q·K computation!)
                aligned_feature = attention_block.fine_stage(
                    f_r[i],             # Value: identity features
                    attn=attention_map  # Reuse coarse attention
                )
                aligned_features[i] = aligned_feature

        # Synthesize final frame from aligned features
        output_frame = self.frame_decoder(aligned_features)
        return output_frame

    def app_encode(self, x):
        """
        Encode appearance/identity from reference frame.

        Returns:
            f_r: Multi-scale dense features for rendering
            i_r: Global identity embedding [B, 512] for IA module
        """
        f_r, i_r = self.dense_feature_encoder(x)
        return f_r, i_r

    def mot_encode(self, x):
        """
        Encode motion from a frame.

        Returns:
            mot_latent: 32-dim motion latent [B, 32]
                        ★ This is THE implicit motion token!
        """
        mot_latent = self.latent_token_encoder(x)
        return mot_latent

    def mot_decode(self, x):
        """
        Decode motion latent to multi-scale motion maps.

        Args:
            x: Motion latent [B, 32] (should be after identity adaptation!)

        Returns:
            mot_map: Tuple (m1, m2, m3, m4) of motion maps
        """
        mot_map = self.latent_token_decoder(x)
        return mot_map

    def id_adapt(self, t, i_r):
        """
        Adapt motion latent to specific identity.

        Args:
            t: Generic motion latent [B, 32]
            i_r: Identity embedding [B, 512]

        Returns:
            ta: Personalized motion latent [B, 32]
        """
        return self.adapt(t, i_r)

    def forward(self, x_current, x_reference):
        """
        Full forward pass for motion transfer.

        This implements the complete IMTalker pipeline:
        1. Encode reference: get identity features + motion latent
        2. Encode current: get motion latent
        3. Adapt both motions to reference identity (IA module)
        4. Decode adapted motions to multi-scale motion maps
        5. Cross-attention transfer: align identity features to current motion
        6. Synthesize output frame

        Args:
            x_current: Current/driving frame [B, 3, H, W]
                       (provides motion to transfer)
            x_reference: Reference/source frame [B, 3, H, W]
                        (provides identity to preserve)

        Returns:
            output_frame: Generated frame with reference identity + current motion
            t_c: Raw motion latent from current frame (for training/analysis)

        Token Flow Visualization:
        ────────────────────────
        x_reference ──┬── app_encode ──► f_r (features), i_r (identity)
                      └── mot_encode ──► t_r (motion)

        x_current ────── mot_encode ──► t_c (motion)

        t_r + i_r ──► adapt ──► ta_r (adapted ref motion)
        t_c + i_r ──► adapt ──► ta_c (adapted cur motion)

        ta_r ──► mot_decode ──► ma_r (ref motion maps)
        ta_c ──► mot_decode ──► ma_c (cur motion maps)

        decode(ma_c, ma_r, f_r) ──► output_frame
        """
        # ============ ENCODING ============
        # Reference: extract identity features AND motion
        f_r, i_r = self.app_encode(x_reference)  # f_r: features, i_r: identity [B, 512]
        t_r = self.mot_encode(x_reference)        # t_r: motion latent [B, 32]

        # Current: extract motion only (identity comes from reference!)
        t_c = self.mot_encode(x_current)          # t_c: motion latent [B, 32]

        # ============ IDENTITY ADAPTATION ============
        # Adapt BOTH motions to the REFERENCE identity
        # This is the key to preserving identity during transfer!
        ta_r = self.adapt(t_r, i_r)  # Reference motion adapted to its own identity
        ta_c = self.adapt(t_c, i_r)  # Current motion adapted to REFERENCE identity

        # ============ MOTION DECODING ============
        # Expand adapted motion latents to multi-scale motion maps
        ma_r = self.mot_decode(ta_r)  # Reference motion maps (m1, m2, m3, m4)
        ma_c = self.mot_decode(ta_c)  # Current motion maps (m1, m2, m3, m4)

        # ============ IMPLICIT MOTION TRANSFER + SYNTHESIS ============
        # Cross-attention: Q=ma_c, K=ma_r, V=f_r
        # Samples identity features (f_r) at positions corresponding to current motion
        output_frame = self.decode(ma_c, ma_r, f_r)

        # Return frame and raw motion token (useful for training/analysis)
        return output_frame, t_c