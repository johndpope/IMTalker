#!/usr/bin/env python
"""
IMTalker Checkpoint Export Script
=================================

Exports IMTalker renderer checkpoint to ONNX and TorchScript formats
using export-friendly modules for full compatibility.

Usage:
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --output ./exports
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --format onnx
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --format torchscript
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --convert-attention

Outputs:
    - latent_token_encoder.onnx   : Encodes driving image to motion latent
    - latent_token_decoder.onnx   : Decodes motion latent to style codes
    - dense_feature_encoder.onnx  : Encodes reference image to identity features
    - motion_decoder.onnx         : Full decoder (requires attention conversion)
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from argparse import Namespace
from pathlib import Path


def get_default_args():
    """Get default model arguments."""
    return Namespace(
        standard_attention_layers=2,
        feature_embedding_dim=256,
        attention_heads=8,
        attention_dropout=0.0,
        num_heads=8,
        swin_res_threshold=32,
        window_size=8,
    )


def load_model(checkpoint_path: str, device: str = 'cpu'):
    """Load IMTRenderer from checkpoint."""
    from renderer.models import IMTRenderer

    args = get_default_args()
    model = IMTRenderer(args)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    model = model.to(device)

    return model, args


def convert_model(model, args, convert_attention: bool = False):
    """Convert model to export-friendly version."""
    from renderer.lia_resblocks_exportable import convert_to_exportable

    exportable = convert_to_exportable(
        model,
        convert_attention=convert_attention,
        args=args
    )
    exportable.eval()

    return exportable


def export_onnx(
    module: nn.Module,
    dummy_inputs: tuple,
    output_path: str,
    input_names: list,
    output_names: list,
    dynamic_axes: dict = None,
    opset_version: int = 18,
):
    """Export a module to ONNX format."""
    torch.onnx.export(
        module,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Exported: {output_path} ({size_mb:.2f} MB)")
    return True


def export_torchscript(
    module: nn.Module,
    dummy_inputs: tuple,
    output_path: str,
    method: str = 'trace',
):
    """Export a module to TorchScript format."""
    if method == 'trace':
        scripted = torch.jit.trace(module, dummy_inputs)
    else:
        scripted = torch.jit.script(module)

    scripted.save(output_path)
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Exported: {output_path} ({size_mb:.2f} MB)")
    return True


class LatentTokenEncoderWrapper(nn.Module):
    """Wrapper for latent_token_encoder to ensure clean export."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: [B, 3, 256, 256] driving/current frame
        Returns:
            motion_latent: [B, 32] motion latent code
        """
        return self.encoder(image)


class LatentTokenDecoderWrapper(nn.Module):
    """Wrapper for latent_token_decoder to ensure clean export."""
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, motion_latent: torch.Tensor):
        """
        Args:
            motion_latent: [B, 32] motion latent code
        Returns:
            Tuple of style codes at different resolutions
        """
        return self.decoder(motion_latent)


class DenseFeatureEncoderWrapper(nn.Module):
    """Wrapper for dense_feature_encoder to ensure clean export."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 3, 256, 256] reference image
        Returns:
            features: Multi-scale feature maps
            identity: Identity embedding
        """
        return self.encoder(image)


class MotionDecoderWrapper(nn.Module):
    """Wrapper for motion decoder (decode method)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        ref_features: list,
        identity: torch.Tensor,
        motion_latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ref_features: List of reference feature maps
            identity: Identity embedding
            motion_latent: Motion latent code
        Returns:
            output: Generated frame [B, 3, 256, 256]
        """
        # Decode motion latent to style codes
        styles = self.model.latent_token_decoder(motion_latent)

        # Run decoder
        output = self.model.decode(ref_features, identity, styles)

        return output


def export_latent_token_encoder(model, output_dir: str, device: str, format: str = 'onnx'):
    """Export latent_token_encoder."""
    print("\n[1/4] Exporting latent_token_encoder...")

    encoder = LatentTokenEncoderWrapper(model.latent_token_encoder)
    encoder.eval().to(device)

    dummy_image = torch.randn(1, 3, 256, 256, device=device)

    if format in ['onnx', 'all']:
        path = os.path.join(output_dir, 'latent_token_encoder.onnx')
        try:
            export_onnx(
                encoder, (dummy_image,), path,
                input_names=['image'],
                output_names=['motion_latent'],
                dynamic_axes={'image': {0: 'batch'}, 'motion_latent': {0: 'batch'}},
            )
        except Exception as e:
            print(f"  ONNX export failed: {e}")

    if format in ['torchscript', 'all']:
        path = os.path.join(output_dir, 'latent_token_encoder.pt')
        try:
            export_torchscript(encoder, (dummy_image,), path)
        except Exception as e:
            print(f"  TorchScript export failed: {e}")


def export_latent_token_decoder(model, output_dir: str, device: str, format: str = 'onnx'):
    """Export latent_token_decoder."""
    print("\n[2/4] Exporting latent_token_decoder...")

    decoder = LatentTokenDecoderWrapper(model.latent_token_decoder)
    decoder.eval().to(device)

    dummy_latent = torch.randn(1, 32, device=device)

    if format in ['onnx', 'all']:
        path = os.path.join(output_dir, 'latent_token_decoder.onnx')
        try:
            export_onnx(
                decoder, (dummy_latent,), path,
                input_names=['motion_latent'],
                output_names=['style_8', 'style_16', 'style_32', 'style_64'],
                dynamic_axes={'motion_latent': {0: 'batch'}},
            )
        except Exception as e:
            print(f"  ONNX export failed: {e}")

    if format in ['torchscript', 'all']:
        path = os.path.join(output_dir, 'latent_token_decoder.pt')
        try:
            export_torchscript(decoder, (dummy_latent,), path)
        except Exception as e:
            print(f"  TorchScript export failed: {e}")


def export_dense_feature_encoder(model, output_dir: str, device: str, format: str = 'onnx'):
    """Export dense_feature_encoder."""
    print("\n[3/4] Exporting dense_feature_encoder...")

    encoder = DenseFeatureEncoderWrapper(model.dense_feature_encoder)
    encoder.eval().to(device)

    dummy_image = torch.randn(1, 3, 256, 256, device=device)

    if format in ['onnx', 'all']:
        path = os.path.join(output_dir, 'dense_feature_encoder.onnx')
        try:
            export_onnx(
                encoder, (dummy_image,), path,
                input_names=['image'],
                output_names=['features', 'identity'],
                dynamic_axes={'image': {0: 'batch'}},
            )
        except Exception as e:
            print(f"  ONNX export failed: {e}")

    if format in ['torchscript', 'all']:
        path = os.path.join(output_dir, 'dense_feature_encoder.pt')
        try:
            export_torchscript(encoder, (dummy_image,), path)
        except Exception as e:
            print(f"  TorchScript export failed: {e}")


def export_full_renderer(model, output_dir: str, device: str, format: str = 'onnx'):
    """Export the full renderer as a single model."""
    print("\n[4/4] Exporting full renderer...")

    class FullRenderer(nn.Module):
        def __init__(self, renderer):
            super().__init__()
            self.renderer = renderer

        def forward(self, driving_image: torch.Tensor, reference_image: torch.Tensor):
            """
            Args:
                driving_image: [B, 3, 256, 256]
                reference_image: [B, 3, 256, 256]
            Returns:
                output: [B, 3, 256, 256]
                motion_latent: [B, 32]
            """
            return self.renderer(driving_image, reference_image)

    full_renderer = FullRenderer(model)
    full_renderer.eval().to(device)

    dummy_driving = torch.randn(1, 3, 256, 256, device=device)
    dummy_reference = torch.randn(1, 3, 256, 256, device=device)

    if format in ['onnx', 'all']:
        path = os.path.join(output_dir, 'full_renderer.onnx')
        try:
            export_onnx(
                full_renderer, (dummy_driving, dummy_reference), path,
                input_names=['driving_image', 'reference_image'],
                output_names=['output', 'motion_latent'],
                dynamic_axes={
                    'driving_image': {0: 'batch'},
                    'reference_image': {0: 'batch'},
                    'output': {0: 'batch'},
                    'motion_latent': {0: 'batch'},
                },
            )
        except Exception as e:
            print(f"  ONNX export failed: {e}")
            print("  Note: Full renderer export may require --convert-attention flag")

    if format in ['torchscript', 'all']:
        path = os.path.join(output_dir, 'full_renderer.pt')
        try:
            export_torchscript(full_renderer, (dummy_driving, dummy_reference), path)
        except Exception as e:
            print(f"  TorchScript export failed: {e}")
            print("  Note: Full renderer export may require --convert-attention flag")


def verify_exports(output_dir: str):
    """Verify exported models can be loaded."""
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Check ONNX files
    try:
        import onnx
        import onnxruntime as ort

        onnx_files = list(Path(output_dir).glob('*.onnx'))
        for onnx_file in onnx_files:
            try:
                model = onnx.load(str(onnx_file))
                onnx.checker.check_model(model)

                # Try loading with ONNX Runtime
                session = ort.InferenceSession(str(onnx_file))
                print(f"  ✓ {onnx_file.name} - Valid ONNX model")
            except Exception as e:
                print(f"  ✗ {onnx_file.name} - {e}")
    except ImportError:
        print("  Note: Install onnx and onnxruntime for verification")

    # Check TorchScript files
    pt_files = list(Path(output_dir).glob('*.pt'))
    for pt_file in pt_files:
        try:
            model = torch.jit.load(str(pt_file))
            print(f"  ✓ {pt_file.name} - Valid TorchScript model")
        except Exception as e:
            print(f"  ✗ {pt_file.name} - {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Export IMTalker checkpoint to ONNX/TorchScript',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export all components to ONNX
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt

    # Export with attention module conversion (required for full renderer)
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --convert-attention

    # Export to TorchScript format
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --format torchscript

    # Export to both formats
    python export_checkpoint.py --checkpoint ./checkpoints/renderer.ckpt --format all
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        default='./checkpoints/renderer.ckpt',
        help='Path to renderer checkpoint'
    )
    parser.add_argument(
        '--output', '-o',
        default='./exports',
        help='Output directory for exported models'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['onnx', 'torchscript', 'all'],
        default='onnx',
        help='Export format'
    )
    parser.add_argument(
        '--convert-attention',
        action='store_true',
        help='Convert attention modules to export-friendly versions'
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for export'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification of exported models'
    )
    parser.add_argument(
        '--components',
        nargs='+',
        choices=['encoder', 'decoder', 'identity', 'full', 'all'],
        default=['all'],
        help='Which components to export'
    )

    args = parser.parse_args()

    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("IMTalker Checkpoint Export")
    print("=" * 60)
    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Output directory:  {args.output}")
    print(f"Format:            {args.format}")
    print(f"Device:            {args.device}")
    print(f"Convert attention: {args.convert_attention}")
    print(f"Components:        {args.components}")

    # Load model
    print("\nLoading model...")
    model, model_args = load_model(args.checkpoint, args.device)
    print("  Model loaded successfully")

    # Convert to exportable
    print("\nConverting to export-friendly version...")
    model = convert_model(model, model_args, convert_attention=args.convert_attention)
    model = model.to(args.device)
    print("  Conversion complete")

    # Export components
    components = args.components
    if 'all' in components:
        components = ['encoder', 'decoder', 'identity', 'full']

    if 'encoder' in components:
        export_latent_token_encoder(model, args.output, args.device, args.format)

    if 'decoder' in components:
        export_latent_token_decoder(model, args.output, args.device, args.format)

    if 'identity' in components:
        export_dense_feature_encoder(model, args.output, args.device, args.format)

    if 'full' in components:
        export_full_renderer(model, args.output, args.device, args.format)

    # Verify
    if not args.skip_verify:
        verify_exports(args.output)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    # List exported files
    print("\nExported files:")
    for f in sorted(Path(args.output).glob('*')):
        if f.is_file():
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:40s} {size_mb:8.2f} MB")


if __name__ == "__main__":
    main()
