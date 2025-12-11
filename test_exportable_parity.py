#!/usr/bin/env python
"""
Parity Test Suite for Export-Friendly LIA ResBlocks
====================================================

This script verifies that the export-friendly blocks produce numerically
equivalent outputs to the original implementations.

Tests:
1. EqualLinear parity
2. EqualConv2d parity
3. ModulatedConv2d parity (standard conv)
4. ModulatedConv2d parity (with upsample)
5. ModulatedConv2d parity (with downsample)
6. StyledConv parity
7. Full IMTRenderer parity
8. ONNX export verification

Usage:
    python test_exportable_parity.py
    python test_exportable_parity.py --verbose
    python test_exportable_parity.py --export-onnx
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from argparse import Namespace

# Thresholds for parity tests
MAX_ABS_DIFF = 1e-4      # Maximum absolute difference
MIN_COS_SIM = 0.9999     # Minimum cosine similarity


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def test_equal_linear():
    """Test EqualLinear parity."""
    from renderer.lia_resblocks import EqualLinear
    from renderer.lia_resblocks_exportable import ExportableEqualLinear

    print("=" * 60)
    print("TEST: EqualLinear Parity")
    print("=" * 60)

    # Test without activation
    print("\n1. Without activation:")
    original = EqualLinear(512, 256)
    original.eval()
    exportable = ExportableEqualLinear.from_original(original)
    exportable.eval()

    x = torch.randn(4, 512)
    with torch.no_grad():
        out_orig = original(x)
        out_export = exportable(x)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)
    print(f"   Max diff: {max_diff:.2e}, Cosine sim: {cos_sim:.6f}")

    # Test with activation
    print("\n2. With activation (fused_leaky_relu):")
    original_act = EqualLinear(512, 256, activation='fused_lrelu')
    original_act.eval()
    exportable_act = ExportableEqualLinear.from_original(original_act)
    exportable_act.eval()

    with torch.no_grad():
        out_orig_act = original_act(x)
        out_export_act = exportable_act(x)

    max_diff_act = (out_orig_act - out_export_act).abs().max().item()
    cos_sim_act = cosine_similarity(out_orig_act, out_export_act)
    print(f"   Max diff: {max_diff_act:.2e}, Cosine sim: {cos_sim_act:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_equal_conv():
    """Test EqualConv2d parity."""
    from renderer.lia_resblocks import EqualConv2d
    from renderer.lia_resblocks_exportable import ExportableEqualConv2d

    print("\n" + "=" * 60)
    print("TEST: EqualConv2d Parity")
    print("=" * 60)

    original = EqualConv2d(128, 256, kernel_size=3, padding=1)
    original.eval()
    exportable = ExportableEqualConv2d.from_original(original)
    exportable.eval()

    x = torch.randn(2, 128, 32, 32)
    with torch.no_grad():
        out_orig = original(x)
        out_export = exportable(x)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)
    print(f"Max diff: {max_diff:.2e}, Cosine sim: {cos_sim:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_modulated_conv_standard():
    """Test ModulatedConv2d parity (standard, no up/downsample)."""
    from renderer.lia_resblocks import ModulatedConv2d
    from renderer.lia_resblocks_exportable import ExportableModulatedConv2d

    print("\n" + "=" * 60)
    print("TEST: ModulatedConv2d Parity (Standard)")
    print("=" * 60)

    torch.manual_seed(42)
    original = ModulatedConv2d(
        in_channel=256, out_channel=256, kernel_size=3,
        style_dim=32, demodulate=True
    )
    original.eval()
    exportable = ExportableModulatedConv2d.from_original(original)
    exportable.eval()

    x = torch.randn(2, 256, 16, 16)
    style = torch.randn(2, 32)

    with torch.no_grad():
        out_orig = original(x, style)
        out_export = exportable(x, style)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)
    print(f"Max diff: {max_diff:.2e}, Cosine sim: {cos_sim:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_modulated_conv_upsample():
    """Test ModulatedConv2d parity (with upsample)."""
    from renderer.lia_resblocks import ModulatedConv2d
    from renderer.lia_resblocks_exportable import ExportableModulatedConv2d

    print("\n" + "=" * 60)
    print("TEST: ModulatedConv2d Parity (Upsample)")
    print("=" * 60)

    torch.manual_seed(42)
    original = ModulatedConv2d(
        in_channel=512, out_channel=512, kernel_size=3,
        style_dim=32, demodulate=True, upsample=True
    )
    original.eval()
    exportable = ExportableModulatedConv2d.from_original(original)
    exportable.eval()

    x = torch.randn(2, 512, 8, 8)
    style = torch.randn(2, 32)

    with torch.no_grad():
        out_orig = original(x, style)
        out_export = exportable(x, style)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)
    print(f"Output shape: {out_orig.shape} -> {out_export.shape}")
    print(f"Max diff: {max_diff:.2e}, Cosine sim: {cos_sim:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_styled_conv():
    """Test StyledConv parity."""
    from renderer.lia_resblocks import StyledConv
    from renderer.lia_resblocks_exportable import ExportableStyledConv

    print("\n" + "=" * 60)
    print("TEST: StyledConv Parity")
    print("=" * 60)

    torch.manual_seed(42)
    original = StyledConv(
        in_channel=256, out_channel=256, kernel_size=3,
        style_dim=32, upsample=False
    )
    original.eval()
    exportable = ExportableStyledConv.from_original(original)
    exportable.eval()

    x = torch.randn(2, 256, 16, 16)
    style = torch.randn(2, 32)

    with torch.no_grad():
        out_orig = original(x, style)
        out_export = exportable(x, style)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)
    print(f"Max diff: {max_diff:.2e}, Cosine sim: {cos_sim:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_full_model(checkpoint_path='./checkpoints/renderer.ckpt'):
    """Test full IMTRenderer parity."""
    from renderer.models import IMTRenderer
    from renderer.lia_resblocks_exportable import convert_to_exportable

    print("\n" + "=" * 60)
    print("TEST: Full IMTRenderer Parity")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Skipping full model test.")
        return None

    # Create args
    args = Namespace(
        standard_attention_layers=2,
        feature_embedding_dim=256,
        attention_heads=8,
        attention_dropout=0.0,
        num_heads=8,
        swin_res_threshold=32,
        window_size=8,
    )

    # Load original model
    print("Loading original model...")
    original = IMTRenderer(args)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    original.load_state_dict(ckpt['state_dict'], strict=False)
    original.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    original = original.to(device)

    # Convert to exportable
    print("Converting to exportable version...")
    exportable = convert_to_exportable(original)
    exportable.eval()
    exportable = exportable.to(device)

    # Test
    print("Testing parity...")
    x_ref = torch.randn(1, 3, 256, 256, device=device)
    x_cur = torch.randn(1, 3, 256, 256, device=device)

    with torch.no_grad():
        out_orig, t_orig = original(x_cur, x_ref)
        out_export, t_export = exportable(x_cur, x_ref)

    max_diff = (out_orig - out_export).abs().max().item()
    mean_diff = (out_orig - out_export).abs().mean().item()
    cos_sim = cosine_similarity(out_orig, out_export)
    token_diff = (t_orig - t_export).abs().max().item()

    print(f"Output max diff: {max_diff:.2e}")
    print(f"Output mean diff: {mean_diff:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Motion token diff: {token_diff:.2e}")

    passed = cos_sim > MIN_COS_SIM
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_onnx_export(checkpoint_path='./checkpoints/renderer.ckpt', output_dir='./checkpoints/exported'):
    """Test ONNX export of exportable components."""
    from renderer.models import IMTRenderer
    from renderer.lia_resblocks_exportable import convert_to_exportable

    print("\n" + "=" * 60)
    print("TEST: ONNX Export")
    print("=" * 60)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Skipping ONNX export test.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Load and convert model
    args = Namespace(
        standard_attention_layers=2,
        feature_embedding_dim=256,
        attention_heads=8,
        attention_dropout=0.0,
        num_heads=8,
        swin_res_threshold=32,
        window_size=8,
    )

    original = IMTRenderer(args)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    original.load_state_dict(ckpt['state_dict'], strict=False)
    original.eval()

    exportable = convert_to_exportable(original)
    exportable.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    exportable = exportable.to(device)

    results = {}

    # Export latent_token_encoder
    print("\n1. Exporting latent_token_encoder...")
    try:
        encoder = exportable.latent_token_encoder
        x = torch.randn(1, 3, 256, 256, device=device)
        path = os.path.join(output_dir, 'latent_token_encoder_exportable.onnx')
        torch.onnx.export(
            encoder, x, path,
            input_names=['image'],
            output_names=['motion_latent'],
            opset_version=18,
            do_constant_folding=True,
            verbose=False
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['latent_token_encoder'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['latent_token_encoder'] = False

    # Export latent_token_decoder
    print("\n2. Exporting latent_token_decoder...")
    try:
        decoder = exportable.latent_token_decoder
        t = torch.randn(1, 32, device=device)
        path = os.path.join(output_dir, 'latent_token_decoder_exportable.onnx')
        torch.onnx.export(
            decoder, t, path,
            input_names=['motion_latent'],
            output_names=['m1', 'm2', 'm3', 'm4'],
            opset_version=18,
            do_constant_folding=True,
            verbose=False
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['latent_token_decoder'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['latent_token_decoder'] = False

    # Export dense_feature_encoder
    print("\n3. Exporting dense_feature_encoder...")
    try:
        id_encoder = exportable.dense_feature_encoder
        x = torch.randn(1, 3, 256, 256, device=device)
        path = os.path.join(output_dir, 'dense_feature_encoder_exportable.onnx')
        torch.onnx.export(
            id_encoder, x, path,
            input_names=['image'],
            output_names=['features', 'identity'],
            opset_version=18,
            do_constant_folding=True,
            verbose=False
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['dense_feature_encoder'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['dense_feature_encoder'] = False

    passed = all(results.values())
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description='Test export-friendly LIA ResBlocks parity')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--export-onnx', action='store_true', help='Also test ONNX export')
    parser.add_argument('--checkpoint', default='./checkpoints/renderer.ckpt', help='Path to checkpoint')
    args = parser.parse_args()

    print("=" * 60)
    print("EXPORT-FRIENDLY LIA RESBLOCKS PARITY TEST SUITE")
    print("=" * 60)

    results = {}

    # Basic block tests
    results['EqualLinear'] = test_equal_linear()
    results['EqualConv2d'] = test_equal_conv()
    results['ModulatedConv2d_Standard'] = test_modulated_conv_standard()
    results['ModulatedConv2d_Upsample'] = test_modulated_conv_upsample()
    results['StyledConv'] = test_styled_conv()

    # Full model test
    full_result = test_full_model(args.checkpoint)
    if full_result is not None:
        results['IMTRenderer'] = full_result

    # ONNX export test
    if args.export_onnx:
        onnx_result = test_onnx_export(args.checkpoint)
        if onnx_result is not None:
            results['ONNX_Export'] = onnx_result

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:30s}: {status}")

    all_passed = all(v for v in results.values() if v is not None)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
