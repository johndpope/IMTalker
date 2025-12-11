#!/usr/bin/env python
"""
Parity Test Suite for Export-Friendly Attention Modules
========================================================

This script verifies that the export-friendly attention modules produce
numerically equivalent outputs to the original implementations.

Tests:
1. ExportFriendlyMultiheadAttention parity
2. ExportFriendlySwinAttention parity
3. ExportFriendlyTransformerBlock parity
4. ExportFriendlySwinBlock parity
5. ExportFriendlyUpsampler (no direct parity - different architecture)
6. ExportFriendlyCrossAttention parity (standard attention path)
7. ExportFriendlySelfAttention parity
8. ONNX export verification

Usage:
    python test_attention_exportable.py
    python test_attention_exportable.py --verbose
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from argparse import Namespace

# Thresholds for parity tests
MAX_ABS_DIFF = 1e-4      # Maximum absolute difference
MIN_COS_SIM = 0.999      # Minimum cosine similarity (slightly relaxed for attention)


def cosine_similarity(a, b):
    """Compute cosine similarity between two tensors."""
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def test_multihead_attention():
    """Test ExportFriendlyMultiheadAttention parity with StandardUnifiedAttention."""
    from renderer.attention_modules import StandardUnifiedAttention
    from renderer.attention_modules_exportable import ExportFriendlyMultiheadAttention

    print("=" * 60)
    print("TEST: MultiheadAttention Parity")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    num_heads = 8
    seq_len = 64

    original = StandardUnifiedAttention(dim=dim, num_heads=num_heads)
    original.eval()

    exportable = ExportFriendlyMultiheadAttention.from_original(original)
    exportable.eval()

    # Test inputs
    query = torch.randn(2, seq_len, dim)
    key = torch.randn(2, seq_len, dim)
    value = torch.randn(2, seq_len, dim)

    with torch.no_grad():
        out_orig, attn_orig = original(query, key, value)
        out_export, attn_export = exportable(query, key, value)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)

    print(f"Output max diff: {max_diff:.2e}")
    print(f"Output cosine similarity: {cos_sim:.6f}")

    # Attention map comparison (shapes may differ due to head averaging)
    print(f"Original attn shape: {attn_orig.shape}")
    print(f"Exportable attn shape: {attn_export.shape}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_swin_attention():
    """Test ExportFriendlySwinAttention parity with SwinUnifiedAttention."""
    from renderer.attention_modules import SwinUnifiedAttention
    from renderer.attention_modules_exportable import ExportFriendlySwinAttention

    print("\n" + "=" * 60)
    print("TEST: SwinAttention Parity")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    num_heads = 8
    window_size = 8

    original = SwinUnifiedAttention(
        dim=dim, num_heads=num_heads, window_size=window_size
    )
    original.eval()

    exportable = ExportFriendlySwinAttention.from_original(original)
    exportable.eval()

    # Test inputs - windowed format [B*num_windows, ws*ws, C]
    num_windows = 4
    batch_size = 2
    B_ = batch_size * num_windows
    ws_sq = window_size * window_size

    query = torch.randn(B_, ws_sq, dim)
    key = torch.randn(B_, ws_sq, dim)
    value = torch.randn(B_, ws_sq, dim)

    with torch.no_grad():
        out_orig = original(query, key, value)
        out_export = exportable(query, key, value)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)

    print(f"Output max diff: {max_diff:.2e}")
    print(f"Output cosine similarity: {cos_sim:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_transformer_block():
    """Test ExportFriendlyTransformerBlock parity with UnifiedTransformerBlock."""
    from renderer.attention_modules import UnifiedTransformerBlock
    from renderer.attention_modules_exportable import ExportFriendlyTransformerBlock

    print("\n" + "=" * 60)
    print("TEST: TransformerBlock Parity")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    resolution = (16, 16)
    num_heads = 8

    original = UnifiedTransformerBlock(
        dim=dim, input_resolution=resolution, num_heads=num_heads
    )
    original.eval()

    exportable = ExportFriendlyTransformerBlock.from_original(original)
    exportable.eval()

    # Test self-attention mode
    print("\n1. Self-attention mode:")
    x = torch.randn(2, dim, resolution[0], resolution[1])

    with torch.no_grad():
        out_orig = original(x)
        out_export = exportable(x)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)

    print(f"   Output max diff: {max_diff:.2e}")
    print(f"   Output cosine similarity: {cos_sim:.6f}")

    passed_self = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM

    # Test cross-attention mode
    print("\n2. Cross-attention mode:")
    query = torch.randn(2, dim, resolution[0], resolution[1])
    key = torch.randn(2, dim, resolution[0], resolution[1])
    value = torch.randn(2, dim, resolution[0], resolution[1])

    with torch.no_grad():
        out_orig = original(query, key, value)
        out_export = exportable(query, key, value)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)

    print(f"   Output max diff: {max_diff:.2e}")
    print(f"   Output cosine similarity: {cos_sim:.6f}")

    passed_cross = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM

    passed = passed_self and passed_cross
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_swin_block():
    """Test ExportFriendlySwinBlock parity with UnifiedSwinBlock."""
    from renderer.attention_modules import UnifiedSwinBlock
    from renderer.attention_modules_exportable import ExportFriendlySwinBlock

    print("\n" + "=" * 60)
    print("TEST: SwinBlock Parity")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    resolution = (64, 64)
    num_heads = 8
    window_size = 8

    # Test without shift
    print("\n1. Without shift:")
    original = UnifiedSwinBlock(
        dim=dim, input_resolution=resolution, num_heads=num_heads,
        window_size=window_size, shift_size=0
    )
    original.eval()

    exportable = ExportFriendlySwinBlock.from_original(original)
    exportable.eval()

    x = torch.randn(2, dim, resolution[0], resolution[1])

    with torch.no_grad():
        out_orig = original(x)
        out_export = exportable(x)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)

    print(f"   Output max diff: {max_diff:.2e}")
    print(f"   Output cosine similarity: {cos_sim:.6f}")

    passed_no_shift = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM

    # Test with shift
    print("\n2. With shift:")
    original_shift = UnifiedSwinBlock(
        dim=dim, input_resolution=resolution, num_heads=num_heads,
        window_size=window_size, shift_size=window_size // 2
    )
    original_shift.eval()

    exportable_shift = ExportFriendlySwinBlock.from_original(original_shift)
    exportable_shift.eval()

    with torch.no_grad():
        out_orig_shift = original_shift(x)
        out_export_shift = exportable_shift(x)

    max_diff = (out_orig_shift - out_export_shift).abs().max().item()
    cos_sim = cosine_similarity(out_orig_shift, out_export_shift)

    print(f"   Output max diff: {max_diff:.2e}")
    print(f"   Output cosine similarity: {cos_sim:.6f}")

    passed_shift = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM

    passed = passed_no_shift and passed_shift
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_upsampler():
    """Test ExportFriendlyUpsampler functionality (no direct parity)."""
    from renderer.attention_modules import GuidedResampler
    from renderer.attention_modules_exportable import ExportFriendlyUpsampler

    print("\n" + "=" * 60)
    print("TEST: Upsampler Functionality")
    print("=" * 60)

    print("\nNote: ExportFriendlyUpsampler uses a different architecture")
    print("than GuidedResampler, so we test functionality, not parity.")

    torch.manual_seed(42)
    dim = 256
    ratio = 4
    num_heads = 8

    original = GuidedResampler(dim=dim, downsample_ratio=ratio)
    exportable = ExportFriendlyUpsampler.from_original(original, num_heads=num_heads)
    exportable.eval()

    # Test inputs
    H, W = 64, 64
    H_low, W_low = H // ratio, W // ratio
    N_low = H_low * W_low

    v_high = torch.randn(2, dim, H, W)
    attn_map = torch.softmax(torch.randn(2, N_low, N_low), dim=-1)

    # Test that exportable version runs and produces correct shape
    with torch.no_grad():
        out_export = exportable(v_high, attn_map)

    print(f"Input shape: {v_high.shape}")
    print(f"Output shape: {out_export.shape}")
    print(f"Expected shape: {v_high.shape}")

    shape_correct = out_export.shape == v_high.shape
    print(f"\nShape correct: {shape_correct}")

    # Test ONNX exportability
    try:
        import io
        dummy_high = torch.randn(1, dim, H, W)
        dummy_attn = torch.softmax(torch.randn(1, N_low, N_low), dim=-1)

        # Export to bytes (don't save to file)
        f = io.BytesIO()
        torch.onnx.export(
            exportable, (dummy_high, dummy_attn), f,
            input_names=['v_high_feat', 'coarse_attn_map'],
            output_names=['warped_feat'],
            opset_version=18,
            do_constant_folding=True,
        )
        onnx_exportable = True
        print("ONNX export: SUCCESS")
    except Exception as e:
        onnx_exportable = False
        print(f"ONNX export: FAILED - {e}")

    passed = shape_correct and onnx_exportable
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_cross_attention_standard():
    """Test ExportFriendlyCrossAttention parity (standard attention path)."""
    from renderer.attention_modules import CrossAttention
    from renderer.attention_modules_exportable import ExportFriendlyCrossAttention

    print("\n" + "=" * 60)
    print("TEST: CrossAttention Parity (Standard Path)")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    resolution = (16, 16)  # Below swin_res_threshold
    num_heads = 8

    args = Namespace(
        num_heads=num_heads,
        swin_res_threshold=32,
    )

    original = CrossAttention(args=args, dim=dim, resolution=resolution)
    original.eval()

    exportable = ExportFriendlyCrossAttention.from_original(
        original, num_heads=num_heads, swin_res_threshold=32
    )
    exportable.eval()

    # Test inputs
    A = torch.randn(2, dim, resolution[0], resolution[1])
    B = torch.randn(2, dim, resolution[0], resolution[1])
    C = torch.randn(2, dim, resolution[0], resolution[1])

    with torch.no_grad():
        out_orig, attn_orig = original(A, B, C, None)
        out_export, attn_export = exportable(A, B, C, None)

    max_diff = (out_orig - out_export).abs().max().item()
    cos_sim = cosine_similarity(out_orig, out_export)

    print(f"Output max diff: {max_diff:.2e}")
    print(f"Output cosine similarity: {cos_sim:.6f}")

    passed = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_self_attention():
    """Test ExportFriendlySelfAttention parity."""
    from renderer.attention_modules import SelfAttention
    from renderer.attention_modules_exportable import ExportFriendlySelfAttention

    print("\n" + "=" * 60)
    print("TEST: SelfAttention Parity")
    print("=" * 60)

    torch.manual_seed(42)
    dim = 256
    num_heads = 8
    window_size = 8

    # Test low resolution (standard transformer)
    print("\n1. Low resolution (standard transformer):")
    resolution_low = (16, 16)

    args = Namespace(
        num_heads=num_heads,
        window_size=window_size,
        swin_res_threshold=32,
    )

    original_low = SelfAttention(args=args, dim=dim, resolution=resolution_low)
    original_low.eval()

    exportable_low = ExportFriendlySelfAttention.from_original(
        original_low, window_size=window_size, swin_res_threshold=32
    )
    exportable_low.eval()

    x_low = torch.randn(2, dim, resolution_low[0], resolution_low[1])

    with torch.no_grad():
        out_orig_low = original_low(x_low)
        out_export_low = exportable_low(x_low)

    max_diff = (out_orig_low - out_export_low).abs().max().item()
    cos_sim = cosine_similarity(out_orig_low, out_export_low)

    print(f"   Output max diff: {max_diff:.2e}")
    print(f"   Output cosine similarity: {cos_sim:.6f}")

    passed_low = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM

    # Test high resolution (Swin)
    print("\n2. High resolution (Swin blocks):")
    resolution_high = (64, 64)

    original_high = SelfAttention(args=args, dim=dim, resolution=resolution_high)
    original_high.eval()

    exportable_high = ExportFriendlySelfAttention.from_original(
        original_high, window_size=window_size, swin_res_threshold=32
    )
    exportable_high.eval()

    x_high = torch.randn(2, dim, resolution_high[0], resolution_high[1])

    with torch.no_grad():
        out_orig_high = original_high(x_high)
        out_export_high = exportable_high(x_high)

    max_diff = (out_orig_high - out_export_high).abs().max().item()
    cos_sim = cosine_similarity(out_orig_high, out_export_high)

    print(f"   Output max diff: {max_diff:.2e}")
    print(f"   Output cosine similarity: {cos_sim:.6f}")

    passed_high = max_diff < MAX_ABS_DIFF and cos_sim > MIN_COS_SIM

    passed = passed_low and passed_high
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def test_onnx_export():
    """Test ONNX export of attention modules."""
    from renderer.attention_modules_exportable import (
        ExportFriendlyMultiheadAttention,
        ExportFriendlySwinAttention,
        ExportFriendlyTransformerBlock,
        ExportFriendlySwinBlock,
    )

    print("\n" + "=" * 60)
    print("TEST: ONNX Export")
    print("=" * 60)

    results = {}
    output_dir = './checkpoints/exported'
    os.makedirs(output_dir, exist_ok=True)

    # Test MultiheadAttention
    print("\n1. ExportFriendlyMultiheadAttention...")
    try:
        mha = ExportFriendlyMultiheadAttention(dim=256, num_heads=8)
        mha.eval()
        q = torch.randn(1, 64, 256)
        k = torch.randn(1, 64, 256)
        v = torch.randn(1, 64, 256)

        path = os.path.join(output_dir, 'mha_exportable.onnx')
        torch.onnx.export(
            mha, (q, k, v), path,
            input_names=['query', 'key', 'value'],
            output_names=['output', 'attn_weights'],
            opset_version=18,
            do_constant_folding=True,
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['MultiheadAttention'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['MultiheadAttention'] = False

    # Test SwinAttention
    print("\n2. ExportFriendlySwinAttention...")
    try:
        swin_attn = ExportFriendlySwinAttention(dim=256, num_heads=8, window_size=8)
        swin_attn.eval()
        q = torch.randn(4, 64, 256)  # 4 windows, ws*ws=64
        k = torch.randn(4, 64, 256)
        v = torch.randn(4, 64, 256)

        path = os.path.join(output_dir, 'swin_attn_exportable.onnx')
        torch.onnx.export(
            swin_attn, (q, k, v), path,
            input_names=['query', 'key', 'value'],
            output_names=['output'],
            opset_version=18,
            do_constant_folding=True,
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['SwinAttention'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['SwinAttention'] = False

    # Test TransformerBlock
    print("\n3. ExportFriendlyTransformerBlock...")
    try:
        transformer = ExportFriendlyTransformerBlock(
            dim=256, input_resolution=(16, 16), num_heads=8
        )
        transformer.eval()
        x = torch.randn(1, 256, 16, 16)

        path = os.path.join(output_dir, 'transformer_block_exportable.onnx')
        torch.onnx.export(
            transformer, (x,), path,
            input_names=['query'],
            output_names=['output'],
            opset_version=18,
            do_constant_folding=True,
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['TransformerBlock'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['TransformerBlock'] = False

    # Test SwinBlock
    print("\n4. ExportFriendlySwinBlock...")
    try:
        swin_block = ExportFriendlySwinBlock(
            dim=256, input_resolution=(64, 64), num_heads=8, window_size=8
        )
        swin_block.eval()
        x = torch.randn(1, 256, 64, 64)

        path = os.path.join(output_dir, 'swin_block_exportable.onnx')
        torch.onnx.export(
            swin_block, (x,), path,
            input_names=['query'],
            output_names=['output'],
            opset_version=18,
            do_constant_folding=True,
        )
        size = os.path.getsize(path) / 1e6
        print(f"   SUCCESS: {path} ({size:.2f} MB)")
        results['SwinBlock'] = True
    except Exception as e:
        print(f"   FAILED: {e}")
        results['SwinBlock'] = False

    passed = all(results.values())
    print(f"\nResult: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(description='Test export-friendly attention modules')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--onnx', action='store_true', help='Also test ONNX export')
    args = parser.parse_args()

    print("=" * 60)
    print("EXPORT-FRIENDLY ATTENTION MODULES PARITY TEST SUITE")
    print("=" * 60)

    results = {}

    # Run parity tests
    results['MultiheadAttention'] = test_multihead_attention()
    results['SwinAttention'] = test_swin_attention()
    results['TransformerBlock'] = test_transformer_block()
    results['SwinBlock'] = test_swin_block()
    results['Upsampler'] = test_upsampler()
    results['CrossAttention_Standard'] = test_cross_attention_standard()
    results['SelfAttention'] = test_self_attention()

    # ONNX export tests
    if args.onnx:
        results['ONNX_Export'] = test_onnx_export()

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
