#!/usr/bin/env python3
"""
Debug script to trace the exact source of modulo operations in the model.
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/media/2TB/IMTalker')

# Monkey-patch torch operations to trace modulo
_original_remainder = torch.remainder
_original_fmod = torch.fmod
_original_mod = torch.Tensor.__mod__
_original_floor_divide = torch.Tensor.__floordiv__

def traced_remainder(input, other):
    import traceback
    print("\n" + "="*80)
    print("🔴 torch.remainder called!")
    print(f"Input shape: {input.shape if hasattr(input, 'shape') else input}")
    print(f"Other: {other}")
    print("Traceback:")
    traceback.print_stack(limit=20)
    print("="*80 + "\n")
    return _original_remainder(input, other)

def traced_fmod(input, other):
    import traceback
    print("\n" + "="*80)
    print("🔴 torch.fmod called!")
    print(f"Input shape: {input.shape if hasattr(input, 'shape') else input}")
    print(f"Other: {other}")
    print("Traceback:")
    traceback.print_stack(limit=20)
    print("="*80 + "\n")
    return _original_fmod(input, other)

def traced_mod(self, other):
    import traceback
    print("\n" + "="*80)
    print("🔴 Tensor.__mod__ called!")
    print(f"Self shape: {self.shape if hasattr(self, 'shape') else self}")
    print(f"Other: {other}")
    print("Traceback:")
    traceback.print_stack(limit=20)
    print("="*80 + "\n")
    return _original_mod(self, other)

def traced_floordiv(self, other):
    import traceback
    # Filter out non-tensor divisors (static values)
    if isinstance(other, (int, float)):
        return _original_floor_divide(self, other)
    print("\n" + "="*80)
    print("🔴 Tensor.__floordiv__ called with tensor divisor!")
    print(f"Self shape: {self.shape if hasattr(self, 'shape') else self}")
    print(f"Other: {other}")
    print("Traceback:")
    traceback.print_stack(limit=20)
    print("="*80 + "\n")
    return _original_floor_divide(self, other)

# Apply patches
torch.remainder = traced_remainder
torch.fmod = traced_fmod
torch.Tensor.__mod__ = traced_mod
# Uncomment to also trace floor division:
# torch.Tensor.__floordiv__ = traced_floordiv

print("Patches applied, now loading model...")

from renderer.models import IMTRenderer
from renderer.attention_modules_exportable import convert_attention_modules
from renderer.lia_resblocks_exportable import convert_to_exportable

# Load model
from argparse import Namespace

def get_default_args():
    return Namespace(
        standard_attention_layers=2,
        feature_embedding_dim=256,
        attention_heads=8,
        attention_dropout=0.0,
        num_heads=8,
        swin_res_threshold=32,
        window_size=8,
    )

checkpoint_path = '/media/2TB/IMTalker/checkpoints/renderer.ckpt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

args = get_default_args()
renderer = IMTRenderer(args)
state_dict = checkpoint.get('state_dict', checkpoint)
renderer.load_state_dict(state_dict, strict=False)

# Convert to export-friendly
print("\n\n" + "="*80)
print("CONVERTING TO EXPORT-FRIENDLY MODULES...")
print("="*80 + "\n")
convert_to_exportable(renderer, convert_attention=True, args=args)
convert_attention_modules(renderer, args=args)

renderer.eval()

print("\n\n" + "="*80)
print("RUNNING FORWARD PASS...")
print("="*80 + "\n")

# Create dummy inputs matching IMTRenderer.forward(x_current, x_reference)
B = 1
H, W = 256, 256
x_current = torch.randn(B, 3, H, W)   # Driving frame
x_reference = torch.randn(B, 3, H, W) # Reference frame

with torch.no_grad():
    try:
        output_frame, t_c = renderer(x_current, x_reference)
        print(f"\n✅ Forward pass succeeded!")
        print(f"   Output frame shape: {output_frame.shape}")
        print(f"   Motion latent shape: {t_c.shape}")
    except Exception as e:
        import traceback
        print(f"\n❌ Forward pass failed: {e}")
        traceback.print_exc()

print("\n" + "="*80)
print("DONE - Any modulo operations above are the culprits!")
print("="*80)
