#!/usr/bin/env python3
"""Simple test to export the full renderer to ONNX."""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/media/2TB/IMTalker')

from argparse import Namespace
from renderer.models import IMTRenderer
from renderer.lia_resblocks_exportable import convert_to_exportable
from renderer.attention_modules_exportable import convert_attention_modules

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

# Load model
print("Loading model...")
checkpoint_path = '/media/2TB/IMTalker/checkpoints/renderer.ckpt'
checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)

args = get_default_args()
model = IMTRenderer(args)
model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=False)

# Convert
print("Converting to export-friendly...")
model = convert_to_exportable(model, convert_attention=True, args=args)
# Note: convert_attention_modules is already called inside convert_to_exportable when convert_attention=True

model.eval().cuda()

# Create wrapper
class FullRenderer(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def forward(self, driving_image, reference_image):
        return self.renderer(driving_image, reference_image)

full_renderer = FullRenderer(model)
full_renderer.eval().cuda()

# Create inputs
print("Creating dummy inputs...")
dummy_driving = torch.randn(1, 3, 256, 256, device='cuda')
dummy_reference = torch.randn(1, 3, 256, 256, device='cuda')

# Test forward
print("Testing forward pass...")
with torch.no_grad():
    out, latent = full_renderer(dummy_driving, dummy_reference)
    print(f"  Output: {out.shape}, Latent: {latent.shape}")

# Export
print("\nExporting to ONNX...")
output_path = '/media/2TB/IMTalker/exports/full_renderer_test.onnx'

try:
    torch.onnx.export(
        full_renderer,
        (dummy_driving, dummy_reference),
        output_path,
        input_names=['driving_image', 'reference_image'],
        output_names=['output', 'motion_latent'],
        opset_version=17,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"✅ Export succeeded: {output_path}")
except Exception as e:
    import traceback
    print(f"❌ Export failed: {e}")
    traceback.print_exc()
