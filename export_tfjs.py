#!/usr/bin/env python3
"""
Export IMTalker renderer directly to TensorFlow SavedModel format.
Then convert to TensorFlow.js using tensorflowjs_converter.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, '/media/2TB/IMTalker')

from argparse import Namespace
from renderer.models import IMTRenderer
from renderer.lia_resblocks_exportable import convert_to_exportable

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

print("Loading PyTorch model...")
checkpoint_path = '/media/2TB/IMTalker/checkpoints/renderer.ckpt'
ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

args = get_default_args()
model = IMTRenderer(args)
model.load_state_dict(ckpt.get('state_dict', ckpt), strict=False)
model = convert_to_exportable(model, convert_attention=True, args=args)
model.eval()

# Wrapper class
class FullRenderer(nn.Module):
    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def forward(self, driving_image, reference_image):
        return self.renderer(driving_image, reference_image)

wrapper = FullRenderer(model)
wrapper.eval()

print("Converting to TensorFlow using nobuco...")
import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer
import tensorflow as tf

# Create sample inputs
driving = torch.randn(1, 3, 256, 256)
reference = torch.randn(1, 3, 256, 256)

try:
    # Convert PyTorch model to Keras
    keras_model = nobuco.pytorch_to_keras(
        wrapper,
        args=[driving, reference],
        inputs_channel_order=ChannelOrder.PYTORCH,
        outputs_channel_order=ChannelOrder.PYTORCH,
    )

    # Save as TensorFlow SavedModel
    output_dir = '/media/2TB/IMTalker/exports/tfjs/tf_saved_model'
    os.makedirs(output_dir, exist_ok=True)
    keras_model.save(output_dir)
    print(f"✅ TensorFlow SavedModel saved to: {output_dir}")

    # Now convert to TensorFlow.js
    print("\nConverting to TensorFlow.js...")
    tfjs_dir = '/media/2TB/IMTalker/exports/tfjs/web_model'
    os.makedirs(tfjs_dir, exist_ok=True)

    import subprocess
    result = subprocess.run([
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        '--saved_model_tags=serve',
        output_dir,
        tfjs_dir
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ TensorFlow.js model saved to: {tfjs_dir}")
        # List generated files
        for f in os.listdir(tfjs_dir):
            size = os.path.getsize(os.path.join(tfjs_dir, f))
            print(f"   {f}: {size/1e6:.2f} MB")
    else:
        print(f"❌ TensorFlow.js conversion failed:")
        print(result.stderr)

except Exception as e:
    import traceback
    print(f"❌ Conversion failed: {e}")
    traceback.print_exc()
