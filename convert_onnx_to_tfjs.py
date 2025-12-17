#!/usr/bin/env python3
"""
Convert ONNX model to TensorFlow.js using onnx-tf.

This bypasses the onnx2tf bug by using the official ONNX-TensorFlow backend.
"""

import os
import sys
import argparse
import subprocess

def convert_onnx_to_savedmodel(onnx_path: str, output_dir: str):
    """Convert ONNX to TensorFlow SavedModel using onnx-tf."""
    import onnx
    from onnx_tf.backend import prepare

    print(f"Loading ONNX model from {onnx_path}...")
    onnx_model = onnx.load(onnx_path)

    # Check model validity
    print("Checking ONNX model...")
    onnx.checker.check_model(onnx_model)

    print("Converting to TensorFlow SavedModel...")
    print("(This may take several minutes for large models)")

    # Prepare TF representation - handles NCHW->NHWC automatically
    tf_rep = prepare(onnx_model, device='CPU')

    # Export to SavedModel format
    os.makedirs(output_dir, exist_ok=True)
    tf_rep.export_graph(output_dir)

    print(f"SavedModel exported to {output_dir}")
    return output_dir


def convert_savedmodel_to_tfjs(savedmodel_dir: str, tfjs_dir: str):
    """Convert TensorFlow SavedModel to TensorFlow.js format."""
    print(f"Converting SavedModel to TF.js...")

    os.makedirs(tfjs_dir, exist_ok=True)

    cmd = [
        'tensorflowjs_converter',
        '--input_format=tf_saved_model',
        '--output_format=tfjs_graph_model',
        '--signature_name=serving_default',
        savedmodel_dir,
        tfjs_dir
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        # Try without signature_name
        print("Retrying without signature_name...")
        cmd = [
            'tensorflowjs_converter',
            '--input_format=tf_saved_model',
            '--output_format=tfjs_graph_model',
            savedmodel_dir,
            tfjs_dir
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError("TF.js conversion failed")

    print(f"TF.js model exported to {tfjs_dir}")

    # List output files
    print("\nOutput files:")
    for f in os.listdir(tfjs_dir):
        size = os.path.getsize(os.path.join(tfjs_dir, f))
        print(f"  {f}: {size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorFlow.js')
    parser.add_argument('--onnx', default='exports/full_renderer_test.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--savedmodel', default='exports/tf_saved_model',
                        help='Output directory for TF SavedModel')
    parser.add_argument('--tfjs', default='exports/tfjs_model',
                        help='Output directory for TF.js model')
    parser.add_argument('--skip-savedmodel', action='store_true',
                        help='Skip ONNX to SavedModel conversion (use existing)')
    args = parser.parse_args()

    if not args.skip_savedmodel:
        if not os.path.exists(args.onnx):
            print(f"Error: ONNX model not found at {args.onnx}")
            print("Run the ONNX export first: python export_engines.py")
            sys.exit(1)

        convert_onnx_to_savedmodel(args.onnx, args.savedmodel)
    else:
        print(f"Using existing SavedModel at {args.savedmodel}")

    convert_savedmodel_to_tfjs(args.savedmodel, args.tfjs)

    print("\n" + "="*60)
    print("SUCCESS! TF.js model ready for deployment.")
    print("="*60)
    print(f"\nTo use in browser:")
    print(f"  const model = await tf.loadGraphModel('{args.tfjs}/model.json');")
    print(f"\nTo test locally:")
    print(f"  npx http-server {args.tfjs} -c-1 --cors")


if __name__ == '__main__':
    main()
