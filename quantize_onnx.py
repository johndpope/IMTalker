#!/usr/bin/env python3
"""
Quantize ONNX model for browser deployment.

Reduces model size from ~512MB to ~128MB using int8 quantization.
"""

import os
import argparse
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path: str, output_path: str):
    """Quantize ONNX model using dynamic int8 quantization."""
    print(f"Input model: {input_path}")
    print(f"Input size: {os.path.getsize(input_path) / 1024 / 1024:.1f} MB")

    # First run shape inference to ensure all types are known
    print("\nRunning shape inference...")
    model = onnx.load(input_path)
    model = onnx.shape_inference.infer_shapes(model)

    # Save preprocessed model
    preprocessed_path = input_path.replace('.onnx', '_preprocessed.onnx')
    onnx.save(model, preprocessed_path)
    print(f"Preprocessed model saved to: {preprocessed_path}")

    print("\nQuantizing with dynamic int8...")
    print("(This may take a few minutes)")

    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        extra_options={'DefaultTensorType': onnx.TensorProto.FLOAT}
    )

    print(f"\nOutput model: {output_path}")
    print(f"Output size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")

    reduction = (1 - os.path.getsize(output_path) / os.path.getsize(input_path)) * 100
    print(f"Size reduction: {reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Quantize ONNX model')
    parser.add_argument('--input', default='exports/full_renderer_test.onnx',
                        help='Input ONNX model path')
    parser.add_argument('--output', default='exports/full_renderer_quantized.onnx',
                        help='Output quantized model path')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input model not found at {args.input}")
        return

    quantize_model(args.input, args.output)

    print("\n" + "="*60)
    print("Quantization complete!")
    print("="*60)
    print("\nTo use quantized model in web demo, update index.html:")
    print("  const modelUrl = '../exports/full_renderer_quantized.onnx';")


if __name__ == '__main__':
    main()
