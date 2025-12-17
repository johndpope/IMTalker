#!/usr/bin/env python3
"""
ONNX Runtime inference for IMTalker renderer.
This can run on any system with onnxruntime installed - no TF.js needed.

Usage:
    python inference_onnx.py --driving driving.jpg --reference reference.jpg --output output.png

Requirements:
    pip install onnxruntime-gpu  # or onnxruntime for CPU
    pip install opencv-python pillow numpy
"""

import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image
import time


def preprocess_image(image_path: str, size: int = 256) -> np.ndarray:
    """Load and preprocess image to NCHW float32 tensor."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)

    # Convert to numpy and normalize to [-1, 1]
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0

    # HWC -> NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]

    return arr


def postprocess_output(output: np.ndarray) -> Image.Image:
    """Convert model output tensor to PIL Image."""
    # NCHW -> HWC
    if output.ndim == 4:
        output = output[0]  # Remove batch dim

    output = output.transpose(1, 2, 0)  # CHW -> HWC

    # Denormalize from [-1, 1] to [0, 255]
    output = (output + 1.0) * 127.5
    output = np.clip(output, 0, 255).astype(np.uint8)

    return Image.fromarray(output)


def main():
    parser = argparse.ArgumentParser(description='IMTalker ONNX Inference')
    parser.add_argument('--model', type=str, default='exports/full_renderer_test.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--driving', type=str, required=True,
                        help='Path to driving image')
    parser.add_argument('--reference', type=str, required=True,
                        help='Path to reference image')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Path to output image')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda',
                        help='Device to run inference on')
    args = parser.parse_args()

    # Set up ONNX Runtime session
    print(f"Loading model: {args.model}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device == 'cuda' else ['CPUExecutionProvider']

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(args.model, sess_options, providers=providers)

    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]

    print(f"Inputs: {input_names}")
    print(f"Outputs: {output_names}")
    print(f"Using provider: {session.get_providers()[0]}")

    # Preprocess images
    print(f"Loading images...")
    driving = preprocess_image(args.driving)
    reference = preprocess_image(args.reference)

    print(f"Driving shape: {driving.shape}")
    print(f"Reference shape: {reference.shape}")

    # Run inference
    print("Running inference...")
    start = time.perf_counter()

    outputs = session.run(
        output_names,
        {
            'driving_image': driving,
            'reference_image': reference,
        }
    )

    elapsed = time.perf_counter() - start
    print(f"Inference time: {elapsed*1000:.2f} ms")

    # Process output
    output_img = outputs[0]  # First output is the rendered image
    print(f"Output shape: {output_img.shape}")

    result = postprocess_output(output_img)
    result.save(args.output)
    print(f"Saved result to: {args.output}")

    # Show latent info if available
    if len(outputs) > 1:
        latent = outputs[1]
        print(f"Motion latent shape: {latent.shape}")


if __name__ == '__main__':
    main()
