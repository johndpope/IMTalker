"""
ONNX Runtime vs PyTorch parity test for the full IMTRenderer export.

Compares the exported full_renderer.onnx (exact attention path, feature-modulated
convs) against the original PyTorch model loaded from renderer.ckpt.

Usage:
    python test_onnx_parity.py --checkpoint ./checkpoints/renderer.ckpt \
        --onnx ./exports_exact/full_renderer.onnx
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F


def get_default_args():
    from export_checkpoint import get_default_args as _g
    return _g()


def load_original(checkpoint_path):
    from renderer.models import IMTRenderer
    args = get_default_args()
    model = IMTRenderer(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    clean = {k.replace("gen.", ""): v for k, v in state_dict.items() if k.startswith("gen.")}
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="./checkpoints/renderer.ckpt")
    parser.add_argument("--onnx", default="./exports_exact/full_renderer.onnx")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    import onnxruntime as ort

    print("Loading PyTorch model...")
    model = load_original(args.checkpoint)

    print("Loading ONNX session...")
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    print(f"  inputs: {input_names}")

    worst_cos, worst_max = 1.0, 0.0
    for t in range(args.trials):
        torch.manual_seed(100 + t)
        # Sigmoid-squashed noise stands in for natural images in [0, 1]
        driving = torch.sigmoid(torch.randn(1, 3, 256, 256) * 0.5)
        reference = torch.sigmoid(torch.randn(1, 3, 256, 256) * 0.5)

        with torch.no_grad():
            out_pt = model(driving, reference)
        frame_pt = out_pt[0] if isinstance(out_pt, (tuple, list)) else out_pt

        ort_out = sess.run(None, {
            input_names[0]: driving.numpy(),
            input_names[1]: reference.numpy(),
        })
        frame_ort = torch.from_numpy(ort_out[0])

        max_diff = (frame_pt - frame_ort).abs().max().item()
        cos = F.cosine_similarity(frame_pt.flatten(), frame_ort.flatten(), dim=0).item()
        psnr = 10 * np.log10(1.0 / max(((frame_pt - frame_ort) ** 2).mean().item(), 1e-12))
        print(f"trial {t}: frame {tuple(frame_pt.shape)} max_diff={max_diff:.2e} "
              f"cosine={cos:.6f} psnr={psnr:.1f}dB")
        worst_cos = min(worst_cos, cos)
        worst_max = max(worst_max, max_diff)

    passed = worst_cos > 0.9999 and worst_max < 1e-3
    print(f"\nWorst: max_diff={worst_max:.2e} cosine={worst_cos:.6f}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
