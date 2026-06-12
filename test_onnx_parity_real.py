"""
ONNX vs PyTorch parity on REAL face images (sources + driving video frames).

The random-noise parity test produces near-uniform attention maps, which make
GuidedResampler's argmax warp flip selections on FP-level drift. Real images
have peaked attention; this measures the divergence that actually matters.
"""

import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from export_checkpoint import load_model


def load_image(path, size=256):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def video_frame(path, idx, size=256):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (size, size))
    return torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def main():
    import onnxruntime as ort

    model, _ = load_model("./checkpoints/renderer.ckpt", device="cpu")
    sess = ort.InferenceSession("./exports_exact/full_renderer.onnx",
                                providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    pairs = []
    sources = sorted(glob.glob("./assets/source_*.png") + glob.glob("./assets/source_*.jpg"))[:3]
    for i, src in enumerate(sources):
        drv = video_frame(f"./assets/driving_{i + 1}.mp4", 30)
        if drv is not None:
            pairs.append((drv, load_image(src), f"driving_{i + 1}[30] -> {src.split('/')[-1]}"))
    # Self-reenactment: same video, different frames
    f0, f60 = video_frame("./assets/driving_1.mp4", 0), video_frame("./assets/driving_1.mp4", 60)
    if f0 is not None and f60 is not None:
        pairs.append((f60, f0, "driving_1[60] -> driving_1[0] (self)"))

    worst_cos, worst_psnr = 1.0, 99.0
    for driving, reference, label in pairs:
        with torch.no_grad():
            out_pt = model(driving, reference)
        frame_pt = out_pt[0]
        ort_out = sess.run(None, {names[0]: driving.numpy(), names[1]: reference.numpy()})
        frame_ort = torch.from_numpy(ort_out[0])

        md = (frame_pt - frame_ort).abs().max().item()
        cos = F.cosine_similarity(frame_pt.flatten(), frame_ort.flatten(), dim=0).item()
        mse = ((frame_pt - frame_ort) ** 2).mean().item()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        print(f"{label:45s} max={md:.2e} cos={cos:.6f} psnr={psnr:.1f}dB")
        worst_cos, worst_psnr = min(worst_cos, cos), min(worst_psnr, psnr)

        out = (frame_ort[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(f"./exports_exact/parity_{label.split('[')[0]}_onnx.png",
                    cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    print(f"\nWorst: cosine={worst_cos:.6f} psnr={worst_psnr:.1f}dB")
    print(f"Result: {'PASS' if worst_cos > 0.999 and worst_psnr > 35 else 'FAIL'}")


if __name__ == "__main__":
    main()
