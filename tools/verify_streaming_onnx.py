"""Numerically verify the streaming RVC ONNX against the torch wrapper.

Runs StreamingRVCExportWrapper and the exported ONNX on identical inputs
(phone, phone_lengths, pitch, pitchf, sid, rnd) and compares the outputs.

The only source of difference is SineGen's internal rand/randn: in torch it is
fresh per call, in ONNX it is baked at trace time. For voiced frames the
unvoiced noise is suppressed (noise_std=0.003), so the two should still agree
closely on the structured (harmonic/formant) component.

Usage:
    python tools/verify_streaming_onnx.py \
        --model assets/weights/hatsune_miku_model.pth \
        --onnx  assets/weights/hatsune_miku_model.streaming.onnx \
        --block-time 0.35 --extra-time 2.5 --crossfade-time 0.05
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.modules.onnx.export_streaming import (  # noqa: E402
    StreamingRVCExportWrapper,
    get_streaming_shape,
    build_dummy_inputs,
)
from infer.lib.jit.get_synthesizer import get_synthesizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--block-time", type=float, default=0.35)
    ap.add_argument("--crossfade-time", type=float, default=0.05)
    ap.add_argument("--extra-time", type=float, default=2.5)
    ap.add_argument("--formant-shift", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    net_g, cpt = get_synthesizer(args.model, torch.device("cpu"))
    sr = int(cpt["config"][-1])
    stream_shape = get_streaming_shape(
        sr, args.block_time, args.crossfade_time, args.extra_time, args.formant_shift
    )
    dummy, vec_channels, inter_channels, rnd_length = build_dummy_inputs(cpt, stream_shape)

    wrapper = StreamingRVCExportWrapper(
        net_g,
        cpt.get("f0", 1),
        stream_shape["skip_head"],
        stream_shape["return_length"],
        stream_shape["return_length2"],
    )
    wrapper.eval()
    with torch.no_grad():
        audio_torch = wrapper(
            dummy["phone"],
            dummy["phone_lengths"],
            dummy["pitch"],
            dummy["pitchf"],
            dummy["sid"],
            dummy["rnd"],
        ).cpu().numpy()
    print("torch wrapper output shape:", audio_torch.shape)

    sess = onnxruntime.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    feed = {
        "phone": dummy["phone"].cpu().numpy(),
        "phone_lengths": dummy["phone_lengths"].cpu().numpy(),
        "pitch": dummy["pitch"].cpu().numpy(),
        "pitchf": dummy["pitchf"].cpu().numpy(),
        "sid": dummy["sid"].cpu().numpy(),
        "rnd": dummy["rnd"].cpu().numpy(),
    }
    input_names = {i.name for i in sess.get_inputs()}
    feed = {k: v for k, v in feed.items() if k in input_names}
    audio_onnx = sess.run(None, feed)[0]
    print("onnx output shape:", audio_onnx.shape)

    if audio_torch.shape != audio_onnx.shape:
        print("FAIL: shape mismatch", audio_torch.shape, audio_onnx.shape)
        sys.exit(1)

    diff = np.abs(audio_torch - audio_onnx)
    peak = max(np.abs(audio_torch).max(), np.abs(audio_onnx).max(), 1e-8)
    print("torch range: [%.4f, %.4f]" % (audio_torch.min(), audio_torch.max()))
    print("onnx  range: [%.4f, %.4f]" % (audio_onnx.min(), audio_onnx.max()))
    print("max abs diff:  %.6f" % diff.max())
    print("mean abs diff: %.6f" % diff.mean())
    print("peak signal:   %.4f" % peak)
    print("rel max diff:  %.4f%%" % (100.0 * diff.max() / peak))
    # Correlation on the structured component
    a = audio_torch.ravel()
    b = audio_onnx.ravel()
    corr = np.corrcoef(a, b)[0, 1]
    print("pearson corr:  %.6f" % corr)

    # Acceptance: shapes match, outputs finite, and correlation high. The abs
    # diff is dominated by SineGen's baked-vs-fresh unvoiced noise.
    ok = (
        np.isfinite(audio_onnx).all()
        and audio_torch.shape == audio_onnx.shape
        and corr > 0.95
    )
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
