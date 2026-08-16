# -*- coding: utf-8 -*-
"""离线变声诊断：任意 wav/flac -> RVC ONNX 管线 -> 输出 wav（与实时管线同一引擎）。
用途：区分"模型/参数问题"与"声卡路由问题"。
用法:
    .venv-onnx-stream\Scripts\python.exe convert\offline_convert.py 输入.wav 输出.wav --pitch 12 --dml-device 0
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from apps.live_onnx import RVCOnnxEngine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--pitch", type=int, default=0, help="半音变调，男变女建议 12")
    ap.add_argument("--auto-pitch", action="store_true", help="自动音高补偿（按输入 f0 映射到目标音高域）")
    ap.add_argument("--auto-pitch-mode", default="binary", choices=["binary", "continuous", "cdf"],
                    help="cdf=分布级分位数映射（需 --target-quantiles）；binary=男+12/女+0；continuous=连续映射")
    ap.add_argument("--target-quantiles", default=None, help="目标分位数 .npy（cdf 模式，calibrate_pitch.py 生成）")
    ap.add_argument("--auto-pitch-target", type=float, default=220.0, help="目标基准 f0（Hz）")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--rms-mix", type=float, default=None, help="音量包络混合 0~1（gui_v1 同款）")
    ap.add_argument("--dml-device", type=int, default=0)
    ap.add_argument("--onnx", default="assets/weights/miku/onnx/stream150.onnx")
    ap.add_argument("--meta", default="assets/weights/miku/onnx/stream150.onnx.json")
    ap.add_argument("--index", default="assets/weights/miku/hatsune_miku_model.index")
    ap.add_argument("--embedded-index", default=None, help="内嵌质心表（与 --index 二选一）")
    ap.add_argument("--f0method", default="rmvpe", choices=["rmvpe", "fcpe"])
    ap.add_argument("--provider", default="dml", choices=["dml", "cpu"])
    ap.add_argument("--cpu-encoder", action="store_true", help="hubert/rmvpe 走 CPU，解码器走 DML")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    os.chdir(ROOT)
    audio, sr = torchaudio.load(args.input)
    audio = audio.mean(0)
    if sr != 48000:
        audio = torchaudio.functional.resample(audio, sr, 48000)
    print("input:", args.input, "dur=%.1fs" % (audio.shape[0] / 48000))

    engine = RVCOnnxEngine(
        dec_onnx=args.onnx,
        dec_meta=args.meta,
        index_path=args.index,
        embedded_index=args.embedded_index,
        index_rate=args.index_rate,
        rms_mix_rate=args.rms_mix,
        pitch=args.pitch,
        auto_pitch=args.auto_pitch,
        auto_pitch_mode=args.auto_pitch_mode,
        target_quantiles=args.target_quantiles,
        auto_pitch_target=args.auto_pitch_target,
        provider=args.provider,
        dml_device=args.dml_device,
        f0method=args.f0method,
        rng_seed=args.seed,
        cpu_encoder=args.cpu_encoder,
    )
    engine.warmup()

    block = engine.block_frame
    outs = []
    for i in range(0, audio.shape[0] - block + 1, block):
        out, _dt = engine.process_block(audio[i:i + block].numpy())
        outs.append(out)
    out_all = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
    peak = float(np.abs(out_all).max()) if out_all.size else 0.0
    if peak > 0.99:
        out_all = out_all * (0.95 / peak)
    torchaudio.save(args.output, torch.from_numpy(out_all).unsqueeze(0), 48000)
    print("output:", args.output, "dur=%.1fs peak=%.2f" % (out_all.shape[0] / 48000, peak))


if __name__ == "__main__":
    main()
