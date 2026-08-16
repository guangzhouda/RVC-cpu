# -*- coding: utf-8 -*-
"""自动音高校准：测出目标模型的最佳男/女音高补偿（替代拍脑袋的 +12）

原理：
    1) 女声样本 + pitch 0 -> 输出 f0 中位数 = 该模型的目标自然音高 anchor
    2) 男声样本 + 扫 pitch（默认 8~16）-> 各输出 f0
    3) 输出最接近 anchor 的 pitch = 最佳 male_shift
    4) 写入 calibration.json（随模型目录分发，live_onnx 读取）

用法:
    .venv-onnx-stream/Scripts/python.exe eval/calibrate_pitch.py \
        --onnx assets/weights/furina/onnx/stream150.int8.onnx --meta assets/weights/furina/onnx/stream150.onnx.json \
        --male-sample data/eval/LibriSpeech/dev-clean-2/1272/135031/1272-135031-0000.flac \
        --female-sample data/eval/LibriSpeech/dev-clean-2/3536/23268/3536-23268-0000.flac \
        --out assets/weights/furina/calibration.json
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from apps.live_onnx import RVCOnnxEngine  # noqa: E402

_CENTS = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))


def load_48k(path):
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(-1)
    x = torch.from_numpy(a.astype(np.float32))
    if sr != 48000:
        x = torchaudio.functional.resample(x, sr, 48000)
    return x


def out_f0_median(wav_path, sess):
    a, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
    hidden = sess.run(None, {"audio": a.astype(np.float32)[None]})[0][0]
    nf = a.shape[0] // 160 + 1
    h = hidden[:nf]
    center = np.argmax(h, axis=1) + 4
    sal_pad = np.pad(h, ((0, 0), (4, 4)))
    starts = center - 4
    ends = center + 5
    todo_s = np.stack([sal_pad[i, starts[i]:ends[i]] for i in range(h.shape[0])], axis=0)
    todo_c = np.stack([_CENTS[starts[i]:ends[i]] for i in range(h.shape[0])], axis=0)
    devided = np.sum(todo_s * todo_c, axis=1) / np.sum(todo_s, axis=1)
    maxx = np.max(sal_pad, axis=1)
    devided[maxx <= 0.03] = 0
    f0 = 10 * (2 ** (devided / 1200))
    f0[f0 == 10] = 0
    nz = f0[f0 > 0]
    return float(np.median(nz)) if nz.size else 0.0


def convert(engine, x, out_path):
    block = engine.block_frame
    outs = []
    for i in range(x.shape[0] // block):
        out, _ = engine.process_block(x[i * block:(i + 1) * block].numpy())
        outs.append(out)
    out_all = np.concatenate(outs)
    peak = float(np.abs(out_all).max())
    if peak > 0.99:
        out_all = out_all * (0.95 / peak)
    torchaudio.save(out_path, torch.from_numpy(out_all).unsqueeze(0), 48000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--male-sample", required=True)
    ap.add_argument("--female-sample", required=True)
    ap.add_argument("--range", default="8,16", help="男声 pitch 扫描范围（半音，默认 8~16）")
    ap.add_argument("--out", default="calibration.json")
    ap.add_argument("--dml-device", type=int, default=0)
    ap.add_argument("--tmp-dir", default="logs/offline_test/_calib")
    args = ap.parse_args()

    os.makedirs(args.tmp_dir, exist_ok=True)
    lo, hi = map(int, args.range.split(","))

    # 1) 目标 anchor：女声样本 + pitch 0
    male_x = load_48k(args.male_sample)
    female_x = load_48k(args.female_sample)
    eng = RVCOnnxEngine(dec_onnx=args.onnx, dec_meta=args.meta, index_rate=0,
                        pitch=0, dml_device=args.dml_device)
    f_out = os.path.join(args.tmp_dir, "female_p0.wav")
    convert(eng, female_x, f_out)
    sess = ort.InferenceSession(os.path.join(ROOT, "assets/rmvpe/rmvpe.onnx"), providers=["CPUExecutionProvider"])
    anchor = out_f0_median(f_out, sess)
    print(f"女声样本 +0 -> 输出 f0 = {anchor:.0f}Hz（目标自然音高）")

    # 1.5) 目标 f0 分布：女声样本输出的完整 f0 序列 -> 48 分位数（cdf 模式用）
    def _f0_seq(path):
        a2, sr2 = sf.read(path, dtype="float32")
        if sr2 != 16000:
            a2 = torchaudio.functional.resample(torch.from_numpy(a2), sr2, 16000).numpy()
        h2 = sess.run(None, {"audio": a2.astype(np.float32)[None]})[0][0]
        n2 = a2.shape[0] // 160 + 1
        h2 = h2[:n2]
        c2 = np.argmax(h2, axis=1) + 4
        sp2 = np.pad(h2, ((0, 0), (4, 4)))
        st2 = c2 - 4
        en2 = c2 + 5
        ts2 = np.stack([sp2[i, st2[i]:en2[i]] for i in range(h2.shape[0])], axis=0)
        tc2 = np.stack([_CENTS[st2[i]:en2[i]] for i in range(h2.shape[0])], axis=0)
        dv2 = np.sum(ts2 * tc2, axis=1) / np.sum(ts2, axis=1)
        mx2 = np.max(sp2, axis=1)
        dv2[mx2 <= 0.03] = 0
        f02 = 10 * (2 ** (dv2 / 1200))
        f02[f02 == 10] = 0
        return f02

    f0seq = _f0_seq(f_out)
    nz_seq = f0seq[(f0seq >= 50) & (f0seq <= 1100)]
    if nz_seq.size >= 200:
        tgt_q = np.quantile(np.log2(nz_seq), np.linspace(0.01, 0.99, 48)).astype(np.float64)
        tgt_q_path = os.path.splitext(args.out)[0] + ".quantiles.npy"
        np.save(tgt_q_path, tgt_q)
        print(f"目标分位数（cdf 模式）-> {tgt_q_path} ({nz_seq.size} 有声帧)")
    else:
        print(f"[warn] 有声帧不足 {nz_seq.size}/200，跳过 cdf 分位数输出")

    # 2) 男声样本扫 pitch
    best = None
    for p in range(lo, hi + 1):
        eng2 = RVCOnnxEngine(dec_onnx=args.onnx, dec_meta=args.meta, index_rate=0,
                             pitch=p, dml_device=args.dml_device)
        m_out = os.path.join(args.tmp_dir, f"male_p{p}.wav")
        convert(eng2, male_x, m_out)
        f0 = out_f0_median(m_out, sess)
        diff = abs(f0 - anchor)
        print(f"  pitch {p:+d} -> 输出 f0 = {f0:.0f}Hz (差 {diff:.0f}Hz)")
        if best is None or diff < best[1]:
            best = (p, diff)

    male_shift = best[0]
    print(f"最佳 male_shift = {male_shift:+d}（输出 f0 最接近 {anchor:.0f}Hz）")

    calib = {"target_f0": round(anchor), "male_shift": male_shift,
             "female_shift": 0, "threshold": 165.0,
             "note": "由 eval/calibrate_pitch.py 自动校准，随模型分发"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print("calibration ->", args.out)


if __name__ == "__main__":
    main()
