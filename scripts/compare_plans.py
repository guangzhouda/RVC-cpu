# -*- coding: utf-8 -*-
"""方案对比（测试录制用）：E 方案 vs F 方案 输出差异 + 相对原声的接近度

用法:
    .venv-onnx-stream\Scripts\python.exe scripts\compare_plans.py ^
        --e-out logs\record\E\output.wav --f-out logs\record\F\output.wav
        [--e-in logs\record\E\input.wav] [--f-in logs\record\F\input.wav]
        [--ref 原声.wav]

说明:
    先跑两个方案各录一段（live_onnx.py --record-out ... --record-in ...），
    然后本脚本输出:
      1) E输出 vs F输出      —— 两方案客观差距（内容相同，直接可比）
      2) 原声 vs E输出/F输出 —— 哪个更接近原声（若给了 --ref）
    指标: PESQ-WB(高好) STOI(高好) SI-SNR(dB,高好) LSD(dB,低好) melCos(高好) f0Corr(高好)
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from evaluate_quality import (  # noqa: E402
    load_48k, align, pesq_mos, si_snr, lsd_db, mel_cos, f0_metrics,
)
from pystoi import stoi as _stoi  # noqa: E402


def report(ref_path, deg_path, label, f0_shift=0.0):
    ref, _ = load_48k(ref_path)
    deg, _ = load_48k(deg_path)
    ref, deg = align(ref, deg)
    # RMS 归一化：deg 缩放到 ref 电平（避免音量差异污染 PESQ/STOI/LSD/melCos；
    # 指标反映音色/内容差异而非响度差异）
    rms_r = float(torch.sqrt((ref ** 2).mean())) + 1e-8
    rms_d = float(torch.sqrt((deg ** 2).mean())) + 1e-8
    deg = deg * (rms_r / rms_d)
    tag_rms = " refRMS=%.0fdB degRMS=%.0fdB" % (20 * np.log10(rms_r), 20 * np.log10(rms_d))
    n = ref.shape[0]
    if n < 16000:
        print(f"[skip] {label}: 音频过短 ({n/48000:.1f}s < 0.33s)")
        return None
    ref16 = torchaudio.functional.resample(ref, 48000, 16000)
    deg16 = torchaudio.functional.resample(deg, 48000, 16000)
    m = {}
    m["pesq"] = pesq_mos(ref16, deg16)
    m["stoi"] = float(_stoi(ref16.numpy(), deg16.numpy(), 16000, extended=False))
    m["sisnr"] = si_snr(ref, deg)
    m["lsd"] = lsd_db(ref, deg)
    m["melcos"] = mel_cos(ref, deg)
    corr, rmse = f0_metrics(ref, deg, f0_shift)
    m["f0corr"] = corr
    m["f0rmse"] = rmse
    print(f"[{label}] ({n/48000:.1f}s){tag_rms}  PESQ={m['pesq']:.3f}  STOI={m['stoi']:.3f}  "
          f"SI-SNR={m['sisnr']:.1f}dB  LSD={m['lsd']:.1f}dB  "
          f"melCos={m['melcos']:.3f}  f0Corr={corr:.3f}  f0RMSE={rmse:.0f}Hz")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-out", required=True, help="E 方案录制输出 (output.wav)")
    ap.add_argument("--f-out", required=True, help="F 方案录制输出 (output.wav)")
    ap.add_argument("--e-in", default=None, help="E 方案录制输入 (input.wav)")
    ap.add_argument("--f-in", default=None, help="F 方案录制输入 (input.wav)")
    ap.add_argument("--ref", default=None, help="原声参考（可选，评估接近度）")
    args = ap.parse_args()

    for p in (args.e_out, args.f_out):
        if not os.path.exists(p):
            print(f"[error] 文件不存在: {p}")
            sys.exit(1)

    print("=" * 78)
    print("① E方案 vs F方案（同一内容，两方案输出差距）")
    print("=" * 78)
    m_ef = report(args.e_out, args.f_out, "E vs F")
    if m_ef is None:
        sys.exit(1)

    ref = args.ref or args.e_in or args.f_in
    if ref:
        print()
        print("=" * 78)
        print(f"② 原声 vs 各方案（接近度，ref = {ref}）")
        print("=" * 78)
        m_re = report(ref, args.e_out, "ref vs E")
        m_rf = report(ref, args.f_out, "ref vs F")
        if m_re and m_rf:
            print("-" * 78)
            print("接近度结论（每列看更接近原声的一方，PESQ/STOI/SI-SNR/melCos/f0Corr 取高，LSD/f0RMSE 取低）:")
            for k, name in [("pesq", "PESQ"), ("stoi", "STOI"), ("sisnr", "SI-SNR"),
                            ("lsd", "LSD"), ("melcos", "melCos"), ("f0corr", "f0Corr")]:
                e, f = m_re[k], m_rf[k]
                if np.isnan(e) and np.isnan(f):
                    continue
                better = "E" if e > f else ("F" if f > e else "=")
                print(f"  {name:8s}: E={e:.3f}  F={f:.3f}  -> 更接近原声: {better}")
    else:
        print("\n（未提供 --ref，跳过接近度对比；可加 --ref 原声.wav 或 --e-in/--f-in）")


if __name__ == "__main__":
    main()