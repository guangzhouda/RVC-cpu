# -*- coding: utf-8 -*-
"""变声质量客观评估：PESQ-WB / STOI / SI-SNR / LSD / mel余弦 / f0轮廓相关

用法:
    .venv-onnx-stream\Scripts\python.exe eval\evaluate_quality.py --ref 原声.wav --cand 变声.wav
    .venv-onnx-stream\Scripts\python.exe eval\evaluate_quality.py --ref 原声.wav --cand 变声.wav --f0-shift 12
        （--f0-shift N: 比较 f0 轮廓时先把参考音高平移 N 半音，用于评估 pitch≠0 的转换）

指标说明:
    PESQ-WB  1.0~4.5，越高越好（语音感知质量，含延迟/噪声/失真损伤）
    STOI     0~1，越高越好（可懂度）
    SI-SNR   dB，越高越好（尺度不变信噪比）
    LSD      dB，越低越好（对数谱距离）
    melCos   0~1，越高越好（mel 谱余弦相似度）
    f0Corr   -1~1，越高越好（音高轮廓相关性 = 韵律/抑扬顿挫保留度）
    f0RMSE   Hz，越低越好
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from torch_pesq import PesqLoss  # noqa: E402
from pystoi import stoi as _stoi  # noqa: E402


def load_48k(path):
    audio, sr = torchaudio.load(path)
    audio = audio.mean(0)
    if sr != 48000:
        audio = torchaudio.functional.resample(audio, sr, 48000)
    return audio, 48000


def align(ref, deg):
    n = min(ref.shape[0], deg.shape[0])
    return ref[:n], deg[:n]


def pesq_mos(ref16, deg16):
    fn = PesqLoss(1.0, 16000)
    with torch.no_grad():
        loss = fn(ref16.unsqueeze(0), deg16.unsqueeze(0))
    mos = 4.5 - float(loss)
    return max(1.0, min(4.5, mos))


def si_snr(ref, deg):
    ref = ref - ref.mean()
    deg = deg - deg.mean()
    scale = float((deg * ref).sum() / (ref * ref).sum())
    proj = scale * ref
    noise = deg - proj
    return float(10 * torch.log10((proj ** 2).sum() / (noise ** 2).sum() + 1e-12))


def lsd_db(ref, deg, n_fft=1024, hop=256):
    win = torch.hann_window(n_fft)
    Sr = torch.stft(ref, n_fft, hop, window=win, return_complex=True).abs() ** 2
    Sd = torch.stft(deg, n_fft, hop, window=win, return_complex=True).abs() ** 2
    log_r = 10 * torch.log10(Sr + 1e-10)
    log_d = 10 * torch.log10(Sd + 1e-10)
    return float(torch.sqrt(((log_r - log_d) ** 2).mean()))


def mel_cos(ref, deg):
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=48000, n_mels=80, n_fft=1024)
    Mr = torch.log(mel(ref.unsqueeze(0)) + 1e-8)
    Md = torch.log(mel(deg.unsqueeze(0)) + 1e-8)
    Mr = torch.nn.functional.normalize(Mr.squeeze(0), dim=0)
    Md = torch.nn.functional.normalize(Md.squeeze(0), dim=0)
    return float((Mr * Md).sum(dim=0).mean())


_RMVPE_CENTS = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))


def _rmvpe_f0(wav_16k_np, sess):
    hidden = sess.run(None, {"audio": wav_16k_np.astype(np.float32)[None]})[0][0]
    nf = wav_16k_np.shape[0] // 160 + 1
    h = hidden[:nf]
    center = np.argmax(h, axis=1) + 4
    sal_pad = np.pad(h, ((0, 0), (4, 4)))
    starts = center - 4
    ends = center + 5
    todo_s = np.stack([sal_pad[i, starts[i]:ends[i]] for i in range(h.shape[0])], axis=0)
    todo_c = np.stack([_RMVPE_CENTS[starts[i]:ends[i]] for i in range(h.shape[0])], axis=0)
    devided = np.sum(todo_s * todo_c, axis=1) / np.sum(todo_s, axis=1)
    maxx = np.max(sal_pad, axis=1)
    devided[maxx <= 0.03] = 0
    f0 = 10 * (2 ** (devided / 1200))
    f0[f0 == 10] = 0
    return f0


def f0_metrics(ref, deg, shift_semitones=0.0, sr=48000):
    """用 RMVPE 提取 f0（对变声后的音色鲁棒；detect_pitch 会锁谐波假频）。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(
        os.path.join(ROOT, "assets", "rmvpe", "rmvpe.onnx"),
        providers=["CPUExecutionProvider"],
    )
    ref16 = torchaudio.functional.resample(ref, 48000, 16000).numpy()
    deg16 = torchaudio.functional.resample(deg, 48000, 16000).numpy()
    fr = torch.from_numpy(_rmvpe_f0(ref16, sess))
    fd = torch.from_numpy(_rmvpe_f0(deg16, sess))
    n = min(fr.shape[0], fd.shape[0])
    fr, fd = fr[:n], fd[:n]
    fr = fr * (2 ** (shift_semitones / 12))
    mask = (fr > 0) & (fd > 0)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    fr, fd = fr[mask], fd[mask]
    corr = float(np.corrcoef(fr.numpy(), fd.numpy())[0, 1])
    rmse = float(torch.sqrt(((fr - fd) ** 2).mean()))
    return corr, rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--f0-shift", type=float, default=0.0)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.chdir(ROOT)
    ref, _ = load_48k(args.ref)
    deg, _ = load_48k(args.cand)
    ref, deg = align(ref, deg)
    print(f"ref={args.ref}  cand={args.cand}  n={ref.shape[0]} ({ref.shape[0]/48000:.1f}s)")

    ref16 = torchaudio.functional.resample(ref, 48000, 16000)
    deg16 = torchaudio.functional.resample(deg, 48000, 16000)

    m = {}
    m["pesq"] = pesq_mos(ref16, deg16)
    m["stoi"] = float(_stoi(ref16.numpy(), deg16.numpy(), 16000, extended=False))
    m["sisnr"] = si_snr(ref, deg)
    m["lsd"] = lsd_db(ref, deg)
    m["melcos"] = mel_cos(ref, deg)
    corr, rmse = f0_metrics(ref, deg, args.f0_shift)
    m["f0corr"] = corr
    m["f0rmse"] = rmse

    label = args.label or os.path.basename(args.cand)
    print(f"[{label}] PESQ={m['pesq']:.3f}  STOI={m['stoi']:.3f}  "
          f"SI-SNR={m['sisnr']:.1f}dB  LSD={m['lsd']:.1f}dB  "
          f"melCos={m['melcos']:.3f}  f0Corr={corr:.3f}  f0RMSE={rmse:.0f}Hz")
    with open(os.path.join(ROOT, "logs", "offline_test", "metrics.jsonl"), "a", encoding="utf-8") as f:
        import json as _json
        f.write(_json.dumps({"label": label, "ref": args.ref, "cand": args.cand,
                             "f0_shift": args.f0_shift, **m}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
