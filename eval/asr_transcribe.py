# -*- coding: utf-8 -*-
"""Whisper 转写原声与变声输出，评估可懂度（WER 终裁）。

用法:
    .venv-onnx-stream\Scripts\python.exe tools\asr_transcribe.py
    依赖: whisper(已手动装入 site-packages) + base 模型(C:\\Users\\<user>\\.cache\\whisper\\base.pt)
"""
import os
import sys

import numpy as np
import whisper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_16k(path):
    import soundfile as sf
    import torch
    import torchaudio
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(-1)
    if sr != 16000:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()
    return a


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1].lower() == h[j - 1].lower() else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)] / len(r)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="logs/offline_test/reference/ref_16k.wav")
    ap.add_argument("--cands", default="logs/offline_test/miku/v2_p0.wav,logs/offline_test/miku/v2_p12.wav")
    args = ap.parse_args()
    os.chdir(ROOT)

    model = whisper.load_model("base")
    ref_text = None
    for path in [args.ref] + [c.strip() for c in args.cands.split(",") if c.strip()]:
        audio = load_16k(path)
        result = model.transcribe(audio, language="en", fp16=False)
        text = result["text"].strip()
        if ref_text is None:
            ref_text = text
            print(f"[ref ] {text}")
        else:
            print(f"[{os.path.basename(path)}] WER={wer(ref_text, text):.3f} | {text}")


if __name__ == "__main__":
    main()
