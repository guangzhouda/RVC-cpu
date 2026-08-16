# -*- coding: utf-8 -*-
"""块级 STOI 诊断：输出块 vs 输入块（短窗对齐），并检查块边界连续性。"""
import os
import sys
import wave

import numpy as np
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from pystoi import stoi as _stoi  # noqa: E402
from apps.live_onnx import RVCOnnxEngine  # noqa: E402


def main():
    with wave.open("logs/offline_test/reference/ref_16k.wav", "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    x16 = torch.from_numpy(pcm.astype(np.float32) / 32767.0)
    audio = torchaudio.functional.resample(x16, 16000, 48000)

    eng = RVCOnnxEngine(dml_device=0)
    eng.warmup()

    block = eng.block_frame
    prev_tail = None
    stois = []
    jumps = []
    for i in range(40):
        chunk = audio[i * block:(i + 1) * block]
        out, dt = eng.process_block(chunk.numpy())

        # 块级 STOI：输入块 vs 输出块（16k）
        ci = torchaudio.functional.resample(chunk, 48000, 16000).numpy()
        co = torchaudio.functional.resample(torch.from_numpy(out), 48000, 16000).numpy()
        s = _stoi(ci, co, 16000, extended=False)
        stois.append(s)

        # 块边界连续性：本块开头 vs 上一块结尾的互相关（输出侧）
        if prev_tail is not None:
            head = out[:960]  # 20ms
            tail = prev_tail[-960:]
            corr = float(np.corrcoef(head, tail)[0, 1])
            jumps.append(corr)
        prev_tail = out

        if i < 8 or i % 5 == 0:
            print(f"blk{i:02d} stoi={s:.3f}" + (f" boundary_corr={jumps[-1]:.3f}" if jumps else ""))

    stois = np.array(stois)
    jumps = np.array(jumps)
    print(f"block STOI: mean={stois.mean():.3f} median={np.median(stois):.3f} min={stois.min():.3f}")
    print(f"boundary corr: mean={jumps.mean():.3f} median={np.median(jumps):.3f} (1.0=平滑连续)")


if __name__ == "__main__":
    main()
