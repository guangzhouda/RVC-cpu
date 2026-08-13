# -*- coding: utf-8 -*-
"""
导出 RMVPE 为 ONNX（输入 16k 音频 [1,T] -> 输出 360 音分显著性 [1,Tf,360]）。

与 torch 版 RMVPE 逐帧一致（f0 maxdiff ~6e-5 Hz）。
注意: torch.stft(return_complex=True) 无法导出 ONNX，这里用 return_complex=False 等价实现。

用法:
    python tools/export_rmvpe_onnx.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from infer.lib.rmvpe import E2E, MelSpectrogram  # noqa: E402


class MelSpectrogramExport(MelSpectrogram):
    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        window = torch.hann_window(win_length_new).to(audio.device)
        fft = torch.stft(
            audio, n_fft=n_fft_new, hop_length=hop_length_new,
            win_length=win_length_new, window=window, center=center,
            return_complex=False, onesided=True,
        )  # (B, F, T, 2)
        magnitude = torch.sqrt(fft[..., 0].pow(2) + fft[..., 1].pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RmvpeWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = MelSpectrogramExport(False, 128, 16000, 1024, 160, None, 30, 8000)
        self.net = E2E(4, 1, (2, 2))
        ckpt = torch.load("assets/rmvpe/rmvpe.pt", map_location="cpu")
        self.net.load_state_dict(ckpt)
        self.net.eval().float()

    def forward(self, audio):  # [1, T] @16k
        mel = self.mel(audio, center=True)  # [1, 128, Tf]
        n_frames = mel.shape[-1]
        n_pad = (32 - n_frames % 32) % 32  # 恒 pad 到 32 的倍数（无分支）
        mel = F.pad(mel, (0, n_pad))
        return self.net(mel)  # [1, 360, Tf_pad]


def main():
    torch.set_num_threads(1)
    wrap = RmvpeWrapper().eval()
    dummy = torch.randn(1, 4960).float()  # 实时窗口 = 5120*ceil((block_16k+800)/5120)-160
    out_path = "assets/rmvpe/rmvpe.onnx"
    torch.onnx.export(
        wrap,
        (dummy,),
        out_path,
        input_names=["audio"],
        output_names=["hidden"],
        dynamic_axes={"audio": {1: "time"}, "hidden": {1: "time"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print("saved ->", out_path, "%.1fMB" % (os.path.getsize(out_path) / 1e6))


if __name__ == "__main__":
    main()
