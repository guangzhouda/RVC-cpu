# -*- coding: utf-8 -*-
"""GTCRN 流式降噪（前置/后置降噪用，16kHz 频域模型，48.2K 参数超轻量）。

GTCRN: Grouped Temporal Convolutional Recurrent Network (speech enhancement)
- 输入: mix [1,257,1,2] = STFT 频谱(512 FFT, 实部/虚部) 单帧
- 状态: conv_cache / tra_cache / inter_cache（GRU/卷积缓存，跨帧传递）
- 输出: enh 增强频谱 + 新状态

用法: denoiser = GTCRNDenoiser(); out = denoiser.process(x_16k)  # x: float32 numpy 16k
"""
import os

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

N_FFT = 512
HOP = 256
SR = 16000


class GTCRNDenoiser:
    def __init__(self, onnx_path="assets/gtcrn/gtcrn_simple.onnx", providers=None, dml_device=0):
        self.onnx_path = onnx_path
        if providers is None:
            avail = ort.get_available_providers()
            if "DmlExecutionProvider" in avail:
                providers = [("DmlExecutionProvider", {"device_id": dml_device}), "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.state = {
            "conv_cache": np.zeros((2, 1, 16, 16, 33), dtype=np.float32),
            "tra_cache": np.zeros((2, 3, 1, 1, 16), dtype=np.float32),
            "inter_cache": np.zeros((2, 1, 33, 16), dtype=np.float32),
        }
        self.window = torch.hann_window(N_FFT)
        self._n_fft = N_FFT
        self._hop = HOP
        self._buf = np.zeros(0, dtype=np.float32)  # 跨块残留（帧对齐）

    def reset(self):
        for k in self.state:
            self.state[k] = np.zeros_like(self.state[k])
        self._buf = np.zeros(0, dtype=np.float32)

    def _process_frames(self, spec_real, spec_imag):
        """逐帧推理（T 帧），返回增强的实/虚部。spec_real/imag: (1, F, T)"""
        T = spec_real.shape[2]
        out_r = np.empty_like(spec_real)
        out_i = np.empty_like(spec_imag)
        for t in range(T):
            mix = np.zeros((1, 257, 1, 2), dtype=np.float32)
            mix[0, :, 0, 0] = spec_real[0, :, t]
            mix[0, :, 0, 1] = spec_imag[0, :, t]
            feed = {"mix": mix}
            for k, v in self.state.items():
                feed[k] = v
            outs = self.sess.run(None, feed)
            enh = outs[0][0, :, 0, :]  # (257, 2)
            out_r[0, :, t] = enh[:, 0]
            out_i[0, :, t] = enh[:, 1]
            self.state["conv_cache"] = outs[1]
            self.state["tra_cache"] = outs[2]
            self.state["inter_cache"] = outs[3]
        return out_r, out_i

    def process(self, x_16k):
        """x_16k: float32 numpy 1D @16kHz -> 增强后的 16k numpy（流式，输出长度 = 输入长度）"""
        buf_len = self._buf.shape[0] if self._buf.size else 0
        x = torch.from_numpy(np.ascontiguousarray(x_16k, dtype=np.float32))
        if buf_len:
            x = torch.cat([torch.from_numpy(self._buf), x])
        # 分帧：保证帧数整数
        pad = (HOP - (x.shape[0] - N_FFT) % HOP) % HOP
        if pad:
            x = F.pad(x, (0, pad))
        spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                          window=self.window, center=False, return_complex=True)  # (F, T)
        spec_r = spec.real.numpy()[None]  # (1, F, T)
        spec_i = spec.imag.numpy()[None]
        enh_r, enh_i = self._process_frames(spec_r, spec_i)
        # 手动 OLA iSTFT（torch.istft 的边缘检查在流式块上会误报）
        spec_enh = torch.complex(torch.from_numpy(enh_r[0]), torch.from_numpy(enh_i[0]))  # (F, T)
        T = spec_enh.shape[1]
        xf = torch.fft.irfft(spec_enh, n=N_FFT, dim=0) * self.window[:, None]  # (N_FFT, T)
        out_len = (T - 1) * HOP + N_FFT
        y = torch.zeros(out_len, dtype=torch.float32)
        win_sq = torch.zeros(out_len, dtype=torch.float32)
        for t in range(T):
            s = t * HOP
            y[s:s + N_FFT] += xf[:, t]
            win_sq[s:s + N_FFT] += self.window ** 2
        win_sq = torch.clamp(win_sq, min=1e-8)
        y = y / win_sq
        # 丢弃与上一块重叠的前 buf_len 采样，只吐新段；_buf 取真实音频尾部（不含 padding 区）
        real_len = buf_len + len(x_16k)
        seg = y[buf_len:real_len].numpy()
        if seg.size < len(x_16k):
            seg = np.pad(seg, (0, len(x_16k) - seg.size))
        self._buf = y[real_len - HOP:real_len].numpy().copy()
        return seg
