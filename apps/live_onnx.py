# -*- coding: utf-8 -*-
"""
RVC 实时变声（纯 ONNX 推理，无需 CUDA / PyTorch 模型推理）
============================================================

后端自动选择：优先 DirectML（核显/独显，Windows），否则纯 CPU。
- 前端: HuBERT ONNX 提取音素特征（多线程无 NaN）
- 音高: fcpe（CPU torch，最快）
- 检索: faiss 特征混合（可选）
- 合成: 流式声码器 ONNX（与 gui_v1.py 相同的滑动窗口 + SOLA 拼接）

用法（在仓库根目录）:
    .venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --list-devices
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --selftest            # 不声卡，测 RTF
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py                       # 麦克风实时变声
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --provider cpu        # 强制纯 CPU

注意: 解码器 ONNX 是按 (block/crossfade/extra) 参数烘焙的，
     默认使用 assets/weights/miku/onnx/stream150.onnx（150ms 流式）。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import onnxruntime as ort  # 必须先于 faiss 导入（Windows DLL 冲突）
import faiss  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchaudio.transforms as tat  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # apps/ 的上一级 = 项目根
sys.path.insert(0, ROOT)


def printt(*args):
    print(*args, flush=True)


class RVCOnnxEngine:
    def __init__(
        self,
        hubert_onnx="assets/hubert/_exp_hubert.onnx",
        dec_onnx="assets/weights/miku/onnx/stream150.onnx",
        dec_meta="assets/weights/miku/onnx/stream150.onnx.json",
        index_path="assets/weights/miku/hatsune_miku_model.index",
        embedded_index=None,
        index_rate=0.75,
        pitch=0,
        auto_pitch=False,
        auto_pitch_mode="binary",
        rms_mix_rate=None,
        target_quantiles=None,
        auto_pitch_target=220.0,
        auto_pitch_threshold=165.0,
        auto_pitch_male_shift=12.0,
        auto_pitch_female_shift=0.0,
        auto_pitch_max=14.0,
        provider="auto",
        dml_device=None,
        f0method="rmvpe",
        rmvpe_onnx="assets/rmvpe/rmvpe.onnx",
        rng_seed=1234,
        cpu_encoder=False,
    ):
        self.dml_device = dml_device
        self.providers = self._pick_providers(provider, dml_device)
        # 编码器（hubert/rmvpe）含 GRU/FFT 等 DML 支持薄弱的算子，可选强制 CPU；
        # 解码器（纯卷积）留在 DML。诊断用：DML 输出异常时此模式可定位问题。
        enc_providers = ["CPUExecutionProvider"] if cpu_encoder else self.providers
        printt("providers:", self.providers, "encoder:", enc_providers)
        so = ort.SessionOptions()
        self.hsess = ort.InferenceSession(hubert_onnx, sess_options=so, providers=enc_providers)

        meta = json.load(open(dec_meta, encoding="utf-8"))
        stream = meta["stream"]
        self.zc = int(stream["zc"])
        self.block_frame = int(stream["block_frame"])
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(stream["crossfade_frame"])
        self.sola_buffer_frame = int(stream["sola_buffer_frame"])
        self.sola_search_frame = int(stream["sola_search_frame"])
        self.extra_frame = int(stream["extra_frame"])
        self.skip_head = int(stream["skip_head"])
        self.phone_length = int(stream["phone_length"])
        self.rnd_shape = [int(v) for v in meta["inputs"]["rnd"]]
        self.sr = int(meta["target_sr"])
        printt(
            f"stream: sr={self.sr} block={self.block_frame} "
            f"({self.block_frame / self.sr * 1000:.0f}ms) "
            f"extra={self.extra_frame / self.sr:.2f}s phone={self.phone_length}"
        )
        self.dec = ort.InferenceSession(dec_onnx, sess_options=so, providers=self.providers)
        self.dec_input_names = {x.name for x in self.dec.get_inputs()}
        self.rng = np.random.default_rng(rng_seed)

        self.index = None
        self.big_npy = None
        self.centers = None
        if embedded_index and os.path.exists(embedded_index):
            # 内嵌质心表模式：index 蒸馏成的 (K,768) 矩阵，随模型分发（1~2MB）
            self.centers = np.load(embedded_index).astype(np.float32)
            printt(f"embedded index loaded: {self.centers.shape} ({os.path.getsize(embedded_index)/1e6:.2f}MB), rate = {index_rate}")
        elif index_path and os.path.exists(index_path) and index_rate > 0:
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt("index loaded, rate =", index_rate)
        self.index_rate = index_rate
        self.pitch = pitch  # 变调（半音）
        self.auto_pitch = auto_pitch
        self.auto_pitch_mode = auto_pitch_mode
        self.rms_mix_rate = rms_mix_rate if rms_mix_rate is not None else None
        self.target_quantiles = target_quantiles
        self.f0_hist = []          # CDF 模式：滚动 f0 帧缓存（有声帧）
        self._tgt_q = None         # 目标模型 f0 分布的分位数（log2 Hz）
        self._src_q = None
        self._remap = None
        self._cdf_rebuild_cnt = 0
        if target_quantiles and os.path.exists(target_quantiles):
            self._tgt_q = np.load(target_quantiles).astype(np.float64)
            printt(f"target quantiles loaded: {self._tgt_q.shape} ({os.path.getsize(target_quantiles)/1e3:.0f}KB)")
        self.auto_pitch_target = auto_pitch_target
        self.auto_pitch_threshold = auto_pitch_threshold
        self.auto_pitch_male_shift = auto_pitch_male_shift
        self.auto_pitch_female_shift = auto_pitch_female_shift
        self.auto_pitch_max = auto_pitch_max
        self.auto_shift = 0.0  # 自动音高补偿（半音），平滑更新
        self._med_hist = []    # 滑动窗口：最近 ~0.8 秒有声 f0 帧（跨块稳定统计，换人过渡 ~1.5s）

        self.f0method = f0method
        if f0method == "rmvpe":
            self.rmvpe_sess = ort.InferenceSession(rmvpe_onnx, sess_options=so, providers=enc_providers)
            self._rmvpe_cents = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
            printt("f0 method: rmvpe (ONNX)")
        else:
            import torchfcpe.mel_extractor as _fcpe_mel
            _fcpe_mel.print = lambda *a, **k: None  # 静音 torchfcpe 的调试打印
            from torchfcpe import spawn_bundled_infer_model
            self.fcpe = spawn_bundled_infer_model("cpu")
            printt("f0 method: fcpe")

        # 滑动缓冲 / SOLA 状态（与 gui_v1.py 一致）
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(160 * self.input_wav.shape[0] // self.zc, dtype=torch.float32)
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, dtype=torch.float32)
        self.fade_in_window = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame)) ** 2
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(orig_freq=self.sr, new_freq=16000, dtype=torch.float32)
        self.cache_pitch = torch.zeros(1024, dtype=torch.long)
        self.cache_pitchf = torch.zeros(1024, dtype=torch.float32)
        self._f0_mel_min = 1127 * np.log(1 + 50 / 700)
        self._f0_mel_max = 1127 * np.log(1 + 1100 / 700)

    def _pick_providers(self, provider, dml_device=None):
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if provider == "cuda":
            if "CUDAExecutionProvider" in avail:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            printt("[warn] CUDA EP 不可用，回退")
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in avail:
            # 双显卡笔记本（核显+独显）务必显式指定 DML 设备编号：
            # 本机实测 device_id=0 最快（15ms vs 默认 21ms vs device 1 的 55ms）。
            opts = {"device_id": dml_device} if dml_device is not None else {}
            return [("DmlExecutionProvider", opts), "CPUExecutionProvider"]
        printt("[warn] DirectML 不可用，回退纯 CPU")
        return ["CPUExecutionProvider"]

    def _decode_rmvpe(self, hidden, thred=0.03):
        """hidden: (n_frames, 360) numpy float32 -> f0 Hz（与 RMVPE.decode 逐行一致）"""
        center = np.argmax(hidden, axis=1) + 4
        sal_pad = np.pad(hidden, ((0, 0), (4, 4)))
        starts = center - 4
        ends = center + 5
        todo_s = np.stack([sal_pad[i, starts[i]:ends[i]] for i in range(hidden.shape[0])], axis=0)
        todo_c = np.stack([self._rmvpe_cents[starts[i]:ends[i]] for i in range(hidden.shape[0])], axis=0)
        devided = np.sum(todo_s * todo_c, axis=1) / np.sum(todo_s, axis=1)
        maxx = np.max(sal_pad, axis=1)
        devided[maxx <= thred] = 0
        f0 = 10 * (2 ** (devided / 1200))
        f0[f0 == 10] = 0
        return f0

    def _f0_post(self, f0):
        if self.auto_pitch and self.auto_pitch_mode == "cdf" and self._remap is not None:
            # 分布级映射：先按 CDF 重映射，再叠加用户手动 pitch
            if torch.is_tensor(f0):
                f0n = f0.numpy()
            else:
                f0n = np.asarray(f0)
            f0n = self._remap(f0n)
            f0 = torch.from_numpy(f0n).float()
            if self.pitch:
                f0 = f0 * pow(2, self.pitch / 12)
            f0_mel = 1127 * torch.log(1 + f0 / 700)
            f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self._f0_mel_min) * 254 / (self._f0_mel_max - self._f0_mel_min) + 1
            f0_mel[f0_mel <= 1] = 1
            f0_mel[f0_mel > 255] = 255
            return torch.round(f0_mel).long(), f0
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0).float()
        total_shift = self.pitch + (self.auto_shift if self.auto_pitch else 0.0)
        if total_shift:
            f0 = f0 * pow(2, total_shift / 12)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self._f0_mel_min) * 254 / (self._f0_mel_max - self._f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return torch.round(f0_mel).long(), f0

    def _update_auto_pitch(self, f0):
        """自动音高补偿：

        cdf：分布级映射（f0_map.py）——源 f0 分位数 -> 目标模型 f0 分位数，
             对齐中位数+展宽+形状，男->女最准确；需 --target-quantiles
        binary（默认）：男声固定 +12 / 女声固定 0
        continuous：12*log2(target/med) 连续映射
        """
        if self.auto_pitch_mode == "cdf" and self._tgt_q is not None:
            self._update_cdf(f0)
            # fallback：cdf 映射就绪（约 2~3 秒预热）前，用 continuous 自适应定位（无性别假设）
            self._update_shift(f0, force_continuous=True)
            return
        self._update_shift(f0)

    def _update_shift(self, f0, force_continuous=False):
        """shift 计算：continuous（默认，无性别假设，男女自适应）或 binary（可选）。

        continuous: 12*log2(target/med) —— 男声自动 +10~11，女声自动 ~0
        binary:     输入 med < 阈值 -> male_shift，否则 female_shift（需预设性别倾向）
        """
        f0 = np.asarray(f0, dtype=np.float64)
        nz = f0[f0 > 0]
        if nz.size < 8:
            return  # 静音/弱音块不更新
        # 滑动窗口：累积有声帧（最多 80 帧 ≈ 0.8 秒），中位数跨块统计更稳
        self._med_hist.extend(nz.tolist())
        if len(self._med_hist) > 80:
            self._med_hist = self._med_hist[-80:]
        med = float(np.median(self._med_hist))
        if med <= 0:
            return
        use_binary = (not force_continuous) and self.auto_pitch_mode == "binary"
        if use_binary:
            if med < self.auto_pitch_threshold:
                shift = float(self.auto_pitch_male_shift)
            else:
                shift = float(self.auto_pitch_female_shift)
        else:
            shift = 12.0 * np.log2(self.auto_pitch_target / med)
        shift = max(-self.auto_pitch_max, min(self.auto_pitch_max, shift))
        # 统一快速收敛（0.7/0.3）：目标固定时慢低通没有意义
        self.auto_shift = 0.7 * self.auto_shift + 0.3 * shift
        if self.auto_shift != self.auto_shift:  # NaN 防护
            self.auto_shift = 0.0

    def _update_cdf(self, f0):
        """CDF 模式：维护滚动 f0 缓存（约 3 秒有声帧），周期性重建分位数映射。

        映射闭包单调连续，重建时轻微变化不会跳变；映射本身替代固定 shift。
        """
        from apps.f0_map import MIN_VOICED_FRAMES, build_quantiles, make_remap
        f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
        voiced = f0[(f0 > 0) & (f0 >= 50) & (f0 <= 1100)]
        if voiced.size:
            self.f0_hist.extend(voiced.tolist())
            # 只保留最近 ~3 秒（150ms/块 x 20 块，10ms/帧约 300 帧）
            if len(self.f0_hist) > 600:
                self.f0_hist = self.f0_hist[-600:]
        self._cdf_rebuild_cnt += 1
        # 每 20 块（约 3 秒）重建一次映射
        if self._cdf_rebuild_cnt % 20 != 0:
            return
        src = np.asarray(self.f0_hist, dtype=np.float64)
        if src.size < MIN_VOICED_FRAMES:
            return
        try:
            from apps.f0_map import _P as _F0_P
            src_q = np.quantile(np.log2(src), _F0_P).astype(np.float64)
            src_q = src_q + np.arange(src_q.size, dtype=np.float64) * 1e-6
            self._src_q = src_q
            self._remap = make_remap(src_q, self._tgt_q)  # tgt_q 已由校准保存
        except ValueError:
            return  # 有声帧不足，沿用旧映射

    def process_block(self, indata):
        """输入一个 block_frame 采样点（float32 numpy），返回同长度的变声结果。"""
        t0 = time.perf_counter()
        indata = torch.from_numpy(np.ascontiguousarray(indata, dtype=np.float32))

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = indata
        n16 = 160 * (self.block_frame // self.zc + 1)
        # 前移量必须等于 block_frame_16k（160*block/zc），与官方 gui_v1.py 一致。
        # 之前误用 n16(=block_frame_16k+160) 前移，导致每块时间轴多滑 160 采样(10ms)，
        # 43 块累积漂移 430ms，音高/内容错位（f0Corr≈0、STOI≈0.26 的根因）。
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[self.block_frame_16k:].clone()
        self.input_wav_res[-n16:] = self.resampler(self.input_wav[-self.block_frame - 2 * self.zc:])[160:]

        # 1) HuBERT 特征（ONNX）
        src = self.input_wav_res.numpy()[None]
        T = src.shape[1]
        pad = (640 - T % 640) % 640
        srcp = np.pad(src, ((0, 0), (0, pad))) if pad else src
        feats_raw = self.hsess.run(None, {"source": srcp.astype(np.float32)})[0]
        nf = -(-T // 320)
        feats = torch.from_numpy(feats_raw[:, :nf, :])
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        # 2) 特征检索混合（faiss index 或内嵌质心表二选一）
        if self.centers is not None:
            npy = feats[0][self.skip_head // 2:].float().numpy().astype("float32")
            sim = npy @ self.centers.T  # (T, K) 内积相似度
            k = min(8, self.centers.shape[0])
            ix = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            score = np.take_along_axis(sim, ix, axis=1)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy2 = np.sum(self.centers[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats[0][self.skip_head // 2:] = (
                torch.from_numpy(npy2).unsqueeze(0) * self.index_rate
                + (1 - self.index_rate) * feats[0][self.skip_head // 2:]
            )
        elif self.index is not None:
            npy = feats[0][self.skip_head // 2:].float().numpy().astype("float32")
            score, ix = self.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy2 = np.sum(self.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats[0][self.skip_head // 2:] = (
                    torch.from_numpy(npy2).unsqueeze(0) * self.index_rate
                    + (1 - self.index_rate) * feats[0][self.skip_head // 2:]
                )

        # 3) 音高（fcpe 或 rmvpe-ONNX）
        if self.f0method == "rmvpe":
            fex = 5120 * ((self.block_frame_16k + 800 - 1) // 5120 + 1) - 160
            win = self.input_wav_res[-fex:]
            hidden = self.rmvpe_sess.run(None, {"audio": win.numpy()[None].astype(np.float32)})[0][0]  # (Tf, 360)
            nf = win.shape[0] // 160 + 1
            f0 = self._decode_rmvpe(hidden[:nf, :], thred=0.03)  # 切帧轴 -> (帧, 360)
            pitch, pitchf = self._f0_post(f0)
            if self.auto_pitch:
                self._update_auto_pitch(f0)
        else:
            pitch, pitchf = self._f0_post(self.fcpe.infer(
                self.input_wav_res[-(self.block_frame_16k + 800):].unsqueeze(0).float(),
                sr=16000, decoder_mode="local_argmax", threshold=0.006,
            ).float().squeeze())
            if self.auto_pitch:
                self._update_auto_pitch(pitchf.numpy() if hasattr(pitchf, "numpy") else np.asarray(pitchf))
        shift = self.block_frame_16k // 160
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        self.cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
        self.cache_pitchf[4 - pitch.shape[0]:] = pitchf[3:-1]
        c_pitch = self.cache_pitch[None, -self.phone_length:]
        c_pitchf = self.cache_pitchf[None, -self.phone_length:]

        # 4) 流式声码器（ONNX）
        feats_i = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats_i = feats_i[:, :self.phone_length, :]
        rnd = self.rng.standard_normal(self.rnd_shape, dtype=np.float32)
        feed = {
            "phone": feats_i.numpy().astype(np.float32),
            "phone_lengths": np.array([self.phone_length], dtype=np.int64),
            "pitch": c_pitch.numpy().astype(np.int64),
            "pitchf": c_pitchf.numpy().astype(np.float32),
            "sid": np.array([0], dtype=np.int64),
            "rnd": rnd,
        }
        feed = {k: v for k, v in feed.items() if k in self.dec_input_names}
        infer_wav = self.dec.run(None, feed)[0]
        infer_wav = torch.from_numpy(infer_wav).squeeze(0).squeeze(0).float()

        # 4.5) 音量包络混合（gui_v1 同款 rms_mix_rate）：输出响度跟随输入起伏，更自然
        if self.rms_mix_rate is not None and self.rms_mix_rate < 1.0:
            import librosa
            src = self.input_wav[self.extra_frame:self.extra_frame + infer_wav.shape[0]].numpy()
            rms1 = librosa.feature.rms(y=src, frame_length=4 * self.zc, hop_length=self.zc)
            rms1 = torch.from_numpy(rms1)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1,
                                 mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav.numpy(), frame_length=4 * self.zc, hop_length=self.zc)
            rms2 = torch.from_numpy(rms2)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1,
                                 mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1.0 - self.rms_mix_rate))

        # 5) SOLA 拼接
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input ** 2, torch.ones(1, 1, self.sola_buffer_frame)) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[:self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[self.block_frame: self.block_frame + self.sola_buffer_frame]

        out = infer_wav[:self.block_frame].numpy()
        return out, time.perf_counter() - t0

    def warmup(self, blocks=5):
        z = np.zeros(self.block_frame, dtype=np.float32)
        for _ in range(blocks):
            self.process_block(z)
        printt("warmup done")


def run_selftest(engine, blocks=40, wav_path=None):
    if wav_path:
        import soundfile as sf
        audio, sr = sf.read(wav_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        if sr != engine.sr:
            import torchaudio.transforms as tat
            audio = tat.Resample(sr, engine.sr, dtype=torch.float32)(torch.from_numpy(audio.astype(np.float32))).numpy()
    else:
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(blocks * engine.block_frame) * 0.05).astype(np.float32)

    times = []
    for i in range(min(blocks, len(audio) // engine.block_frame)):
        chunk = audio[i * engine.block_frame: (i + 1) * engine.block_frame]
        out, dt = engine.process_block(chunk)
        times.append(dt * 1000)
    times = np.array(times[1:]) if len(times) > 1 else np.array(times)
    budget = engine.block_frame / engine.sr * 1000
    avg, p95 = float(times.mean()), float(np.percentile(times, 95))
    printt(f"providers={engine.providers}")
    printt(f"avg={avg:.0f}ms  p95={p95:.0f}ms / budget={budget:.0f}ms  ->  RTF={budget / avg:.2f}x")
    printt("REALTIME OK" if p95 < budget else "TOO SLOW")


def resolve_device(spec, kind, list_on_error=True):
    """设备解析：None=系统默认；纯数字=索引；其他=名称子串匹配（方向通道>0）。

    kind: "input"（麦克风/虚拟输入）或 "output"（耳机/扬声器/虚拟输出）
    """
    import sounddevice as sd

    if spec is None or spec == "":
        return None
    if str(spec).isdigit():
        return int(spec)
    devs = sd.query_devices()
    need_in = kind == "input"
    cands = []
    for i, d in enumerate(devs):
        ch = d["max_input_channels"] if need_in else d["max_output_channels"]
        if ch <= 0:
            continue
        if spec.lower() in d["name"].lower():
            cands.append((i, d["name"], d["default_samplerate"]))
    if cands:
        idx, name, dsr = cands[0]
        printt(f"[device] {kind}: #{idx} {name} ({dsr}Hz)")
        return idx
    if list_on_error:
        printt(f"[error] 找不到包含 '{spec}' 的{('输入' if need_in else '输出')}设备，可用设备：")
        for i, d in enumerate(devs):
            ch = d["max_input_channels"] if need_in else d["max_output_channels"]
            if ch > 0:
                printt(f"  {i}: {d['name']}  ({d['default_samplerate']}Hz)")
        sys.exit(1)
    return None


def run_live(engine, in_device, out_device, latency="high", wasapi_exclusive=False,
             denoise=False, denoise_out=False, silence_dbfs=None, denoise_mix=0.7,
             agc=False, agc_target_db=-23.0, agc_max_gain=16.0,
             record_in=None, record_out=None):
    import sounddevice as sd

    block = engine.block_frame
    sr = engine.sr
    stats = []

    # 录制（测试用）：input.wav=原始麦克风，output.wav=变声输出（均设备采样率 int16）
    import wave as _wave

    def _open_wav(path, rate):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        w = _wave.open(path, "wb")
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        return w

    def _write_wav(w, x):
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes())

    rec_in = rec_out = None

    # GTCRN 前置/后置降噪（16k 域）与静音门限
    denoiser_in = denoiser_out = None
    if denoise or denoise_out:
        from apps.gtcrn_denoise import GTCRNDenoiser
        if denoise:
            denoiser_in = GTCRNDenoiser(dml_device=engine.dml_device or 0)
            printt("[denoise] 前置 GTCRN 已启用")
        if denoise_out:
            denoiser_out = GTCRNDenoiser(dml_device=engine.dml_device or 0)
            printt("[denoise] 后置 GTCRN 已启用")
        global _rs_48_16, _rs_16_48
        _rs_48_16 = tat.Resample(orig_freq=sr, new_freq=16000, dtype=torch.float32)
        _rs_16_48 = tat.Resample(orig_freq=16000, new_freq=sr, dtype=torch.float32)
    if silence_dbfs is not None:
        printt(f"[gate] 静音门限 {silence_dbfs}dBFS（低于阈值输出静音）")
    _agc_gain = 1.0
    _agc_target = 10.0 ** (agc_target_db / 20.0)
    if agc:
        printt(f"[agc] 自动增益已启用（目标 {agc_target_db}dBFS，最大 x{agc_max_gain:.0f}）")

    def _apply_agc(x):
        nonlocal _agc_gain
        rms = float(np.sqrt((x ** 2).mean() + 1e-12))
        if rms > 1e-5:
            g = _agc_target / rms
            g = max(1.0, min(agc_max_gain, g))  # 只放大不缩小（防爆音）
            _agc_gain = 0.8 * _agc_gain + 0.2 * g  # 平滑
        return (x * _agc_gain).astype(np.float32)

    def _denoise_block(x, den):
        x16 = _rs_48_16(torch.from_numpy(x)).numpy()
        x16 = den.process(x16)
        return _rs_16_48(torch.from_numpy(x16)).numpy()

    # 设备原生采样率开流，避免 Windows 音频引擎的低质量重采样（44.1k 设备尤其明显：
    # 48k 流会被 WASAPI 二次重采样，声音发闷/"含水"）。内部用 torchaudio 高质量重采样。
    def _dev_sr(dev):
        try:
            return int(sd.query_devices(dev)["default_samplerate"]) if dev is not None else None
        except Exception:
            return None

    in_sr = _dev_sr(in_device)
    out_sr = _dev_sr(out_device)
    dev_sr = (in_sr or out_sr) or sr
    need_resample = dev_sr != sr
    if need_resample:
        # 输入输出设备采样率不一致时按输出设备定（共享模式两者通常一致）
        rs_in = tat.Resample(orig_freq=dev_sr, new_freq=sr, dtype=torch.float32)
        rs_out = tat.Resample(orig_freq=sr, new_freq=dev_sr, dtype=torch.float32)
        dev_block = int(round(block * dev_sr / sr))
        printt(f"[device-sr] 设备采样率 {dev_sr}Hz != 模型 {sr}Hz，启用内部高质量重采样（块 {dev_block} 采样）")
    else:
        rs_in = rs_out = None
        dev_block = block

    def callback(indata, outdata, frames, _t, status):
        if status:
            print(status)
        x = indata[:, 0].copy()
        if rec_in is not None:
            _write_wav(rec_in, x)
        if rs_in is not None:
            x = rs_in(torch.from_numpy(x)).numpy()
            if x.shape[0] < block:
                x = np.pad(x, (0, block - x.shape[0]))
            x = x[:block]
        # 前置 GTCRN 降噪（干湿混合：保留 (1-mix) 原声，防止轻声/辅音被吞导致口齿不清）
        if denoiser_in is not None:
            x_den = _denoise_block(x, denoiser_in)
            x = (denoise_mix * x_den + (1.0 - denoise_mix) * x).astype(np.float32)
        # AGC：小声自动放大（降噪之后，避免放大噪声）
        if agc:
            x = _apply_agc(x)
        # 瞬态/静音门限（低于阈值直接输出静音，不做推理）
        if silence_dbfs is not None:
            rms = float(np.sqrt((x ** 2).mean() + 1e-12))
            db = 20.0 * np.log10(max(rms, 1e-7))
            if db < silence_dbfs:
                outdata[:, 0] = 0
                if rec_out is not None:
                    _write_wav(rec_out, np.zeros(dev_block, dtype=np.float32))
                return
        out, dt = engine.process_block(x)
        # 后置 GTCRN 降噪
        if denoiser_out is not None:
            out = _denoise_block(out, denoiser_out)
        if rs_out is not None:
            out = rs_out(torch.from_numpy(out)).numpy()
            if out.shape[0] < dev_block:
                out = np.pad(out, (0, dev_block - out.shape[0]))
            out = out[:dev_block]
        outdata[:, 0] = out
        if rec_out is not None:
            _write_wav(rec_out, out)
        stats.append(dt)
        if len(stats) >= 100:
            avg = np.mean(stats) * 1000
            printt(f"avg {avg:.0f}ms / block {block / sr * 1000:.0f}ms -> RTF {block / sr / (avg / 1000):.2f}x")
            stats.clear()

    if in_device is not None:
        try:
            info = sd.query_devices(in_device)
            printt(f"input  device: {info['name']}  default_samplerate={info.get('default_samplerate', 0)}Hz")
        except Exception:
            pass
    if out_device is not None:
        try:
            info = sd.query_devices(out_device)
            dsr = info.get("default_samplerate", 0)
            printt(f"output device: {info['name']}  default_samplerate={dsr}Hz")
            if dsr < 44100:
                printt(f"[warn] 该输出设备只有 {dsr}Hz（很可能是蓝牙免提模式），音质会明显变差，建议换 48k 设备")
        except Exception:
            pass
    if record_in or record_out:
        if record_in:
            rec_in = _open_wav(record_in, dev_sr)
            printt(f"[record] 录制输入 -> {record_in} ({dev_sr}Hz int16)")
        if record_out:
            rec_out = _open_wav(record_out, dev_sr)
            printt(f"[record] 录制输出 -> {record_out} ({dev_sr}Hz int16)")
    printt(f"model sr={sr} block={block} ({block / sr * 1000:.0f}ms)  dev sr={dev_sr} "
           f"latency={latency} exclusive={wasapi_exclusive}  按 Ctrl+C 退出")
    stream_args = dict(samplerate=dev_sr, blocksize=dev_block, channels=1, dtype="float32",
                       device=(in_device, out_device), callback=callback)
    if wasapi_exclusive:
        stream_args["extra_settings"] = sd.WasapiSettings(exclusive=True)
    else:
        stream_args["latency"] = latency
    try:
        with sd.Stream(**stream_args):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        printt("已停止")
    finally:
        for w, tag in ((rec_in, "input"), (rec_out, "output")):
            if w is not None:
                w.close()
                printt(f"[record] {tag} 已保存并关闭")

def run_inject(engine, wav_path, out_path, denoise=False, denoise_out=False, silence_dbfs=None,
               denoise_mix=0.7, agc=False, agc_target_db=-23.0, agc_max_gain=16.0):
    """文件注入模式：把 WAV 当实时输入逐块处理（与声卡实时共用同一引擎+同一前置链），
    输出写 WAV。用于"录一次原声 -> 分别测 E/F 方案/参数"的公平对比。"""
    import torchaudio as _ta
    sr = engine.sr
    block = engine.block_frame

    # 前置链（与 run_live 一致）
    denoiser_in = denoiser_out = None
    if denoise or denoise_out:
        from apps.gtcrn_denoise import GTCRNDenoiser
        if denoise:
            denoiser_in = GTCRNDenoiser(dml_device=engine.dml_device or 0)
            printt("[denoise] 前置 GTCRN 已启用")
        if denoise_out:
            denoiser_out = GTCRNDenoiser(dml_device=engine.dml_device or 0)
            printt("[denoise] 后置 GTCRN 已启用")
        global _rs_48_16, _rs_16_48
        _rs_48_16 = tat.Resample(orig_freq=sr, new_freq=16000, dtype=torch.float32)
        _rs_16_48 = tat.Resample(orig_freq=16000, new_freq=sr, dtype=torch.float32)
    _agc_gain = 1.0
    _agc_target = 10.0 ** (agc_target_db / 20.0)
    if agc:
        printt(f"[agc] 自动增益已启用（目标 {agc_target_db}dBFS，最大 x{agc_max_gain:.0f}）")
    if silence_dbfs is not None:
        printt(f"[gate] 静音门限 {silence_dbfs}dBFS（低于阈值输出静音）")

    def _apply_agc(x):
        nonlocal _agc_gain
        rms = float(np.sqrt((x ** 2).mean() + 1e-12))
        if rms > 1e-5:
            g = _agc_target / rms
            g = max(1.0, min(agc_max_gain, g))
            _agc_gain = 0.8 * _agc_gain + 0.2 * g
        return (x * _agc_gain).astype(np.float32)

    def _denoise_block(x, den):
        x16 = _rs_48_16(torch.from_numpy(x)).numpy()
        x16 = den.process(x16)
        return _rs_16_48(torch.from_numpy(x16)).numpy()

    audio, asr = _ta.load(wav_path)
    audio = audio.mean(0)
    if asr != sr:
        audio = _ta.functional.resample(audio, asr, sr)
    n = audio.shape[0]
    if n < block:
        printt(f"[inject] 音频过短（{n/sr:.2f}s < {block/sr:.2f}s）")
        return
    rem = n % block
    if rem:
        audio = torch.nn.functional.pad(audio, (0, block - rem))
    printt(f"[inject] {wav_path} ({asr}Hz, {n/sr:.1f}s) -> 48k 逐块注入, block={block/sr*1000:.0f}ms")
    outs = []
    nblk = audio.shape[0] // block
    t0 = time.perf_counter()
    for i in range(nblk):
        x = audio[i * block:(i + 1) * block].numpy().astype(np.float32)
        if denoiser_in is not None:
            x_den = _denoise_block(x, denoiser_in)
            x = (denoise_mix * x_den + (1.0 - denoise_mix) * x).astype(np.float32)
        if agc:
            x = _apply_agc(x)
        if silence_dbfs is not None:
            rms = float(np.sqrt((x ** 2).mean() + 1e-12))
            db = 20.0 * np.log10(max(rms, 1e-7))
            if db < silence_dbfs:
                outs.append(np.zeros(block, dtype=np.float32))
                continue
        out, _dt = engine.process_block(x)
        if denoiser_out is not None:
            out = _denoise_block(out, denoiser_out)
        outs.append(out)
    el = time.perf_counter() - t0
    out_all = np.concatenate(outs)[:n] if outs else np.zeros(0, dtype=np.float32)
    peak = float(np.abs(out_all).max()) if out_all.size else 0.0
    if peak > 0.99:
        out_all = out_all * (0.95 / peak)
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    _ta.save(out_path, torch.from_numpy(out_all).unsqueeze(0), sr)
    printt(f"[inject] 完成: {out_path} ({out_all.shape[0]/sr:.1f}s, peak={peak:.2f}, "
           f"avg {el/nblk*1000:.0f}ms/block = {nblk*block/sr/el:.1f}x 实时)")


def bench_provider(onnx_path, meta_path, threshold=1.3, runs=8):
    """快速对比 CPU 与 DML 的实际推理速度（只用解码器小模型），返回选中的 provider。

    auto 模式下启动时调用；避免在核显不给力的机器上盲目用 DML 反而变慢。
    """
    if "DmlExecutionProvider" not in ort.get_available_providers():
        printt("[auto] DirectML 不可用，使用 CPU")
        return "cpu"
    import json as _json
    meta = _json.load(open(meta_path, encoding="utf-8"))
    inputs = meta["inputs"]
    rng = np.random.default_rng(0)
    feed = {}
    for k, v in inputs.items():
        if k in ("phone", "pitchf", "rnd"):
            feed[k] = rng.standard_normal(v).astype(np.float32)
        elif k in ("phone_lengths", "pitch", "sid"):
            feed[k] = np.zeros(v, dtype=np.int64)

    def time_ep(providers, intra=None):
        # int8 量化模型的 ConvInteger 算子 CPU EP 不支持，CPU 基准失败时返回 None（跳过）
        try:
            so = ort.SessionOptions()
            if intra is not None and providers[0] == "CPUExecutionProvider":
                so.intra_op_num_threads = intra
            sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
            sess.run(None, feed)  # warmup（DML 首次会编译着色器，计入本次）
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter()
                sess.run(None, feed)
                ts.append((time.perf_counter() - t0) * 1000)
            return float(np.mean(ts)), float(np.percentile(ts, 95))
        except Exception as exc:
            printt(f"[auto] 基准失败（{type(exc).__name__}），跳过该后端")
            return None

    cpu_res = time_ep(["CPUExecutionProvider"], intra=8)
    dml_res = time_ep(["DmlExecutionProvider", "CPUExecutionProvider"])
    cpu_ms = cpu_res[0] if cpu_res else None
    dml_ms = dml_res[0] if dml_res else None
    if cpu_ms is None and dml_ms is not None:
        printt("[auto] CPU 不可用（int8 模型常见），使用 DirectML")
        return "dml"
    if cpu_ms is None or dml_ms is None:
        printt("[auto] 基准不完整，使用 DML")
        return "dml"
    printt("[auto] 解码器基准: cpu=%.0fms  dml=%.0fms" % (cpu_ms, dml_ms))
    if dml_ms < cpu_ms / threshold:
        printt("[auto] 使用 DirectML（核显）")
        return "dml"
    printt("[auto] DML 未显著快于 CPU，使用纯 CPU（更稳）")
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="RVC 实时变声（ONNX + DML/CPU）")
    ap.add_argument("--hubert", default="assets/hubert/_exp_hubert.onnx")
    ap.add_argument("--onnx", default="assets/weights/miku/onnx/stream150.onnx")
    ap.add_argument("--meta", default="assets/weights/miku/onnx/stream150.onnx.json")
    ap.add_argument("--index", default="assets/weights/miku/hatsune_miku_model.index")
    ap.add_argument("--embedded-index", default=None,
                    help="内嵌质心表（export/embed_index.py 生成，1~2MB 随模型分发）；与 --index 二选一")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--rms-mix", type=float, default=None,
                    help="音量包络混合 0~1（gui_v1 同款，默认关；0.5 = 输出响度跟随输入起伏，更自然）")
    ap.add_argument("--pitch", type=int, default=0, help="变调（半音），例如 12 = +1 八度")
    ap.add_argument("--auto-pitch", action="store_true",
                    help="自动音高补偿（男女通用，无需手动 pitch）")
    ap.add_argument("--auto-pitch-mode", default="binary", choices=["binary", "continuous", "cdf"],
                    help="cdf=分布级分位数映射（最准，需 --target-quantiles）；binary=男+12/女+0；continuous=连续映射")
    ap.add_argument("--auto-pitch-threshold", type=float, default=165.0,
                    help="binary 模式性别判定阈值（Hz），男声一般 <155，女声一般 >165")
    ap.add_argument("--auto-pitch-male-shift", type=float, default=12.0,
                    help="binary 模式男声固定补偿（半音），默认 +12")
    ap.add_argument("--auto-pitch-female-shift", type=float, default=0.0,
                    help="binary 模式女声固定补偿（半音），默认 0")
    ap.add_argument("--target-quantiles", default=None,
                    help="目标模型 f0 分布分位数文件（.npy，由 calibrate_pitch.py 生成；cdf 模式必填）")
    ap.add_argument("--auto-pitch-target", type=float, default=220.0,
                    help="continuous 模式目标基准 f0（Hz）")
    ap.add_argument("--auto-pitch-max", type=float, default=14.0, help="自动补偿上限（半音）")
    ap.add_argument("--f0method", default="rmvpe", choices=["fcpe", "rmvpe"],
                    help="音高算法：默认 rmvpe（本机 DML 实测比 fcpe 更快更准）；fcpe 可作对比")
    ap.add_argument("--provider", default="auto", choices=["auto", "dml", "cuda", "cpu"],
                    help="auto=优先 DML；cuda=强制 CUDA（需主 .venv 的 onnxruntime-gpu）；cpu=纯 CPU")
    ap.add_argument("--dml-device", type=int, default=None,
                    help="DML 设备编号（双显卡笔记本建议显式指定；本机实测 0 最快）")
    ap.add_argument("--latency", default="high", choices=["low", "high"],
                    help="声卡缓冲延迟（默认 high）")
    ap.add_argument("--wasapi-exclusive", action="store_true",
                    help="WASAPI 独占模式，进一步降低输入/输出缓冲延迟")
    ap.add_argument("--denoise", action="store_true",
                    help="前置降噪：GTCRN（16k 频域模型，assets/gtcrn/gtcrn_simple.onnx）")
    ap.add_argument("--denoise-mix", type=float, default=0.7,
                    help="降噪干湿比 0~1（默认 0.7）：1=纯降噪（轻声可能被吞），低=保留更多原声细节防口齿不清")
    ap.add_argument("--denoise-out", action="store_true",
                    help="后置降噪：输出侧再过一次 GTCRN")
    ap.add_argument("--silence-dbfs", type=float, default=None,
                    help="瞬态/静音门限（dBFS）：输入低于该电平输出静音（如 -60），对应大饼的枪声过滤/静音抑制")
    ap.add_argument("--agc", action="store_true",
                    help="自动增益（AGC）：小声说话时自动放大到合适电平，防止轻声被吞/口齿不清")
    ap.add_argument("--agc-target-db", type=float, default=-23.0, help="AGC 目标电平（dBFS，默认 -23）")
    ap.add_argument("--agc-max-gain", type=float, default=16.0, help="AGC 最大增益（倍，默认 16）")
    ap.add_argument("--input-device", default=None,
                    help="麦克风/输入设备：数字索引或名称子串（如 0 或 'Microphone'），缺省=系统默认")
    ap.add_argument("--output-device", default=None,
                    help="扬声器/耳机/虚拟声卡：数字索引或名称子串，缺省=系统默认")
    ap.add_argument("--check-device", action="store_true",
                    help="只解析并打印所选设备后退出（不启动）")
    ap.add_argument("--record-in", default=None,
                    help="录制原始麦克风输入到 WAV（设备采样率 int16）；测试用")
    ap.add_argument("--record-out", default=None,
                    help="录制变声输出到 WAV（设备采样率 int16）；测试用")
    ap.add_argument("--inject-wav", default=None,
                    help="文件注入模式：把 WAV 当实时输入逐块处理（不开声卡，与实时同一引擎/前置链），公平对比用")
    ap.add_argument("--inject-out", default=None,
                    help="注入模式输出 WAV 路径（默认 logs/inject_out.wav）")
    ap.add_argument("--selftest", action="store_true", help="不开声卡，测推理速度")
    ap.add_argument("--selftest-wav", default="", help="selftest 用真实音频文件（默认随机噪声）")
    ap.add_argument("--blocks", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        printt(sd.query_devices())
        return

    in_idx = resolve_device(args.input_device, "input", list_on_error=True)
    out_idx = resolve_device(args.output_device, "output", list_on_error=True)
    if args.check_device:
        return

    if args.provider == "auto":
        if "DmlExecutionProvider" not in ort.get_available_providers() and "CUDAExecutionProvider" in ort.get_available_providers():
            args.provider = "cuda"
            printt("[auto] 无 DirectML，使用 CUDA EP")
        else:
            args.provider = bench_provider(args.onnx, args.meta)
    os.chdir(ROOT)
    torch.set_num_threads(1)  # torch 侧（fcpe/resample/SOLA）单线程；ORT 自行管理线程

    engine = RVCOnnxEngine(
        hubert_onnx=args.hubert,
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
        auto_pitch_threshold=args.auto_pitch_threshold,
        auto_pitch_male_shift=args.auto_pitch_male_shift,
        auto_pitch_female_shift=args.auto_pitch_female_shift,
        auto_pitch_max=args.auto_pitch_max,
        provider=args.provider,
        dml_device=args.dml_device,
        f0method=args.f0method,
        rng_seed=args.seed,
    )
    engine.warmup()

    if args.selftest:
        run_selftest(engine, blocks=args.blocks, wav_path=args.selftest_wav)
    elif args.inject_wav:
        run_inject(engine, args.inject_wav, args.inject_out or "logs/inject_out.wav",
                   denoise=args.denoise, denoise_out=args.denoise_out,
                   silence_dbfs=args.silence_dbfs, denoise_mix=args.denoise_mix,
                   agc=args.agc, agc_target_db=args.agc_target_db, agc_max_gain=args.agc_max_gain)
    else:
        run_live(engine, in_idx, out_idx,
                 latency=args.latency, wasapi_exclusive=args.wasapi_exclusive,
                 denoise=args.denoise, denoise_out=args.denoise_out,
                 silence_dbfs=args.silence_dbfs, denoise_mix=args.denoise_mix,
                 agc=args.agc, agc_target_db=args.agc_target_db, agc_max_gain=args.agc_max_gain,
                 record_in=args.record_in, record_out=args.record_out)


if __name__ == "__main__":
    main()
