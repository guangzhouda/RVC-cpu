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
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --list-devices
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --selftest            # 不声卡，测 RTF
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py                       # 麦克风实时变声
    .venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --provider cpu        # 强制纯 CPU

注意: 解码器 ONNX 是按 (block/crossfade/extra) 参数烘焙的，
     默认使用 assets/weights/hatsune_miku_model.stream.onnx（250ms/50ms/2.5s）。
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

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def printt(*args):
    print(*args, flush=True)


class RVCOnnxEngine:
    def __init__(
        self,
        hubert_onnx="assets/hubert/_exp_hubert.onnx",
        dec_onnx="assets/weights/hatsune_miku_model.stream.onnx",
        dec_meta="assets/weights/hatsune_miku_model.stream.onnx.json",
        index_path="assets/weights/hatsune_miku_model.index",
        index_rate=0.75,
        pitch=0,
        provider="auto",
        f0method="fcpe",
        rmvpe_onnx="assets/rmvpe/rmvpe.onnx",
        rng_seed=1234,
    ):
        self.providers = self._pick_providers(provider)
        printt("providers:", self.providers)
        so = ort.SessionOptions()
        self.hsess = ort.InferenceSession(hubert_onnx, sess_options=so, providers=self.providers)

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
        if index_path and os.path.exists(index_path) and index_rate > 0:
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
            printt("index loaded, rate =", index_rate)
        self.index_rate = index_rate
        self.pitch = pitch  # 变调（半音）

        self.f0method = f0method
        if f0method == "rmvpe":
            self.rmvpe_sess = ort.InferenceSession(rmvpe_onnx, sess_options=so, providers=self.providers)
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

    def _pick_providers(self, provider):
        if provider == "cpu":
            return ["CPUExecutionProvider"]
        avail = ort.get_available_providers()
        if "DmlExecutionProvider" in avail:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
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
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0).float()
        if self.pitch:
            f0 = f0 * pow(2, self.pitch / 12)
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self._f0_mel_min) * 254 / (self._f0_mel_max - self._f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        return torch.round(f0_mel).long(), f0

    def process_block(self, indata):
        """输入一个 block_frame 采样点（float32 numpy），返回同长度的变声结果。"""
        t0 = time.perf_counter()
        indata = torch.from_numpy(np.ascontiguousarray(indata, dtype=np.float32))

        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = indata
        n16 = 160 * (self.block_frame // self.zc + 1)
        self.input_wav_res[:-n16] = self.input_wav_res[n16:].clone()
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

        # 2) faiss 特征检索混合
        if self.index is not None:
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
        else:
            pitch, pitchf = self._f0_post(self.fcpe.infer(
                self.input_wav_res[-(self.block_frame_16k + 800):].unsqueeze(0).float(),
                sr=16000, decoder_mode="local_argmax", threshold=0.006,
            ).float().squeeze())
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


def run_live(engine, in_device, out_device, latency="high", wasapi_exclusive=False):
    import sounddevice as sd

    block = engine.block_frame
    sr = engine.sr
    stats = []

    def callback(indata, outdata, frames, _t, status):
        if status:
            print(status)
        out, dt = engine.process_block(indata[:, 0].copy())
        outdata[:, 0] = out
        stats.append(dt)
        if len(stats) >= 100:
            avg = np.mean(stats) * 1000
            printt(f"avg {avg:.0f}ms / block {block / sr * 1000:.0f}ms -> RTF {block / sr / (avg / 1000):.2f}x")
            stats.clear()

    if out_device is not None:
        try:
            info = sd.query_devices(out_device)
            dsr = info.get("default_samplerate", 0)
            printt(f"output device: {info['name']}  default_samplerate={dsr}Hz")
            if dsr < 44100:
                printt(f"[warn] 该输出设备只有 {dsr}Hz（很可能是蓝牙免提模式），音质会明显变差，建议换 48k 设备")
        except Exception:
            pass
    printt(f"sr={sr} block={block} ({block / sr * 1000:.0f}ms) latency={latency} exclusive={wasapi_exclusive}  按 Ctrl+C 退出")
    stream_args = dict(samplerate=sr, blocksize=block, channels=1, dtype="float32",
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

    cpu_ms, cpu_p95 = time_ep(["CPUExecutionProvider"], intra=8)
    dml_ms, dml_p95 = time_ep(["DmlExecutionProvider", "CPUExecutionProvider"])
    printt("[auto] 解码器基准: cpu=%.0fms  dml=%.0fms" % (cpu_ms, dml_ms))
    if dml_ms < cpu_ms / threshold:
        printt("[auto] 使用 DirectML（核显）")
        return "dml"
    printt("[auto] DML 未显著快于 CPU，使用纯 CPU（更稳）")
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="RVC 实时变声（ONNX + DML/CPU）")
    ap.add_argument("--hubert", default="assets/hubert/_exp_hubert.onnx")
    ap.add_argument("--onnx", default="assets/weights/hatsune_miku_model.stream.onnx")
    ap.add_argument("--meta", default="assets/weights/hatsune_miku_model.stream.onnx.json")
    ap.add_argument("--index", default="assets/weights/hatsune_miku_model.index")
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--pitch", type=int, default=0, help="变调（半音），例如 12 = +1 八度")
    ap.add_argument("--f0method", default="fcpe", choices=["fcpe", "rmvpe"],
                    help="音高算法：fcpe 最快 / rmvpe 更准（ONNX 加速后同样实时）")
    ap.add_argument("--provider", default="auto", choices=["auto", "dml", "cpu"],
                    help="auto=优先 DML；cpu=强制纯 CPU")
    ap.add_argument("--latency", default="high", choices=["low", "high"],
                    help="声卡缓冲延迟（默认 high）")
    ap.add_argument("--wasapi-exclusive", action="store_true",
                    help="WASAPI 独占模式，进一步降低输入/输出缓冲延迟")
    ap.add_argument("--input-device", type=int, default=None)
    ap.add_argument("--output-device", type=int, default=None)
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

    if args.provider == "auto":
        args.provider = bench_provider(args.onnx, args.meta)
    os.chdir(ROOT)
    torch.set_num_threads(1)  # torch 侧（fcpe/resample/SOLA）单线程；ORT 自行管理线程

    engine = RVCOnnxEngine(
        hubert_onnx=args.hubert,
        dec_onnx=args.onnx,
        dec_meta=args.meta,
        index_path=args.index,
        index_rate=args.index_rate,
        pitch=args.pitch,
        provider=args.provider,
        f0method=args.f0method,
        rng_seed=args.seed,
    )
    engine.warmup()

    if args.selftest:
        run_selftest(engine, blocks=args.blocks, wav_path=args.selftest_wav)
    else:
        run_live(engine, args.input_device, args.output_device,
                 latency=args.latency, wasapi_exclusive=args.wasapi_exclusive)


if __name__ == "__main__":
    main()
