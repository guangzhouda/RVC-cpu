"""
LLVC 实时变声 — 使用真正的 LLVC 流式模型（causal + cached buffers，无需 SOLA）

与 compare_infer.ChunkedInferer（论文对照基线）不同，这里加载的是
llvc_models/models/checkpoints/llvc/G_500000.pth 蒸馏模型，
chunk 之间靠 enc_buf/dec_buf/out_buf/convnet_pre_ctx 保持连续，拼接天然无缝。

用法:
    python llvc_live.py                 # 麦克风实时变声
    python llvc_live.py --selftest      # 不开声卡，仅测推理速度 (RTF)
    python llvc_live.py --list-devices  # 列出音频设备
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

LLVC_DIR = "E:/Projects/LLVC"
sys.path.insert(0, LLVC_DIR)

from model import Net  # noqa: E402


def load_model(checkpoint_path, config_path):
    with open(config_path) as f:
        config = json.load(f)
    model = Net(**config["model_params"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, config["data"]["sr"]


class LLVCStreamer:
    """按 infer.infer_stream 的方式做逐 chunk 流式推理。

    每个 chunk 长 L * dec_chunk_size * chunk_factor 个采样点，
    输入前拼上前一 chunk 的最后 2L 个点作为卷积上下文，
    整个输入流先丢掉开头 L 个点（对应训练时的 L 点 lookahead 平移）。
    """

    def __init__(self, model, chunk_factor):
        self.model = model
        self.L = model.L
        self.chunk_len = model.dec_chunk_size * self.L * chunk_factor
        self.ctx = np.zeros(2 * self.L, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        self.started = False
        self.reset_buffers()

    def reset_buffers(self):
        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(1, "cpu")
        if hasattr(self.model, "convnet_pre"):
            self.convnet_pre_ctx = self.model.convnet_pre.init_ctx_buf(1, "cpu")
        else:
            self.convnet_pre_ctx = None

    def warmup(self):
        dummy = np.zeros(self.chunk_len, dtype=np.float32)
        for _ in range(3):
            self._infer_chunk(dummy)
        self.ctx[:] = 0
        self.reset_buffers()

    def _infer_chunk(self, chunk):
        x = np.concatenate([self.ctx, chunk])
        self.ctx = chunk[-2 * self.L:].copy()
        x = torch.from_numpy(x).float()[None, None]
        with torch.inference_mode():
            y, self.enc_buf, self.dec_buf, self.out_buf, self.convnet_pre_ctx = \
                self.model(
                    x,
                    self.enc_buf, self.dec_buf, self.out_buf,
                    self.convnet_pre_ctx,
                    pad=(not self.model.lookahead),
                )
        return y[0, 0].numpy()

    def push(self, samples):
        """喂入任意长度的输入采样，返回已凑满 chunk 的输出（可能为空）。"""
        self.pending = np.concatenate([self.pending, samples])
        if not self.started:
            if len(self.pending) <= self.L:
                return np.zeros(0, dtype=np.float32)
            self.pending = self.pending[self.L:]  # 训练时的 L 点前移
            self.started = True
        # 积压超过 4 个 chunk 说明推理跟不上，丢弃旧数据保住实时性
        if len(self.pending) > 4 * self.chunk_len:
            drop = len(self.pending) - 2 * self.chunk_len
            self.pending = self.pending[drop:]
            print(f"[warn] inference too slow, dropped {drop} samples")
        outs = []
        while len(self.pending) >= self.chunk_len:
            chunk = self.pending[:self.chunk_len]
            self.pending = self.pending[self.chunk_len:]
            outs.append(self._infer_chunk(chunk))
        if outs:
            return np.concatenate(outs)
        return np.zeros(0, dtype=np.float32)


def run_selftest(streamer, sr, seconds=5.0):
    rng = np.random.default_rng(0)
    total = int(seconds * sr)
    n_chunks = total // streamer.chunk_len
    times = []
    for _ in range(n_chunks):
        chunk = (rng.standard_normal(streamer.chunk_len) * 0.05).astype(np.float32)
        t0 = time.perf_counter()
        out = streamer.push(chunk)
        times.append(time.perf_counter() - t0)
        assert len(out) in (0, streamer.chunk_len), f"unexpected out len {len(out)}"
    chunk_ms = streamer.chunk_len / sr * 1000
    avg_ms = np.mean(times[1:]) * 1000
    p99_ms = np.percentile(times[1:], 99) * 1000
    print(f"chunk = {streamer.chunk_len} samples ({chunk_ms:.1f}ms)")
    print(f"avg inference = {avg_ms:.2f}ms  p99 = {p99_ms:.2f}ms")
    print(f"RTF = {chunk_ms / avg_ms:.2f}x realtime "
          f"({'OK' if avg_ms < chunk_ms else 'TOO SLOW'})")


def run_live(streamer, sr, in_device, out_device):
    import sounddevice as sd

    block = streamer.chunk_len
    infer_times = []
    out_fifo = np.zeros(0, dtype=np.float32)

    def callback(indata, outdata, frames, _t, status):
        nonlocal out_fifo
        if status:
            print(status)
        t0 = time.perf_counter()
        out = streamer.push(indata[:, 0].copy())
        infer_times.append(time.perf_counter() - t0)
        out_fifo = np.concatenate([out_fifo, out])
        n = min(len(out_fifo), frames)
        outdata[:n, 0] = out_fifo[:n]
        outdata[n:, 0] = 0
        out_fifo = out_fifo[n:]
        if len(infer_times) >= 100:
            avg = np.mean(infer_times) * 1000
            print(f"avg process = {avg:.2f}ms / block = {block / sr * 1000:.1f}ms")
            infer_times.clear()

    print(f"sr={sr} block={block} ({block / sr * 1000:.1f}ms)  Ctrl+C 退出")
    with sd.Stream(samplerate=sr, blocksize=block, channels=1, dtype="float32",
                   device=(in_device, out_device), callback=callback):
        while True:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description="LLVC realtime voice conversion")
    parser.add_argument("--checkpoint", default=os.path.join(
        LLVC_DIR, "llvc_models/models/checkpoints/llvc/G_500000.pth"))
    parser.add_argument("--config", default=os.path.join(
        LLVC_DIR, "experiments/llvc/config.json"))
    parser.add_argument("--chunk-factor", type=int, default=2,
                        help="chunk = L*dec_chunk_size*factor, factor=2 即 26ms")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--selftest", action="store_true",
                        help="不开声卡，用随机噪声测推理速度")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    torch.set_num_threads(args.threads)
    model, sr = load_model(args.checkpoint, args.config)
    streamer = LLVCStreamer(model, args.chunk_factor)
    print(f"model L={model.L} dec_chunk_size={model.dec_chunk_size} sr={sr}")
    streamer.warmup()

    if args.selftest:
        run_selftest(streamer, sr)
    else:
        run_live(streamer, sr, args.input_device, args.output_device)


if __name__ == "__main__":
    main()
