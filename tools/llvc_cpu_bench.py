"""
LLVC CPU 流式离线基准 — 复用 llvc_live.py 的真实流式推理路径，不开声卡。

用法（在仓库根目录）:
    python tools/llvc_cpu_bench.py --wav input.wav
"""
import argparse
import json
import os
import sys
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as tat

import llvc_live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(llvc_live.LLVC_DIR, "llvc_models/models/checkpoints/llvc/G_500000.pth"),
    )
    parser.add_argument(
        "--config",
        default=os.path.join(llvc_live.LLVC_DIR, "experiments/llvc/config.json"),
    )
    parser.add_argument("--wav", required=True, help="输入音频")
    parser.add_argument("--out", default="llvc_cpu_bench_out.wav")
    parser.add_argument(
        "--chunk-factor",
        type=int,
        default=2,
        help="chunk = L * dec_chunk_size * factor",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-chunks", type=int, default=0, help="0=整个文件")
    parser.add_argument("--report", default="", help="可选 JSON 报告输出路径")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model, sr = llvc_live.load_model(args.checkpoint, args.config)
    streamer = llvc_live.LLVCStreamer(model, args.chunk_factor)
    streamer.warmup()

    audio, in_sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    audio = torch.from_numpy(audio.astype(np.float32))
    if in_sr != sr:
        audio = tat.Resample(in_sr, sr, dtype=torch.float32)(audio)
    audio = audio.numpy()
    original_len = len(audio)

    chunk_len = streamer.chunk_len
    if len(audio) % chunk_len != 0:
        pad_len = chunk_len - (len(audio) % chunk_len)
        audio = np.pad(audio, (0, pad_len))

    chunks = np.split(audio, len(audio) // chunk_len)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]

    print(
        f"sr={sr} chunk={chunk_len} ({chunk_len / sr * 1000:.1f}ms) "
        f"chunk_factor={args.chunk_factor} threads={args.threads} chunks={len(chunks)}"
    )

    out_chunks = []
    times = []
    for i, chunk in enumerate(chunks):
        t0 = time.perf_counter()
        out = streamer.push(chunk.astype(np.float32))
        dt = time.perf_counter() - t0
        times.append(dt)
        if len(out) > 0:
            out_chunks.append(out.copy())
        print(f"chunk {i}: total={dt*1000:.0f}ms out={len(out)}")

    # Flush one additional zero chunk to emit the final delayed chunk.
    t0 = time.perf_counter()
    out = streamer.push(np.zeros(chunk_len, dtype=np.float32))
    dt = time.perf_counter() - t0
    times.append(dt)
    if len(out) > 0:
        out_chunks.append(out.copy())
    print(f"flush: total={dt*1000:.0f}ms out={len(out)}")

    if out_chunks:
        out_audio = np.concatenate(out_chunks)
    else:
        out_audio = np.zeros(0, dtype=np.float32)
    out_audio = out_audio[:original_len]

    steady_times = np.array(times[1:]) if len(times) > 1 else np.array(times)
    budget = chunk_len / sr
    avg_ms = float(steady_times.mean() * 1000)
    p95_ms = float(np.percentile(steady_times, 95) * 1000)
    realtime_ok = bool(np.percentile(steady_times, 95) < budget)
    rtf = float(budget / steady_times.mean())

    print(
        f"\navg={avg_ms:.0f}ms  p95={p95_ms:.0f}ms / budget={budget*1000:.1f}ms -> "
        f"{'REALTIME OK' if realtime_ok else 'TOO SLOW'} (RTF={rtf:.2f}x)"
    )
    sf.write(args.out, out_audio, sr, subtype="FLOAT")
    print(f"saved -> {args.out}")

    if args.report:
        report = {
            "engine": "llvc_streaming",
            "device": "cpu",
            "wav": os.path.abspath(args.wav),
            "out": os.path.abspath(args.out),
            "sample_rate": int(sr),
            "chunk_factor": int(args.chunk_factor),
            "chunk_len": int(chunk_len),
            "chunks": int(len(chunks)),
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "rtf": rtf,
            "realtime_ok": realtime_ok,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
