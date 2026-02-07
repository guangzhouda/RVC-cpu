#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# RVC content encoder (vec-768-layer-12) fp32 vs fp16 速度对比（离线验证用）
#
# 说明：
# - 这是“离线工具脚本”，用于评估不同 encoder.onnx 在当前 ORT provider 下的推理耗时。
# - 不影响“运行时无 Python”的目标（SDK 运行时仍是 C/C++ DLL + onnxruntime.dll）。
# - 默认用随机输入模拟 RVC 的 16k 窗口长度（T=total_frames*160）。

import argparse
import time
from typing import List

import numpy as np
import onnxruntime as ort


def _parse_providers(raw: str) -> List[str]:
    ps = [p.strip() for p in raw.split(",") if p.strip()]
    return ps


def _make_session(
    model_path: str, providers: List[str], threads: int, opt: str
) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads and threads > 0:
        so.intra_op_num_threads = int(threads)

    # opt level
    opt = opt.lower().strip()
    if opt == "disable":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif opt == "basic":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    elif opt == "extended":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    else:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    avail = ort.get_available_providers()
    chosen = [p for p in providers if p in avail]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")

    return ort.InferenceSession(model_path, sess_options=so, providers=chosen)


def _bench(sess: ort.InferenceSession, T: int, iters: int, warmup: int) -> float:
    inp = sess.get_inputs()[0]
    name = inp.name
    x = np.random.uniform(-1.0, 1.0, size=(1, 1, int(T))).astype(np.float32)

    for _ in range(int(warmup)):
        sess.run(None, {name: x})

    t0 = time.perf_counter()
    for _ in range(int(iters)):
        sess.run(None, {name: x})
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / float(iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True, help="fp32 encoder.onnx path")
    ap.add_argument(
        "--fp16",
        required=True,
        help="fp16 encoder.onnx path (or any alternative encoder)",
    )
    ap.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated providers. Example: 'DmlExecutionProvider,CPUExecutionProvider' or 'CUDAExecutionProvider,CPUExecutionProvider'",
    )
    ap.add_argument(
        "--threads", type=int, default=0, help="ORT intra-op threads (0=default)"
    )
    ap.add_argument(
        "--opt",
        default="all",
        choices=["disable", "basic", "extended", "all"],
        help="ORT graph optimization level",
    )
    ap.add_argument("--iters", type=int, default=200, help="Benchmark iterations")
    ap.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    ap.add_argument(
        "--T",
        default="3840,7040,12800",
        help="Comma-separated input lengths at 16k (T). Typical: 3840(~0.24s),7040(~0.44s),12800(~0.80s).",
    )
    args = ap.parse_args()

    providers = _parse_providers(args.providers)
    Ts = [int(x.strip()) for x in args.T.split(",") if x.strip()]

    s32 = _make_session(args.fp32, providers, args.threads, args.opt)
    s16 = _make_session(args.fp16, providers, args.threads, args.opt)

    print("available providers:", ort.get_available_providers())
    print("chosen providers:", s32.get_providers())
    print("opt:", args.opt, "threads:", args.threads)
    print("models:")
    print("  fp32:", args.fp32)
    print("  fp16:", args.fp16)

    for T in Ts:
        ms32 = _bench(s32, T, args.iters, args.warmup)
        ms16 = _bench(s16, T, args.iters, args.warmup)
        sp = (ms32 / ms16) if ms16 > 0 else float("nan")
        print(f"T={T:5d}  fp32={ms32:8.3f} ms  fp16={ms16:8.3f} ms  speedup={sp:6.3f}x")


if __name__ == "__main__":
    main()
