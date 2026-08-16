# -*- coding: utf-8 -*-
"""压缩 RVC 检索索引（faiss .index）

原理：index 本质是目标音色音频的逐帧特征库（768 维 float32）。
压缩 = ① 均匀抽样（帧数除以 N） ② 标量量化（SQ8: 1字节/维，÷4；SQ4: 半字节/维，÷8）。
检索本来就是近似 kNN，压缩对音色还原影响很小。

用法:
    .venv-onnx-stream/Scripts/python.exe export/compress_index.py --input assets/weights/xxx/added_xxx.index --output assets/weights/xxx/added_xxx.small.index --sample 4 --quant sq8

参数:
    --sample N   每 N 帧取 1（默认 4；数据越多可越大，30 分钟数据用 4~8）
    --quant      none | sq8 | sq4（默认 sq8）
    --k          抽样后要求最少保留的帧数（默认 30000）
"""
import argparse
import os
import sys

import numpy as np
import faiss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample", type=int, default=4, help="每 N 帧取 1")
    ap.add_argument("--quant", default="sq8", choices=["none", "sq8", "sq4"])
    ap.add_argument("--min-frames", type=int, default=30000, help="抽样后最少保留帧数")
    args = ap.parse_args()

    in_size = os.path.getsize(args.input)
    index = faiss.read_index(args.input)
    dim = index.d
    total = index.ntotal
    print(f"input: {args.input}  ({in_size/1e6:.1f}MB, {total} 帧, dim={dim})")

    # 重建全部向量
    vecs = index.reconstruct_n(0, total).astype(np.float32)
    del index

    # 均匀抽样
    step = max(1, args.sample)
    n_target = total // step
    if n_target < args.min_frames:
        step = max(1, total // args.min_frames)
        n_target = total // step
    idx = np.linspace(0, total - 1, n_target).astype(np.int64)
    vecs = vecs[idx]
    print(f"sampled: {total} -> {len(vecs)} 帧 (step={step})")

    # 构建新索引
    if args.quant == "none":
        new_index = faiss.IndexFlatIP(dim)
        new_index.add(vecs)
    else:
        qt = faiss.ScalarQuantizer.QT_8bit if args.quant == "sq8" else faiss.ScalarQuantizer.QT_4bit
        new_index = faiss.IndexScalarQuantizer(dim, qt)
        new_index.train(vecs[: min(len(vecs), 20000)])
        new_index.add(vecs)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    faiss.write_index(new_index, args.output)
    out_size = os.path.getsize(args.output)
    print(f"output: {args.output}  ({out_size/1e6:.1f}MB, {new_index.ntotal} 帧)")
    print(f"压缩率: {in_size/out_size:.1f}x")

    # 自检：随机向量检索 top1 距离应正常
    rng = np.random.default_rng(0)
    q = rng.standard_normal((10, dim)).astype(np.float32)
    d, i = new_index.search(q, 1)
    print(f"self-check: top1 距离 mean={d.mean():.3f}（负值属正常，IP 检索取最大内积）")


if __name__ == "__main__":
    main()
