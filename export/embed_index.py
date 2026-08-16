# -*- coding: utf-8 -*-
"""把 faiss 检索索引蒸馏成"内嵌质心表"（随模型分发，替代外部 .index）

原理：index = 目标音色逐帧特征库（数万帧 x 768 维）。
k-means 聚成 K 个质心后，音色信息几乎不变（检索本来就是近似），
体积从 167MB -> K*768*4 字节（K=512 时约 1.5MB，fp16 减半）。

用法:
    .venv-onnx-stream/Scripts/python.exe export/embed_index.py --index assets/weights/xxx/added_xxx.index --output assets/weights/xxx/xxx.embed.npy --k 512
    （可选 --fp16 存半精度，体积再减半）

推理端: apps/live_onnx.py 加 --embedded-index xxx.embed.npy 即用内嵌表
（与 --index 二选一；内嵌表体积小、随模型分发，检索质量略低于全量 index）
"""
import argparse
import os

import numpy as np
import faiss

from sklearn.cluster import MiniBatchKMeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="原始 faiss .index")
    ap.add_argument("--output", required=True, help="输出 .npy 质心表")
    ap.add_argument("--k", type=int, default=512, help="质心数（256~2048，越大越接近原 index）")
    ap.add_argument("--fp16", action="store_true", help="存 fp16（体积减半）")
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    vecs = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    print(f"index: {index.ntotal} 帧, dim={index.d}")

    # k-means（质心数不超过帧数）
    k = min(args.k, vecs.shape[0])
    km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=1, max_iter=100, random_state=0)
    km.fit(vecs)
    centers = km.cluster_centers_.astype(np.float32)
    if args.fp16:
        centers = centers.astype(np.float16)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.save(args.output, centers)
    print(f"质心表: {centers.shape} ({os.path.getsize(args.output)/1e6:.2f}MB) -> {args.output}")

    # 自检：随机向量最近邻距离
    rng = np.random.default_rng(0)
    q = rng.standard_normal((10, index.d)).astype(np.float32)
    sim = q @ centers.T
    print(f"self-check: top1 相似度 mean={sim.max(1).mean():.2f}")


if __name__ == "__main__":
    main()
