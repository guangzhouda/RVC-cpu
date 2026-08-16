# -*- coding: utf-8 -*-
"""RVC 30分钟复刻 CLI 训练工具（对应大饼"30分钟音频复刻"流程）

流程 = 预处理切片 -> 提取音高(RMVPE) -> 提取特征(HuBERT) -> 训练 -> 构建检索索引

用法（仓库根目录）:
    .venv\Scripts\python.exe tools\train_cli.py --data "D:\干声目录" --exp 我的音色 --stage all
    .venv\Scripts\python.exe tools\train_cli.py --exp 我的音色 --stage train   # 断点续跑
    .venv\Scripts\python.exe tools\train_cli.py --exp 我的音色 --stage index  # 只重建索引

数据要求（大饼同款规范）:
    - 30 分钟以上目标音色干声（无伴奏/混响，单声道 wav 即可，任意采样率）
    - 建议切成 3~10 秒片段放同一目录（脚本也会自动切片）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PY = sys.executable


def run(cmd, cwd=ROOT):
    print("\n>>> " + " ".join(cmd))
    env = dict(os.environ)
    env["USE_LIBUV"] = "0"  # torch 2.4.0 Windows wheel 兼容（否则 DDP 初始化崩溃）
    p = subprocess.run(cmd, cwd=cwd, env=env)
    if p.returncode != 0:
        print(f"[train_cli] 步骤失败 exit={p.returncode}（cmd: {' '.join(cmd)}）")
        sys.exit(p.returncode)


def exp_dir(exp):
    return os.path.join(ROOT, "logs", exp)


def stage_preprocess(args):
    d = exp_dir(args.exp)
    os.makedirs(d, exist_ok=True)
    sr_num = {"32k": 32000, "40k": 40000, "48k": 48000}[args.sr]
    run([PY, os.path.join(ROOT, "infer/modules/train/preprocess.py"),
         os.path.abspath(args.data), str(sr_num), str(args.n_p), d, "True",
         "%.1f" % args.preprocess_per])


def stage_extract(args):
    d = exp_dir(args.exp)
    if args.f0 != "rmvpe_dml":
        run([PY, os.path.join(ROOT, "infer/modules/train/extract/extract_f0_print.py"),
             d, str(args.n_p), args.f0])
    else:
        run([PY, os.path.join(ROOT, "infer/modules/train/extract/extract_f0_rmvpe_dml.py"), d])
    run([PY, os.path.join(ROOT, "infer/modules/train/extract_feature_print.py"),
         "cuda:0", "1", "0", "0", d, args.version, "False"])


def stage_train(args):
    d = exp_dir(args.exp)
    # 1) 复制官方训练 config（不 import Config——其构造会解析 sys.argv 造成冲突）
    cfg_path = ("v1/%s.json" if args.version == "v1" else "v2/%s.json") % args.sr
    cfg_save = os.path.join(d, "config.json")
    if not os.path.exists(cfg_save):
        shutil.copy(os.path.join(ROOT, "configs", cfg_path), cfg_save)
    # 2) 生成 filelist（照 click_train 逻辑，含静音样本防爆音）
    gt = os.path.join(d, "0_gt_wavs")
    feat = os.path.join(d, "3_feature768" if args.version == "v2" else "3_feature256")
    f0d = os.path.join(d, "2a_f0")
    f0nsf = os.path.join(d, "2b-f0nsf")
    names = set(n.split(".")[0] for n in os.listdir(gt)) \
        & set(n.split(".")[0] for n in os.listdir(feat)) \
        & set(n.split(".")[0] for n in os.listdir(f0d)) \
        & set(n.split(".")[0] for n in os.listdir(f0nsf))
    opt = []
    for name in names:
        opt.append("%s\\%s.wav|%s\\%s.npy|%s\\%s.wav.npy|%s\\%s.wav.npy|%s" % (
            gt.replace("\\", "\\\\"), name,
            feat.replace("\\", "\\\\"), name,
            f0d.replace("\\", "\\\\"), name,
            f0nsf.replace("\\", "\\\\"), name, 0))
    fea_dim = 768 if args.version == "v2" else 256
    for _ in range(2):
        opt.append("%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                   % (ROOT.replace("\\", "\\\\"), args.sr,
                      ROOT.replace("\\", "\\\\"), fea_dim,
                      ROOT.replace("\\", "\\\\"),
                      ROOT.replace("\\", "\\\\"), 0))
    import random
    random.shuffle(opt)
    with open(os.path.join(d, "filelist.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(opt))
    print(f"[train_cli] filelist: {len(opt)} 条（含 2 条静音）")
    # 3) 训练
    cmd = [PY, os.path.join(ROOT, "infer/modules/train/train.py"),
           "-e", args.exp, "-sr", args.sr, "-f0", "1", "-bs", str(args.batch),
           "-g", "0", "-te", str(args.epochs), "-se", str(args.save_every),
           "-pg", os.path.join("assets/pretrained_v2", "f0G%s.pth" % args.sr),
           "-pd", os.path.join("assets/pretrained_v2", "f0D%s.pth" % args.sr),
           "-l", "1", "-c", "0", "-sw", "1", "-v", args.version]
    run(cmd)


def stage_index(args):
    d = exp_dir(args.exp)
    feat = os.path.join(d, "3_feature768" if args.version == "v2" else "3_feature256")
    if not os.path.exists(feat) or not os.listdir(feat):
        print("[train_cli] 请先运行 --stage extract")
        sys.exit(1)
    import faiss
    from sklearn.cluster import MiniBatchKMeans
    big_npy = np.concatenate([np.load(os.path.join(feat, n)) for n in sorted(os.listdir(feat))], 0)
    big_npy = big_npy[np.random.permutation(big_npy.shape[0])]
    if big_npy.shape[0] > 2e5:
        print("[train_cli] kmeans %s -> 10k centers" % big_npy.shape[0])
        big_npy = (MiniBatchKMeans(n_clusters=10000, verbose=False, batch_size=256 * args.n_p,
                                   compute_labels=False, init="random")
                   .fit(big_npy).cluster_centers_)
    np.save(os.path.join(d, "total_fea.npy"), big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    dim = 768 if args.version == "v2" else 256
    index = faiss.index_factory(dim, "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    for i in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[i:i + 8192])
    out = os.path.join(d, "added_IVF%s_Flat_nprobe_1_%s_%s.index" % (n_ivf, args.exp, args.version))
    faiss.write_index(index, out)
    print("[train_cli] 索引已生成:", out)


def main():
    ap = argparse.ArgumentParser(description="RVC 30分钟复刻 CLI 训练工具")
    ap.add_argument("--data", default="", help="干声目录（--stage preprocess/all 时必填）")
    ap.add_argument("--exp", required=True, help="实验名（模型名），输出在 logs/<exp>/")
    ap.add_argument("--stage", default="all", choices=["preprocess", "extract", "train", "index", "all"])
    ap.add_argument("--sr", default="48k", choices=["32k", "40k", "48k"],
                    help="训练采样率，默认 48k（与实时 ONNX 管线一致）")
    ap.add_argument("--f0", default="rmvpe", choices=["rmvpe", "fcpe", "crepe", "harvest", "rmvpe_dml"])
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    ap.add_argument("--batch", type=int, default=8, help="batch size（4060 8GB 建议 8~12）")
    ap.add_argument("--epochs", type=int, default=150, help="总 epoch（30分钟数据 100~200 之间）")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--n-p", type=int, default=4, help="预处理并行进程数")
    ap.add_argument("--preprocess-per", type=float, default=3.7, help="切片最短时长阈值(秒)")
    args = ap.parse_args()

    os.chdir(ROOT)
    stages = ["preprocess", "extract", "train", "index"] if args.stage == "all" else [args.stage]
    if "preprocess" in stages and not args.data:
        print("[train_cli] --data 未指定（干声目录必填）")
        sys.exit(1)
    for s in stages:
        print(f"\n========== stage: {s} ==========")
        {"preprocess": stage_preprocess, "extract": stage_extract,
         "train": stage_train, "index": stage_index}[s](args)
    print("\n[train_cli] 完成。权重: logs/%s/weights/G_xxx.pth，索引: logs/%s/added_*.index" % (args.exp, args.exp))
    print("导出实时模型:  .venv\\Scripts\\python.exe tools\\export_streaming_onnx.py --model logs\\%s\\weights\\G_xxx.pth --output logs\\%s\\model.stream.onnx --block-time 0.15" % (args.exp, args.exp))


if __name__ == "__main__":
    main()
