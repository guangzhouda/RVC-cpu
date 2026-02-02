#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from typing import Iterable, List, Tuple

# ---- 在导入 huggingface_hub 之前设置环境变量，强制显示进度条 ----
os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")   # 显示 tqdm 进度条
# 如已安装 hf_transfer（pip install -U hf_transfer），打开下行可显著提速大文件
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import hf_hub_download

# 兼容不同版本的异常导入
try:
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
except Exception:
    try:
        from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError
    except Exception:
        class HfHubHTTPError(Exception): ...
        class LocalEntryNotFoundError(Exception): ...

REPO = "lj1995/VoiceConversionWebUI"
REV = "main"

# 优先使用环境变量指定的端点，其次镜像，最后官方
ENDPOINTS: List[str] = [
    (os.environ.get("HF_ENDPOINT", "").rstrip("/") or "https://hf-mirror.com"),
    "https://huggingface.co",
]

BASE_DIR = Path(__file__).resolve().parent.parent

DOWNLOADS: List[Tuple[str, Path]] = [
    ("hubert_base.pt", BASE_DIR / "assets/hubert"),
    ("rmvpe.pt", BASE_DIR / "assets/rmvpe"),
    ("uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
     BASE_DIR / "assets/uvr5_weights/onnx_dereverb_By_FoxJoy"),
]

_pretrained = [
    "D32k.pth","D40k.pth","D48k.pth",
    "G32k.pth","G40k.pth","G48k.pth",
    "f0D32k.pth","f0D40k.pth","f0D48k.pth",
    "f0G32k.pth","f0G40k.pth","f0G48k.pth",
]
DOWNLOADS += [(f"pretrained/{m}", BASE_DIR / "assets/pretrained") for m in _pretrained]
DOWNLOADS += [(f"pretrained_v2/{m}", BASE_DIR / "assets/pretrained_v2") for m in _pretrained]

_uvr5 = [
    "HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth",
    "HP2_all_vocals.pth",
    "HP3_all_vocals.pth",
    "HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth",
    "HP5_only_main_vocal.pth",
    "VR-DeEchoAggressive.pth",
    "VR-DeEchoDeReverb.pth",
    "VR-DeEchoNormal.pth",
]
DOWNLOADS += [(f"uvr5_weights/{m}", BASE_DIR / "assets/uvr5_weights") for m in _uvr5]

def try_download(filename: str, local_dir: Path, endpoints: Iterable[str], revision: str = "main") -> str:
    """
    逐个 endpoint 尝试下载；成功返回本地路径。
    新版 huggingface_hub 会自动断点续传与缓存，无需 resume_download/local_dir_use_symlinks。
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    last_err = None
    for i, ep in enumerate(endpoints):
        # 固定当前尝试的端点（镜像优先，失败回退）
        os.environ["HF_ENDPOINT"] = ep

        # 指数退避，降低限流/瞬断影响
        if i > 0:
            time.sleep(1.2 ** i)

        try:
            # 只指定 local_dir，由库处理进度条、断点续传与缓存
            path = hf_hub_download(
                repo_id=REPO,
                filename=filename,
                revision=revision,
                local_dir=str(local_dir),
                force_download=False,  # 如需强制重下设为 True
            )
            return path
        except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as e:
            last_err = e
            continue

    raise RuntimeError(f"All endpoints failed for {filename}: {last_err}")

def main():
    print("Starting downloads (stable mode, serialized)...", flush=True)
    print(f"Repo: {REPO} @ {REV}", flush=True)
    print("Endpoint priority:", " -> ".join(ENDPOINTS), flush=True)

    ok, fail = 0, 0
    for rel, out_dir in DOWNLOADS:
        print(f"\nDownloading {rel} ...", flush=True)
        try:
            local_path = try_download(rel, out_dir, ENDPOINTS, REV)
            size_mb = Path(local_path).stat().st_size / (1024 * 1024)
            print(f"[OK] {Path(rel).name} -> {local_path}  ({size_mb:.2f} MB)", flush=True)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {Path(rel).name}: {e}", flush=True)
            fail += 1

    print("\nSummary:", flush=True)
    print(f"  Success: {ok}", flush=True)
    print(f"  Failed : {fail}", flush=True)
    if fail == 0:
        print("All models downloaded!", flush=True)

if __name__ == "__main__":
    main()
