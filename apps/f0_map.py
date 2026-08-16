"""CDF / quantile F0 mapping (pure numpy) — 移植自 Meloie (sstina/meloie, MIT).

输入侧音高校准：构建"说话人 f0 分布"到"目标模型 f0 分布"的单调映射（log2 Hz 域），
逐帧重映射后再喂解码器。分布级映射对齐中位数+展宽+形状，优于标量转置（+12）。

来源: https://github.com/sstina/meloie/blob/main/meloie/engine/f0_map.py (MIT)
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

F0_MIN_HZ = 50.0
F0_MAX_HZ = 1100.0
QUANTILE_COUNT = 48
MIN_VOICED_FRAMES = 200

_P = np.linspace(0.01, 0.99, QUANTILE_COUNT)


def _voiced_log2(f0_hz) -> np.ndarray:
    f0 = np.asarray(f0_hz, dtype=np.float64).reshape(-1)
    voiced = f0[(f0 >= F0_MIN_HZ) & (f0 <= F0_MAX_HZ)]
    return np.log2(voiced) if voiced.size else voiced


def build_quantiles(voice_f0_hz, target_f0_hz) -> Tuple[np.ndarray, np.ndarray]:
    """构建 (src_q, tgt_q) log2-Hz 分位数锚点；任一侧有声帧不足时抛 ValueError。"""
    lv = _voiced_log2(voice_f0_hz)
    lt = _voiced_log2(target_f0_hz)
    if lv.size < MIN_VOICED_FRAMES or lt.size < MIN_VOICED_FRAMES:
        raise ValueError(
            "CDF 映射需要每侧 >=200 有声帧（约2秒）："
            f"源 {lv.size} 帧 / 目标 {lt.size} 帧"
        )
    src_q = np.quantile(lv, _P).astype(np.float64)
    tgt_q = np.quantile(lt, _P).astype(np.float64)
    src_q = src_q + np.arange(src_q.size, dtype=np.float64) * 1e-6  # np.interp 要求严格递增
    return src_q, tgt_q


def make_remap(src_q, tgt_q) -> Callable[[np.ndarray], np.ndarray]:
    """返回 f0_remap(f0_hz) -> f0_hz 闭包；有声帧分位数映射，无声帧保持 0。"""
    src_q = np.asarray(src_q, dtype=np.float64)
    tgt_q = np.asarray(tgt_q, dtype=np.float64)

    def f0_remap(f0: np.ndarray) -> np.ndarray:
        f0 = np.asarray(f0)
        out = f0.astype(np.float64, copy=True)
        voiced = f0 > 0
        if np.any(voiced):
            out[voiced] = np.exp2(np.interp(np.log2(out[voiced]), src_q, tgt_q))
        return np.ascontiguousarray(out, dtype=f0.dtype)

    return f0_remap
