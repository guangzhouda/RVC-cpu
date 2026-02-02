"""
onnx_quantize_dynamic.py
日期: 2026-02-02
执行者: Codex

用途：
  使用 onnxruntime.quantization 对 ONNX 模型做“动态量化”（weight-only int8）。
  - 优点：不需要校准数据，速度快，适合先把 CPU 性能拉起来验证路线。
  - 缺点：对卷积占比很高的模型加速有限；音质可能有变化（需自己听感验证）。

注意：
  - 该脚本仅用于离线生成量化模型；运行时（DLL/EXE）不需要 Python。
  - 建议使用与运行时一致的 onnxruntime 版本（本仓库 venv 已固定到 1.17.1）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def _size_mb(p: Path) -> float:
    return p.stat().st_size / (1024 * 1024)


def _guess_out_path(inp: Path, suffix: str) -> Path:
    return inp.with_suffix("").with_name(inp.stem + suffix + ".onnx")


def _load_opset(model_path: Path) -> int:
    m = onnx.load(str(model_path))
    # 取主域（ai.onnx）最大 opset
    ops = [i.version for i in m.opset_import if i.domain in ("", "ai.onnx")]
    return max(ops) if ops else -1


def quant_one(
    inp: Path,
    out: Path,
    op_types_to_quantize: Optional[List[str]],
    per_channel: bool,
    reduce_range: bool,
    use_external_data_format: bool,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

    # weight-only 动态量化：QInt8（更常见于 CPU）
    quantize_dynamic(
        model_input=str(inp),
        model_output=str(out),
        per_channel=per_channel,
        reduce_range=reduce_range,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=op_types_to_quantize,
        use_external_data_format=use_external_data_format,
    )

    # 快速检查：能否被 onnx checker 通过
    m = onnx.load(str(out))
    onnx.checker.check_model(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input ONNX path")
    ap.add_argument("--output", default="", help="Output ONNX path (default: <input>.int8.onnx)")
    ap.add_argument(
        "--ops",
        default="MatMul,Gemm",
        help="Comma-separated op types to quantize (default: MatMul,Gemm). Use empty to let ORT decide.",
    )
    ap.add_argument("--per-channel", action="store_true", help="Enable per-channel quantization")
    ap.add_argument("--reduce-range", action="store_true", help="Enable reduce-range (may help older CPUs)")
    ap.add_argument(
        "--external-data",
        action="store_true",
        help="Use external data format for large models (produces extra .data file)",
    )
    args = ap.parse_args()

    inp = Path(args.input).resolve()
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    out = Path(args.output).resolve() if args.output else _guess_out_path(inp, ".int8")

    ops = [s.strip() for s in args.ops.split(",") if s.strip()] if args.ops is not None else None
    if ops == []:
        ops = None

    print(f"[Input ] {inp} ({_size_mb(inp):.2f} MiB), opset={_load_opset(inp)}")
    print(f"[Output] {out}")
    print(f"[Config] weight_type=QInt8 per_channel={bool(args.per_channel)} reduce_range={bool(args.reduce_range)} ops={ops}")

    quant_one(
        inp=inp,
        out=out,
        op_types_to_quantize=ops,
        per_channel=bool(args.per_channel),
        reduce_range=bool(args.reduce_range),
        use_external_data_format=bool(args.external_data),
    )

    print(f"[OK] wrote: {out} ({_size_mb(out):.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

