"""
根据 aggregate.json 生成可直接阅读的 Markdown 报告。
"""
import argparse
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, digits=2):
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}"


def engine_row(label, block_or_chunk, data):
    return (
        f"| {label} | {block_or_chunk} | {data['count']} | {fmt(data['avg_ms_mean'])} | "
        f"{fmt(data['p95_ms_mean'])} | {fmt(data['rtf_mean'])} | {data['realtime_ok_count']} |"
    )


def compare_row(label, data):
    return (
        f"| {label} | {data['count']} | {fmt(data['wave_rmse_mean'],4)} | "
        f"{fmt(data['logmel_l1_mean'],4)} | {fmt(data['mcd_mean'],2)} | "
        f"{fmt(data['f0_rmse_hz_mean'],2)} | {fmt(data['f0_corr_mean'],3)} | {fmt(data['si_sdr_db_mean'],2)} |"
    )


def build_section(title, aggregate, budget_note):
    lines = []
    lines.append(f"## {title}")
    lines.append("")
    lines.append(
        f"样本数: `{aggregate['num_samples']}`，平均时长: `{fmt(aggregate['duration_sec_mean'])}s`，"
        f"中位时长: `{fmt(aggregate['duration_sec_median'])}s`。"
    )
    lines.append("")
    lines.append("### 实时性能")
    lines.append("")
    lines.append("| Engine | Budget | N | Avg ms | P95 ms | RTF | Realtime OK Count |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    lines.append(engine_row("RVC GPU baseline", budget_note["rvc_gpu"], aggregate["rvc_gpu"]))
    lines.append(engine_row("RVC CPU", budget_note["rvc_cpu"], aggregate["rvc_cpu"]))
    lines.append(engine_row("LLVC CPU", budget_note["llvc_cpu"], aggregate["llvc_cpu"]))
    lines.append("")
    lines.append("### 相对 RVC GPU baseline 的客观指标")
    lines.append("")
    lines.append("| Candidate | N | Wave RMSE | Log-mel L1 | MCD | F0 RMSE (Hz) | F0 Corr | SI-SDR (dB) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(compare_row("RVC CPU", aggregate["compare_rvc_cpu_vs_gpu"]))
    lines.append(compare_row("LLVC CPU", aggregate["compare_llvc_cpu_vs_gpu"]))
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--short-aggregate", required=True)
    parser.add_argument("--long-aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    short = load_json(args.short_aggregate)
    long = load_json(args.long_aggregate)

    lines = []
    lines.append("# Realtime Eval Report")
    lines.append("")
    lines.append("本报告比较三条链路：")
    lines.append("- `RVC PyTorch GPU` 作为 baseline")
    lines.append("- `RVC PyTorch CPU`")
    lines.append("- `LLVC CPU streaming`")
    lines.append("")
    lines.append("客观指标统一以 `RVC GPU baseline` 输出为参考。")
    lines.append("")
    lines.append(build_section("短句集", short, {"rvc_gpu": "250ms block", "rvc_cpu": "250ms block", "llvc_cpu": "26ms chunk"}))
    lines.append(build_section("长句集", long, {"rvc_gpu": "250ms block", "rvc_cpu": "250ms block", "llvc_cpu": "26ms chunk"}))
    lines.append("## 已知限制")
    lines.append("")
    lines.append("- 当前输入集来自 `LibriSpeech dev-clean-2`，为英文干净语音，不代表中文场景。")
    lines.append("- 本机上 `RVC CPU` 的 HuBERT 多线程会产出非有限值，因此 benchmark 自动回退到 `threads=1`。")
    lines.append("- `LLVC` 与 `RVC` 不是同一目标模型架构，因此客观指标仅表示“接近 RVC GPU baseline 的程度”，不是同架构等价替换证明。")
    lines.append("- `MCD` 这里是基于直接对齐后的 MFCC 均值距离，适合做相对比较，不适合和其他论文数值横向对齐。")
    lines.append("")

    content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(content)
    print(args.output)


if __name__ == "__main__":
    main()
