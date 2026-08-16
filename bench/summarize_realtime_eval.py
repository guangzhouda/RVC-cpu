"""
汇总 run_realtime_eval_suite 生成的 summary.json，输出均值/中位数报告。
"""
import argparse
import json
import math
import statistics


def mean(xs):
    return float(statistics.mean(xs)) if xs else None


def median(xs):
    return float(statistics.median(xs)) if xs else None


def summarize_engine(records, key):
    rows = [r[key] for r in records]
    return {
        "count": len(rows),
        "avg_ms_mean": mean([r["avg_ms"] for r in rows]),
        "avg_ms_median": median([r["avg_ms"] for r in rows]),
        "p95_ms_mean": mean([r["p95_ms"] for r in rows]),
        "rtf_mean": mean([r["rtf"] for r in rows]),
        "realtime_ok_count": int(sum(1 for r in rows if r["realtime_ok"])),
    }


def summarize_compare(records, key):
    rows = [r[key]["metrics"] for r in records]
    return {
        "count": len(rows),
        "wave_rmse_mean": mean([r["wave_rmse"] for r in rows]),
        "logmel_l1_mean": mean([r["logmel_l1"] for r in rows]),
        "mcd_mean": mean([r["mcd"] for r in rows if r["mcd"] is not None]),
        "f0_rmse_hz_mean": mean([r["f0_rmse_hz"] for r in rows if r["f0_rmse_hz"] is not None]),
        "f0_corr_mean": mean([r["f0_corr"] for r in rows if r["f0_corr"] is not None]),
        "si_sdr_db_mean": mean([r["si_sdr_db"] for r in rows]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        records = json.load(f)

    durations = [r["duration_sec"] for r in records]
    report = {
        "summary_path": args.summary,
        "num_samples": len(records),
        "duration_sec_mean": mean(durations),
        "duration_sec_median": median(durations),
        "rvc_gpu": summarize_engine(records, "rvc_gpu"),
        "rvc_cpu": summarize_engine(records, "rvc_cpu"),
        "llvc_cpu": summarize_engine(records, "llvc_cpu"),
        "compare_rvc_cpu_vs_gpu": summarize_compare(records, "compare_rvc_cpu_vs_gpu"),
        "compare_llvc_cpu_vs_gpu": summarize_compare(records, "compare_llvc_cpu_vs_gpu"),
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
