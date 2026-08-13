"""
批量运行 realtime eval：
- RVC GPU baseline
- RVC CPU
- LLVC CPU
- 对两条 candidate 与 baseline 做客观比较
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

import soundfile as sf


def list_audio_files(root):
    root = pathlib.Path(root)
    files = []
    for ext in ("*.wav", "*.flac", "*.mp3", "*.m4a"):
        files.extend(root.rglob(ext))
    return sorted(files)


def duration_sec(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


def run_cmd(command, workdir):
    print("RUN:", " ".join(command))
    subprocess.run(command, cwd=workdir, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-root", default="data/eval/batch")
    parser.add_argument("--short-count", type=int, default=3)
    parser.add_argument("--min-sec", type=float, default=3.0)
    parser.add_argument("--max-sec", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--llvc-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    workdir = os.getcwd()
    files = []
    for path in list_audio_files(args.input_root):
        sec = duration_sec(path)
        if args.min_sec <= sec <= args.max_sec:
            files.append((path, sec))
        if len(files) >= args.short_count:
            break

    if not files:
        raise RuntimeError("No candidate files found in the requested duration range")

    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx, (path, sec) in enumerate(files):
        stem = path.stem
        sample_dir = out_root / f"{idx:02d}_{stem}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        rvc_gpu_wav = sample_dir / "rvc_gpu.wav"
        rvc_gpu_json = sample_dir / "rvc_gpu.json"
        rvc_cpu_wav = sample_dir / "rvc_cpu.wav"
        rvc_cpu_json = sample_dir / "rvc_cpu.json"
        llvc_cpu_wav = sample_dir / "llvc_cpu.wav"
        llvc_cpu_json = sample_dir / "llvc_cpu.json"
        cmp_rvc_json = sample_dir / "compare_rvc_cpu_vs_gpu.json"
        cmp_llvc_json = sample_dir / "compare_llvc_cpu_vs_gpu.json"

        common_rvc = [
            ".\\.venv\\Scripts\\python.exe",
            "tools\\rvc_cpu_bench.py",
            "--wav",
            str(path),
            "--max-blocks",
            "0",
            "--threads",
            str(args.threads),
            "--seed",
            str(args.seed),
        ]

        run_cmd(
            common_rvc
            + [
                "--device",
                "cuda",
                "--out",
                str(rvc_gpu_wav),
                "--report",
                str(rvc_gpu_json),
            ],
            workdir,
        )
        run_cmd(
            common_rvc
            + [
                "--device",
                "cpu",
                "--out",
                str(rvc_cpu_wav),
                "--report",
                str(rvc_cpu_json),
            ],
            workdir,
        )
        run_cmd(
            [
                ".\\.venv\\Scripts\\python.exe",
                "tools\\llvc_cpu_bench.py",
                "--wav",
                str(path),
                "--threads",
                str(args.llvc_threads),
                "--out",
                str(llvc_cpu_wav),
                "--report",
                str(llvc_cpu_json),
            ],
            workdir,
        )
        run_cmd(
            [
                ".\\.venv\\Scripts\\python.exe",
                "tools\\compare_audio_metrics.py",
                "--baseline",
                str(rvc_gpu_wav),
                "--candidate",
                str(rvc_cpu_wav),
                "--report",
                str(cmp_rvc_json),
            ],
            workdir,
        )
        run_cmd(
            [
                ".\\.venv\\Scripts\\python.exe",
                "tools\\compare_audio_metrics.py",
                "--baseline",
                str(rvc_gpu_wav),
                "--candidate",
                str(llvc_cpu_wav),
                "--report",
                str(cmp_llvc_json),
            ],
            workdir,
        )

        with open(rvc_gpu_json, "r", encoding="utf-8") as f:
            rvc_gpu = json.load(f)
        with open(rvc_cpu_json, "r", encoding="utf-8") as f:
            rvc_cpu = json.load(f)
        with open(llvc_cpu_json, "r", encoding="utf-8") as f:
            llvc_cpu = json.load(f)
        with open(cmp_rvc_json, "r", encoding="utf-8") as f:
            cmp_rvc = json.load(f)
        with open(cmp_llvc_json, "r", encoding="utf-8") as f:
            cmp_llvc = json.load(f)

        summary.append(
            {
                "input": str(path.resolve()),
                "duration_sec": sec,
                "rvc_gpu": rvc_gpu,
                "rvc_cpu": rvc_cpu,
                "llvc_cpu": llvc_cpu,
                "compare_rvc_cpu_vs_gpu": cmp_rvc,
                "compare_llvc_cpu_vs_gpu": cmp_llvc,
            }
        )

    summary_path = out_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
