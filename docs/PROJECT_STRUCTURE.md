# 项目地图（板块划分）

RVC（Retrieval-based Voice Conversion）改造版——按功能分板块，后端能力见下方矩阵。

## 目录结构

```
Retrieval-based-Voice-Conversion-WebUI/
├── apps/                  # 应用入口（从根目录执行：python apps\xxx.py）
│   ├── live_onnx.py       # 实时变声（ONNX，DML/CPU 双后端）—— 主力入口
│   ├── live_gui.py        # 实时变声 GUI（torch CUDA，PySimpleGUI + ASIO）
│   ├── live_llvc.py       # LLVC 实时变声（独立技术线，需 E:/Projects/LLVC）
│   ├── webui.py           # 训练+推理 WebUI（gradio，CUDA / --dml）
│   ├── app_offline.py     # 离线转换 gradio 应用
│   ├── api_server.py      # 推理 API 服务（FastAPI）
│   └── api_legacy_231006.py  # 旧版 API（归档，不维护）
├── train/                 # 训练板块
│   └── train_cli.py       # 30 分钟复刻 CLI（preprocess→extract→train→index）
├── convert/               # 离线转换板块
│   ├── offline_convert.py # 任意 wav → 变声 wav（与实时同引擎）
│   ├── infer_cli.py / infer_batch_rvc.py / onnx_inference_demo.py
├── eval/                  # 评估板块
│   ├── evaluate_quality.py    # PESQ/STOI/SI-SNR/LSD/melCos/RMVPE-f0
│   ├── asr_transcribe.py      # Whisper WER 可懂度终裁
│   ├── diag_blocks.py / compare_audio_metrics.py / calc_rvc_model_similarity.py
├── export/                # 模型导出与验证
│   ├── export_streaming_onnx.py + verify_streaming_onnx.py   # 流式 ONNX 导出/验证
│   ├── export_hubert_onnx.py / export_rmvpe_onnx.py / export_onnx.py
│   └── download_models.py / dlmodels.bat / dlmodels.sh
├── bench/                 # 性能基准与报告
│   ├── rvc_onnx_bench.py / rvc_cpu_bench.py / llvc_cpu_bench.py
│   └── run_realtime_eval_suite.py / summarize / render 报告
├── core/                  # infer/ 核心库（模型、训练、推理、UVR5 —— 内部引用密集，勿移动）
├── assets/                # 模型权重（hubert/rmvpe/pretrained_v2/weights/...）
├── configs/               # 运行配置（config.py、config.json、v1/v2 训练配置）
├── scripts/               # 启动脚本（按后端命名）
│   ├── live-dml.bat / live-cpu.bat / live-cuda.bat / live-llvc.bat
│   └── train-web.bat / train-cli.bat / api.bat
├── env/                   # 环境说明 + requirements 归档
├── docs/                  # 文档（SOP、项目地图）
├── logs/                  # 训练日志、实验、离线评估样本
├── tools/_wheels/         # 离线依赖 wheel 缓存
└── archive/               # 归档（notebook、旧脚本）
```

## 后端能力矩阵

| 功能 | 入口 | CPU | CUDA | DML |
|---|---|---|---|---|
| 实时变声 | apps/live_onnx.py | ✓ | — | ✓✓ 主力 |
| 实时 GUI | apps/live_gui.py | 慢 | ✓✓ | 官方支持 |
| LLVC 实时 | apps/live_llvc.py | ✓ | ✓ | — |
| 训练 | train/train_cli.py | 极慢 | ✓✓ | ✓（torch-directml 环境） |
| 离线转换 | convert/offline_convert.py | ✓ | ✓ | ✓ |
| WebUI 推理 | apps/webui.py | ✓ | ✓✓ | ✓（--dml） |
| API 服务 | apps/api_server.py | ✓ | ✓ | ✓ |
| 评估 | eval/evaluate_quality.py | ✓ | ✓ | ✓ |
| ONNX 导出/验证 | export/* | ✓ | — | — |

## 典型工作流

```bat
:: 1) 实时变声（DML，推荐）
scripts\live-dml.bat

:: 2) 训练自己的音色（30 分钟复刻）
scripts\train-cli.bat --data D:\干声 --exp 我的音色 --stage all --epochs 150

:: 3) 训练后导出实时模型
.venv\Scripts\python.exe export\export_streaming_onnx.py --model assets\weights\我的音色.pth --output logs\我的音色\model.stream150.onnx --block-time 0.15

:: 4) 质量评估
.venv-onnx-stream\Scripts\python.exe eval\evaluate_quality.py --ref 原声.wav --cand 变声.wav
```

## 环境对应

| 环境 | 后端 | 用途 |
|---|---|---|
| .venv | CUDA torch 2.4 | 训练 / WebUI / GUI / API / 导出 |
| .venv-onnx-stream | DirectML + CPU | 实时 ONNX / 离线转换 / 评估 |

详见 env/README.md
