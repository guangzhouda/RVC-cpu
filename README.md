<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
一个基于 VITS 的简单易用的变声框架（RVC），已按功能板块与推理后端重组<br><br>

[![Licence](https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

</div>

> 底模使用接近 50 小时的开源高质量 VCTK 训练集训练，无版权顾虑。

## 项目简介

本仓库基于 [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)，核心特性：

+ 使用 top1 检索替换输入源特征为训练集特征来杜绝音色泄漏
+ 少量数据即可训练（推荐至少 10 分钟低底噪语音；30 分钟效果更佳）
+ 简单易用的网页界面 / 实时变声 / API 服务 / 离线转换
+ 使用 RMVPE 人声音高提取算法（InterSpeech2023）根绝哑音问题
+ 支持 CUDA / DirectML / CPU 三种推理后端（本机：RTX 4060 + AMD 780M）
+ 附带客观质量评估套件（PESQ / STOI / Whisper WER）与 30 分钟复刻训练 CLI

## 项目结构（板块化）

```
├── apps/      应用入口（实时变声 / GUI / WebUI / API）
├── train/     训练板块（30 分钟复刻 CLI + 官方索引工具）
├── convert/   离线转换板块
├── eval/      质量评估板块（PESQ/STOI/WER/诊断）
├── export/    模型导出与验证（流式 ONNX）
├── bench/     性能基准与报告
├── scripts/   启动脚本（按后端命名）
├── env/       环境说明 + requirements 归档
├── infer/     核心库（模型/训练/推理，勿移动）
├── assets/    预训练模型权重
├── docs/      文档（项目地图 / 大饼管线 SOP）
└── tools/     torchgate 降噪库 + 离线依赖 wheel 缓存
```

详见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)（含 CPU/CUDA/DML 后端能力矩阵）。

## 环境

| 环境 | 后端 | 用途 |
|---|---|---|
| .venv | CUDA（torch 2.4 + cu124） | 训练 / WebUI / 实时 GUI / API / 导出 |
| .venv-onnx-stream | DirectML + CPU | 实时变声（主力）/ 离线转换 / 评估 |

详见 [env/README.md](env/README.md)。

## 快速开始

### 1. 实时变声（DML，推荐）

双击 `scripts\live-dml.bat`，或手动：

```bat
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0
```

常用参数：

```bat
--pitch 12                 :: 男变女（大饼官方推荐 9~12）
--index-rate 0.3           :: 检索混合率（糊就降，不像就升）
--input-device 麦克风 --output-device 耳机   :: 设备选择（名称子串）
--list-devices             :: 查看设备列表
```

其他后端：`scripts\live-cpu.bat`（纯 CPU）、`scripts\live-cuda.bat`（CUDA GUI）、`scripts\live-llvc.bat`（LLVC）。

### 2. 训练自己的音色（30 分钟复刻）

```bat
.venv\Scripts\python.exe train\train_cli.py --data "D:\干声目录" --exp 我的音色 --stage all --epochs 150 --batch 8
```

完整流程（数据规范、分步执行、导出实时模型、接入管线）见
[docs/dabing_dml_pipeline_sop.md](docs/dabing_dml_pipeline_sop.md)。

### 3. WebUI / API / 评估

```bat
scripts\train-web.bat                     :: 训练+推理 WebUI（http://127.0.0.1:7897）
scripts\api.bat                           :: 推理 API 服务（FastAPI）
.venv-onnx-stream\Scripts\python.exe eval\evaluate_quality.py --ref 原声.wav --cand 变声.wav   :: 客观评估
```

## 预模型准备

所需预模型（hubert_base.pt / pretrained_v2 / uvr5_weights / rmvpe.pt / rmvpe.onnx）请从
[Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/) 下载，
或使用 `export\download_models.py`。ffmpeg 需安装或放置于根目录。

## 参考项目

+ [ContentVec](https://github.com/auspicious3000/contentvec/) / [VITS](https://github.com/jaywalnut310/vits) / [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio) / [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) / [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [RMVPE](https://github.com/Dream-High/RMVPE)（预训练模型由 [yxlllc](https://github.com/yxlllc/RMVPE) 与 [RVC-Boss](https://github.com/RVC-Boss) 训练测试）

## 协议

MIT License（见 [LICENSE](LICENSE) 与 MIT协议暨相关引用库协议）。
