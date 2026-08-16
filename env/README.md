# 环境说明（venv 对应关系）

| 环境 | 后端 | 用途 | 入口 |
|---|---|---|---|
| `.venv` | CUDA（torch 2.4+cu124）+ CUDA EP | 训练、WebUI、实时 GUI、API、离线转换、导出 | train/、apps/webui.py、apps/live_gui.py、apps/api_server.py、export/ |
| `.venv-onnx-stream` | DirectML + CPU（onnxruntime-directml） | ONNX 实时变声（主力）、离线转换、评估 | apps/live_onnx.py（--dml-device 0）、convert/、eval/ |

> 注意：`.venv-onnx-stream` 通过 `rvc_base_venv.pth` 共享 `.venv` 的 site-packages，
> 两个环境都基于 Python 3.10.4。

## 依赖清单（env/ 目录）

- `requirements.txt` — 主环境（CUDA）基础依赖
- `requirements-dml.txt` / `environment_dml.yaml` — 官方 DML 训练环境（torch-directml，conda pydml）
- `requirements-onnx-realtime.txt` — ONNX 实时管线依赖
- `requirements-amd.txt` / `requirements-ipex.txt` — AMD / Intel 平台
- `requirements-py311.txt` / `requirements-win-for-realtime_vc_gui*.txt` — 兼容清单（历史）

## 离线安装缓存

`tools/_wheels/` 存有 PyPI 直链下载的依赖 wheel（pip SSL 被网络拦截时用）：
本地安装示例：`python -m pip install tools/_wheels/xxx.whl --no-deps --no-index`
