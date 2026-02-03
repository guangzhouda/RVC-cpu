# RVC-cpu

本仓库是 `Retrieval-based-Voice-Conversion-WebUI` 的一个面向 **Windows / 运行时无 Python** 的分支，目标是提供可嵌入其它程序的实时变声 SDK：
- **C/C++ DLL（C ABI）**
- **ONNX Runtime 推理**
- **FAISS 检索保留（RVC 索引融合）**
- 支持 **CPU-only** 与 **GPU**（CUDA / DirectML 二选一运行时）

## 快速入口

- SDK 工程与构建/运行说明：`sdk/rvc_sdk_ort/README.md`
- 一键构建（含 staging）：`scripts/build_rvc_sdk_ort.ps1`
- 下载/准备 DirectML 运行时：`scripts/setup_onnxruntime_directml.ps1`

## 运行时目录说明（很重要）

`onnxruntime.dll` 在 **CUDA 版** 与 **DirectML 版**不是同一个 DLL，导出集合不同，不能放在同一目录里“随便切参数”：
- CUDA 版：`deps/onnxruntime/onnxruntime-win-x64-gpu-*`（配套 `onnxruntime_providers_cuda.dll`）
- DirectML 版：`Microsoft.ML.OnnxRuntime.DirectML`（只需要 `onnxruntime.dll` + `DirectML.dll`）

本仓库构建脚本会把两套运行时分别整理到：
- `build_rvc_sdk_ort/Release/`（CUDA 运行时）
- `build_rvc_sdk_ort/Release_dml/`（DirectML 运行时）

你用哪条 GPU 路线，就从对应目录运行 exe，并加 `--cuda` 或 `--dml`。

