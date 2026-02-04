# rvc_sdk_ort（ORT + FAISS）Windows 实时变声 DLL

本目录是一个独立的 CMake 工程，用于把本仓库的 RVC 实时变声链路做成 **运行时无 Python** 的 DLL（C ABI）。

## 1. 产物

- `rvc_sdk_ort.dll`：SDK 动态库
- `rvc_sdk_ort_demo.exe`：最小 demo（加载模型/索引，跑几次空 block）

SDK 头文件：
- `sdk/rvc_sdk_ort/include/rvc_sdk_ort.h`

API 文档（集成其它软件时看这个）：
- `sdk/rvc_sdk_ort/API.md`

## 2. 运行时不需要 Python（但需要模型/索引文件）

运行时依赖：
- `rvc_sdk_ort.dll`
- `onnxruntime.dll`
  - CUDA EP：需要 GPU 版 onnxruntime（`onnxruntime-win-x64-gpu-*`）以及 `onnxruntime_providers_cuda.dll` / `onnxruntime_providers_shared.dll`
  - DirectML EP：需要 DirectML 版 onnxruntime（`Microsoft.ML.OnnxRuntime.DirectML` 的 `onnxruntime.dll`）以及系统/redistributable 的 `DirectML.dll`
- `faiss` 相关 DLL（如果你选择动态链接）
- 你的模型文件：`content_encoder.onnx`、`synthesizer.onnx`、`*.index`

> Python 只用于“离线导出 ONNX”（一次性），不属于运行时依赖。

## 3. 模型与索引准备

### 3.1 导出 synthesizer.onnx（从 *.pth）

本仓库已有导出脚本：`infer/modules/onnx/export.py`。

示例：
```powershell
python -c "from infer.modules.onnx.export import export_onnx; print(export_onnx('assets/weights/你的模型.pth','synthesizer.onnx'))"
```

### 3.2 content_encoder.onnx

本仓库未内置内容特征模型的 ONNX（例如常见的 `vec-768-layer-12.onnx`），你需要自行准备，并确保：
- 输出维度与 `vec_dim` 一致（256 或 768）
- 与你的 FAISS index 一致（index 的 d 必须等于 vec_dim）

### 3.3 FAISS index

SDK 会读取 `*.index` 并做 k=8 检索融合。请使用 `added_*.index`（与本仓库提示一致）。

## 4. 构建（CMake）

依赖项（建议预编译）：
- onnxruntime（建议 GPU 版：一套同时支持 CPU EP + CUDA EP）
- faiss（C++ 库）

本工程通过两个 CMake 变量查找依赖：
- `ONNXRUNTIME_ROOT`：包含 `include/` 与 `lib/`
- `FAISS_ROOT`：包含 `include/` 与 `lib/`

示例（MSVC）：
```powershell
cmake -S sdk/rvc_sdk_ort -B build_rvc_sdk_ort -G "Visual Studio 17 2022" -A x64 `
  -DONNXRUNTIME_ROOT="C:/deps/onnxruntime" `
  -DFAISS_ROOT="C:/deps/faiss"
cmake --build build_rvc_sdk_ort --config Release
```

### 4.1 使用本仓库已下载的 deps（推荐）

如果你按仓库根目录下的 `deps/` 目录准备了依赖（Codex 已在本机下载到以下路径）：
- `deps/onnxruntime/onnxruntime-win-x64-gpu-1.17.1`
- `deps/faiss`

那么直接这样构建：
```powershell
cmake -S sdk/rvc_sdk_ort -B build_rvc_sdk_ort -G "Visual Studio 17 2022" -A x64 `
  -DONNXRUNTIME_ROOT="deps/onnxruntime/onnxruntime-win-x64-gpu-1.17.1" `
  -DFAISS_ROOT="deps/faiss"
cmake --build build_rvc_sdk_ort --config Release
```

## 5. Demo

```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_demo.exe content_encoder.onnx synthesizer.onnx model.index
```

> Demo 默认配置使用 `io_sample_rate=48000`、`model_sample_rate=40000`、`vec_dim=768`、CPU EP。
> 需要和你的模型匹配，否则会加载失败或推理失败。

### 5.1 运行时 DLL 搜索路径提示

Windows 下运行 demo 时，至少需要在 `rvc_sdk_ort_demo.exe` 同目录（或 `PATH`）能找到：
- `onnxruntime.dll`（在 `ONNXRUNTIME_ROOT/lib/` 目录）
- `faiss.dll`（在 `FAISS_ROOT/bin/` 目录）
- `libblas.dll`、`liblapack.dll`（当前 conda-forge 的 `faiss.dll` 依赖这两个 DLL；缺失会导致 exe/DLL 无法启动）

如果你启用 CUDA EP，还需要：
- `onnxruntime_providers_cuda.dll`（在 `ONNXRUNTIME_ROOT/lib/` 目录）

### 5.2 一键打包（可选）

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_rvc_sdk_ort.ps1
```

产物会整理到：
- `sdk/rvc_sdk_ort/dist/win-x64/`

## 6. 实时变声 Demo（麦克风/系统音频）

可执行文件：
- `build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe`

先列出设备：
```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe --list-devices
```

用默认设备跑（按 Enter 退出）：
```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  --syn "E:\RVC_models\啊兰\alan_synthesizer.onnx" `
  --index "E:\RVC_models\啊兰\added_IVF1210_Flat_nprobe_1_alan_v2.index"
```

指定输入/输出设备（用 `--list-devices` 输出的 index）：
```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  --syn "E:\RVC_models\啊兰\alan_synthesizer.onnx" `
  --index "E:\RVC_models\啊兰\added_IVF1210_Flat_nprobe_1_alan_v2.index" `
  --cap-id 1 `
  --pb-id 1
```

启用 GPU EP：
- CUDA：在同目录有 CUDA 版 ORT DLL 时，加 `--cuda`。
- DirectML：建议使用 `build_rvc_sdk_ort/Release_dml/` 目录下的 exe，并加 `--dml`（因为 CUDA 版与 DML 版的 `onnxruntime.dll` 不能共存于同一目录）。

示例（DirectML）：
```powershell
build_rvc_sdk_ort/Release_dml/rvc_sdk_ort_realtime.exe --dml `
  --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  --syn "E:\RVC_models\啊兰\alan_synthesizer.onnx" `
  --index "E:\RVC_models\啊兰\added_IVF1210_Flat_nprobe_1_alan_v2.index"
```

提示：
- 若 `--cap-id` 选麦克风但 `cap_cb=0`，通常是 Windows 麦克风隐私权限未放行（需要允许桌面应用访问麦克风）。
- CPU-only 下如果跑不动，优先：
  - 使用“流式裁剪版” synthesizer.onnx（见下节 6.1）
  - 减小 `--extra-sec` / `--index-rate` / `--threads` 调参
  - 或启用 `--cuda` / `--dml`

另外，`rvc_sdk_ort_realtime.exe` 默认会先 `--prefill-blocks 2` 再启动播放，用于减少启动瞬间的 underflow（会增加一点点启动延迟）。

排查/低延时调参建议：
- 看实时倍率：`rt>1` 才能长期稳定实时；`rt≈1` 容易出现 underflow/延时漂移/“赫赫声”。
- 打印队列估算延时：加 `--print-latency`，观察 `in_q/out_q` 是否在持续增大（增大=跑不动或偶发卡顿）。
- 钳住延时上限（宁可丢一点音频也不要越跑越慢）：加 `--max-queue-sec 0.3`（示例值），输入积压过多会丢弃最旧音频并重置状态。

### 6.1 导出“流式裁剪版” synthesizer.onnx（推荐用于 CPU 实时）

本仓库新增导出脚本：`infer/modules/onnx/export_stream.py`。

它导出的是 `models.py::infer(skip_head, return_length)` 路径的 ONNX，特点是：
- 生成端在图内直接裁剪到 `return_length`，减少解码端的无效计算（CPU-only 提升更明显）
- **运行时仍不需要 Python**（Python 仅用于离线导出 ONNX）

重要限制：
- 由于 tracing + `.item()` 常量折叠，stream onnx 只适配固定的 `skip_head_frames/return_length_frames`（也就是你的 realtime `extra_sec/block_sec/crossfade_sec` 组合）。
- SDK 会在 `load()` 时自动探测 full/stream，并在 stream 情况下校验输出长度是否与当前配置一致；不一致会提示你重新导出。

示例：为 `--block-sec 0.5 --extra-sec 0.25 --crossfade-sec 0.05` 导出（对应 total_frames=80, return_frames=55, skip_head=25）：
```powershell
python -c "from infer.modules.onnx.export_stream import export_onnx_stream; print(export_onnx_stream('assets/weights/你的模型.pth','synthesizer_stream.onnx', skip_head_frames=25, return_length_frames=55))"
```

然后 realtime 直接把 `--syn` 指向 stream onnx（SDK 会自动识别）：
```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  --syn "E:\RVC_models\啊兰\alan_stream.onnx" `
  --index "E:\RVC_models\啊兰\added_IVF1210_Flat_nprobe_1_alan_v2.index" `
  --cap-id 2 --pb-id 1 `
  --block-sec 0.5 --extra-sec 0.25 --crossfade-sec 0.05 `
  --threads 8 --index-rate 0.1
```

### 6.2 离线 wav->wav 对比工具（定位“怪声”）

可执行文件：
- `build_rvc_sdk_ort/Release/rvc_sdk_ort_file.exe`

用途：把同一段输入音频（wav）离线跑一遍 RVC 链路并输出 wav，便于对比：
- **模型/索引是否本身就有伪影**（离线也有怪声）
- **实时链路参数是否导致伪影**（离线干净、实时不干净）
- 逐个 ablation：`--index-rate 0` / `--gate-rms` / `--rms-mix-rate` / `--noise-scale` / `--rmvpe`

示例（使用 stream onnx 时，必须与导出时的 block/extra/crossfade 匹配）：
```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_file.exe `
  --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  --syn "E:\RVC_models\YaeMiko\bachongshenzi_stream.onnx" `
  --index "E:\RVC_models\YaeMiko\added_IVF256_Flat_nprobe_1_bachongshenzi_v2.index" `
  --in  "E:\test\in.wav" `
  --out "E:\test\out.wav" `
  --block-sec 0.5 --extra-sec 0.25 --crossfade-sec 0.05 `
  --index-rate 0.1 --up-key 6 --threads 8
```

## 7. ONNX 量化（CPU 提速，离线一次性）

本仓库提供动态量化脚本（weight-only int8）：
- `scripts/onnx_quantize_dynamic.py`

示例：量化 content encoder（收益通常更明显）：
```powershell
.codex/venv_onnx_export/Scripts/python.exe scripts/onnx_quantize_dynamic.py `
  --input "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  --output "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.int8.onnx" `
  --per-channel
```

示例：量化 synthesizer（收益可能有限，因为卷积占比高）：
```powershell
.codex/venv_onnx_export/Scripts/python.exe scripts/onnx_quantize_dynamic.py `
  --input "E:\RVC_models\啊兰\alan_synthesizer.onnx" `
  --output "E:\RVC_models\啊兰\alan_synthesizer.int8.onnx" `
  --per-channel
```

然后实时程序直接换路径即可：
```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.int8.onnx" `
  --syn "E:\RVC_models\啊兰\alan_synthesizer.int8.onnx" `
  --index "E:\RVC_models\啊兰\added_IVF1210_Flat_nprobe_1_alan_v2.index" `
  --cap-id 2 --pb-id 1
```
