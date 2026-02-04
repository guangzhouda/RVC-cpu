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

## 模型转换：`*.pth` -> `synthesizer.onnx`（离线一次性）

说明：
- **运行时不需要 Python**；但把 `pth` 导出为 `onnx` 需要 Python（一次性离线操作）。
- realtime 程序里 `--syn` **必须是 `*.onnx`**，不能直接传 `*.pth`（否则会报 `Protobuf parsing failed`）。

导出命令（示例：DuaLive）：
```powershell
.\.codex\venv_onnx_export\Scripts\python.exe -c "from infer.modules.onnx.export import export_onnx; print(export_onnx(r'E:\RVC_models\DuaLive\DUALIVE.pth', r'E:\RVC_models\DuaLive\DUALIVE_synthesizer.onnx'))"
```

## 运行时目录说明（很重要）

`onnxruntime.dll` 在 **CUDA 版** 与 **DirectML 版**不是同一个 DLL，导出集合不同，不能放在同一目录里“随便切参数”：
- CUDA 版：`deps/onnxruntime/onnxruntime-win-x64-gpu-*`（配套 `onnxruntime_providers_cuda.dll`）
- DirectML 版：`Microsoft.ML.OnnxRuntime.DirectML`（只需要 `onnxruntime.dll` + `DirectML.dll`）

本仓库构建脚本会把两套运行时分别整理到：
- `build_rvc_sdk_ort/Release/`（CUDA 运行时）
- `build_rvc_sdk_ort/Release_dml/`（DirectML 运行时）

你用哪条 GPU 路线，就从对应目录运行 exe，并加 `--cuda` 或 `--dml`。

## 实时变声：运行示例（可直接复制）

```powershell
.\build_rvc_sdk_ort\Release\rvc_sdk_ort_realtime.exe `
    --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
    --syn "E:\RVC_models\YaeMiko\bachongshenzi_synthesizer.onnx" `
    --index "E:\RVC_models\YaeMiko\added_IVF256_Flat_nprobe_1_bachongshenzi_v2.index" `
    --rmvpe "E:\RVC_models\test-rvc-onnx\rmvpe.onnx" --rmvpe-threshold 0.03 `
    --cap-id 1 --pb-id 2 `
    --block-sec 0.5 --extra-sec 0.25 --crossfade-sec 0.05 `
    --prefill-blocks 2 `
    --index-rate 0.1 --up-key 12 --threads 8 `
    --rms-mix-rate 0.25 --noise-scale 0.2 --gate-rms 0.01 `
    --print-levels --print-latency --max-queue-sec 0.3
```

### 48k 模型注意事项（例如 DuaLive）

如果你的 `pth` 配置里 `config[-1]=48000`（模型采样率 48k），需要在运行时显式指定：
- `--model-sr 48000`

示例：
```powershell
.\build_rvc_sdk_ort\Release\rvc_sdk_ort_realtime.exe `
    --enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
    --syn "E:\RVC_models\DuaLive\DUALIVE_synthesizer.onnx" `
    --index "E:\RVC_models\DuaLive\added_IVF1817_Flat_nprobe_1_DUALIVE_v2.index" `
    --rmvpe "E:\RVC_models\test-rvc-onnx\rmvpe.onnx" --rmvpe-threshold 0.03 `
    --cap-id 1 --pb-id 2 `
    --io-sr 48000 --model-sr 48000 `
    --block-sec 0.5 --extra-sec 0.25 --crossfade-sec 0.05 `
    --index-rate 0.1 --up-key 12 --threads 8
```

## SDK 集成到其它软件（DLL，不需要附带 exe）

本仓库的 `*.exe` 只是演示/调参工具；真正集成时，你只需要把 **DLL + 头文件 + 运行时依赖** 带走，并在你的工程里调用 C ABI。

### 交付物打包

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_rvc_sdk_ort.ps1
```

产物目录（可直接拷走）：
- `sdk/rvc_sdk_ort/dist/win-x64-cuda/`（CPU+CUDA 版 ORT）
- `sdk/rvc_sdk_ort/dist/win-x64-dml/`（DirectML 版 ORT）

包里不含模型文件，你需要自己放：
- `content_encoder.onnx`
- `synthesizer.onnx`
- `added_*.index`
- 可选：`rmvpe.onnx`

### C/C++ 最小调用流程（示意）

```cpp
#include "rvc_sdk_ort.h"
#include <vector>

int main() {
  rvc_sdk_ort_config_t cfg{};
  cfg.io_sample_rate = 48000;
  cfg.model_sample_rate = 40000;     // 48k 模型请改成 48000
  cfg.block_time_sec = 0.5f;
  cfg.crossfade_sec = 0.05f;
  cfg.extra_sec = 0.25f;
  cfg.index_rate = 0.1f;
  cfg.sid = 0;
  cfg.f0_up_key = 12;
  cfg.vec_dim = 768;
  cfg.ep = RVC_SDK_ORT_EP_CPU;       // 或 RVC_SDK_ORT_EP_CUDA / RVC_SDK_ORT_EP_DML
  cfg.intra_op_num_threads = 8;
  cfg.f0_min_hz = 50.0f;
  cfg.f0_max_hz = 1100.0f;
  cfg.noise_scale = 0.2f;
  cfg.rms_mix_rate = 0.25f;
  cfg.f0_method = RVC_SDK_ORT_F0_RMVPE;
  cfg.rmvpe_threshold = 0.03f;

  rvc_sdk_ort_error_t err{};
  rvc_sdk_ort_handle_t h = rvc_sdk_ort_create(&cfg, &err);
  if (!h) return 1;

  if (rvc_sdk_ort_load(h, "vec-768-layer-12.onnx", "synthesizer.onnx", "added_xxx.index", &err) != 0) return 2;
  if (rvc_sdk_ort_load_rmvpe(h, "rmvpe.onnx", &err) != 0) return 3;

  const int bs = rvc_sdk_ort_get_block_size(h);
  std::vector<float> in(bs), out(bs);
  // TODO: 宿主侧采集/重采样到 io_sr、转单声道 f32[-1,1]，每次喂 bs 个采样点
  (void)rvc_sdk_ort_process_block(h, in.data(), bs, out.data(), bs, &err);
  // TODO: 宿主侧把 out 播放出去（或写文件）

  rvc_sdk_ort_destroy(h);
  return 0;
}
```

音频采集/播放与“推理不放进回调”的线程模型，请直接参考：`sdk/rvc_sdk_ort/demo_realtime/main.cpp`。

更详细的 C API 参数/约束/分发文件清单见：`sdk/rvc_sdk_ort/API.md`。
