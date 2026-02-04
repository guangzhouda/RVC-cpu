# rvc_sdk_ort C API 文档（Windows / 运行时无 Python）

日期：2026-02-04  
执行者：Codex  
仓库：`E:\Projects\RVC-cpu`  

本文档面向把 `rvc_sdk_ort.dll` 集成到其它软件的开发者，描述 **C ABI** 的接口契约、调用顺序、线程模型、以及运行时需要一起分发的文件清单。

> 提示：`rvc_sdk_ort_realtime.exe`/`rvc_sdk_ort_file.exe` 只是 demo/调参工具；你的产品集成通常只需要 `rvc_sdk_ort.dll` + `rvc_sdk_ort.h` + ORT/FAISS 依赖即可。

---

## 1. 运行时目标与边界

- 运行时不依赖 Python（SDK 内部用 ONNX Runtime 推理 + FAISS 检索）
- 输入/输出：**单声道 float32 PCM**，范围建议在 `[-1, 1]`
- 处理粒度：固定大小 block（由 SDK 计算得到）
- 实时推荐线程模型：**音频回调仅做 ring buffer**，推理放在工作线程（参考 `sdk/rvc_sdk_ort/demo_realtime/main.cpp`）

---

## 2. 关键约束（一定要遵守）

### 2.1 采样率约束

`io_sample_rate` 与 `model_sample_rate` 必须能被 100 整除（内部以 10ms 为单位对齐）。

- 常用：`io_sr=48000`
- 常用 v2 生成模型：`model_sr=40000`
- 有些模型是 48k（例如 `config[-1]=48000`）：此时需要 `--model-sr 48000`

### 2.2 block 大小

你不能随便喂任意帧数给 SDK。必须：

1) `bs = rvc_sdk_ort_get_block_size(h)`  
2) 每次调用 `rvc_sdk_ort_process_block()` 的 `in_frames/out_frames` 都等于 `bs`

### 2.3 线程安全

- **同一个 handle 不是线程安全的**：不要多线程并发调用 `rvc_sdk_ort_process_block()`。
- 建议：一个音频流 = 一个 handle；在同一工作线程串行调用。

---

## 3. C API 速览（函数列表）

头文件：`sdk/rvc_sdk_ort/include/rvc_sdk_ort.h`

生命周期：

1) `rvc_sdk_ort_create()`  
2) `rvc_sdk_ort_load()`（可选 `rvc_sdk_ort_load_rmvpe()`）  
3) `rvc_sdk_ort_get_block_size()`  
4) 循环 `rvc_sdk_ort_process_block()`  
5) 需要清空历史时调用 `rvc_sdk_ort_reset_state()`  
6) 结束时 `rvc_sdk_ort_destroy()`

---

## 4. 错误处理约定

所有返回 `int32_t` 的函数：

- 返回 `0`：成功
- 返回非 `0`：失败；同时 `rvc_sdk_ort_error_t`（可选）里会填 `code/message`

建议做法：

- 只要返回值非 0，立刻打印 `err.code/err.message`，并停止该条链路或降级。

---

## 5. 结构体说明：`rvc_sdk_ort_config_t`

重要字段解释（与 `rvc_sdk_ort.h` 注释一致）：

- `io_sample_rate`：宿主 I/O 的采样率（推理输出最终会重采样到该采样率并做 SOLA）
- `model_sample_rate`：生成模型输出采样率（模型本身的 sr）
- `block_time_sec / extra_sec / crossfade_sec`：滑窗推理的时间参数
  - `extra_sec` 越大越稳，但计算越重、延迟也更大
- `index_rate`：检索融合强度（0~1）
- `sid`：说话人 ID（多说话人模型才有意义）
- `f0_up_key`：变调（半音）
- `vec_dim`：内容特征维度（v2 通常 768；v1 通常 256）
- `ep`：ORT Execution Provider（CPU/CUDA/DML）
- `intra_op_num_threads`：CPU EP 的线程数
- `noise_scale`：合成噪声强度（越小越稳）
- `rms_mix_rate`：音量包络混合（越小越倾向于把输出 RMS 拉回输入 RMS，可减少静音段喘声/怪声）
- `f0_method / rmvpe_threshold`：F0 方式（RMVPE 通常更稳，需额外加载 rmvpe.onnx）

---

## 6. API 逐项说明

### 6.1 `rvc_sdk_ort_create`

```c
rvc_sdk_ort_handle_t rvc_sdk_ort_create(const rvc_sdk_ort_config_t* cfg, rvc_sdk_ort_error_t* err);
```

- 输入：`cfg` 不能为空
- 输出：成功返回 handle；失败返回 `NULL`（并在 `err` 里写错误）

### 6.2 `rvc_sdk_ort_destroy`

```c
void rvc_sdk_ort_destroy(rvc_sdk_ort_handle_t h);
```

- 释放 handle（允许传 `NULL`）

### 6.3 `rvc_sdk_ort_load`

```c
int32_t rvc_sdk_ort_load(
  rvc_sdk_ort_handle_t h,
  const char* content_encoder_onnx,
  const char* synthesizer_onnx,
  const char* faiss_index,
  rvc_sdk_ort_error_t* err
);
```

加载三件套：

- `content_encoder_onnx`：例如 `vec-768-layer-12.onnx`
- `synthesizer_onnx`：由 `*.pth` 离线导出（见根目录 `README.md`）
- `faiss_index`：例如 `added_*.index`

注意：`synthesizer_onnx` **必须是 onnx**，不能直接传 `*.pth`。

### 6.4 `rvc_sdk_ort_load_rmvpe`（可选）

```c
int32_t rvc_sdk_ort_load_rmvpe(rvc_sdk_ort_handle_t h, const char* rmvpe_onnx, rvc_sdk_ort_error_t* err);
```

- 仅当 `cfg.f0_method = RVC_SDK_ORT_F0_RMVPE` 时才会在推理中使用

### 6.5 `rvc_sdk_ort_get_block_size`

```c
int32_t rvc_sdk_ort_get_block_size(rvc_sdk_ort_handle_t h);
```

- 返回值：每次 `process_block` 必须喂入/产出多少个采样点（单位：`io_sample_rate` 下的 sample）

### 6.6 `rvc_sdk_ort_get_runtime_info`（可选）

```c
int32_t rvc_sdk_ort_get_runtime_info(rvc_sdk_ort_handle_t h, rvc_sdk_ort_runtime_info_t* out_info);
```

用于调试：

- `total_frames/return_frames/skip_head_frames`：10ms 帧数
- `synth_stream`：当前 synthesizer 是 full 还是 stream（导出方式不同）

### 6.7 `rvc_sdk_ort_reset_state`

```c
int32_t rvc_sdk_ort_reset_state(rvc_sdk_ort_handle_t h);
```

用途：清空历史缓冲/F0 cache/SOLA/rng 等。

建议在这些时机调用：

- “静音 -> 开始说话”边界（可减少残留伪影）
- 宿主发生丢包/输入队列大幅跳变（例如你做了 `max_queue` 丢弃策略）

### 6.8 `rvc_sdk_ort_process_block`

```c
int32_t rvc_sdk_ort_process_block(
  rvc_sdk_ort_handle_t h,
  const float* in_mono,
  int32_t in_frames,
  float* out_mono,
  int32_t out_frames,
  rvc_sdk_ort_error_t* err
);
```

输入：

- `in_mono`：float32 单声道 PCM（建议 `[-1,1]`）
- `in_frames`：必须等于 `rvc_sdk_ort_get_block_size(h)`

输出：

- `out_mono`：float32 单声道 PCM（SDK 内部会做 clamp 与 NaN/Inf 清理）
- `out_frames`：必须等于 block size

---

## 7. 集成分发：需要拷贝哪些文件到同一目录？

Windows 下最简单的做法：把运行时 DLL 都放到 **宿主 exe 同目录**（Windows loader 默认会从 exe 目录找 DLL）。

### 7.1 推荐：用打包脚本生成“可拷走目录”

```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_rvc_sdk_ort.ps1
```

产物：

- `sdk/rvc_sdk_ort/dist/win-x64-cuda/`
- `sdk/rvc_sdk_ort/dist/win-x64-dml/`

其中：

- `include/`：给编译期 include
- `lib/`：给编译期 link（`rvc_sdk_ort.lib`）
- `bin/`：**运行时需要与你的 exe 放同一目录**（或加入 PATH）

### 7.2 CPU-only（不启用 CUDA/DML）

把下面这些放到宿主 exe 同目录（最少集合）：

- `rvc_sdk_ort.dll`
- `onnxruntime.dll`
- `faiss.dll`
- `libblas.dll`
- `liblapack.dll`

> 另外：目标机需要安装 MSVC 运行库（Visual C++ Redistributable）。如果遇到缺少 `VCRUNTIME140*.dll/MSVCP140*.dll`，请安装对应运行库或随应用一起分发。

### 7.3 CUDA（启用 `cfg.ep=CUDA` 或 demo `--cuda`）

在 CPU-only 的基础上，还需要：

- `onnxruntime_providers_cuda.dll`
- `onnxruntime_providers_shared.dll`

以及 CUDA/cuDNN 相关依赖（通常通过安装 CUDA + cuDNN 来提供，或把对应 DLL 加到 PATH/同目录）。

### 7.4 DirectML（启用 `cfg.ep=DML` 或 demo `--dml`）

注意：DirectML 版与 CUDA 版的 `onnxruntime.dll` **不是同一个**，不能混放。

宿主 exe 同目录（最少集合）：

- `rvc_sdk_ort.dll`
- `onnxruntime.dll`（DirectML 版）
- `DirectML.dll`
- `faiss.dll`
- `libblas.dll`
- `liblapack.dll`

---

## 8. 模型文件建议放哪？

模型文件不强制与 exe 同目录，但为了部署简单，建议放到同目录或某个固定子目录，例如：

- `models/vec-768-layer-12.onnx`
- `models/synthesizer.onnx`
- `models/added_xxx.index`
- `models/rmvpe.onnx`（可选）

然后在 `rvc_sdk_ort_load()` 里用相对路径即可。

