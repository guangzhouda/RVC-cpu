# RVC Windows 实时变声 ORT（onnxruntime）C++ SDK（DLL）方案（运行时无 Python）

日期：2026-01-31  
执行者：Codex  
项目：`Retrieval-based-Voice-Conversion-WebUI`（本仓库）

> 目标：面向“给别的程序调用”的 `rvc_sdk_ort.dll`，Windows 平台，**同时支持 GPU 与 CPU-only**，运行时**必须去除 Python**，**必须保留检索（FAISS）**，主要场景：实时变声。

---

## 1. 先回答你的问题：ORT 路线运行时需要 Python 吗？

不需要。

ORT 路线的运行时组成是：

- 你的 `rvc_sdk_ort.dll`
- `onnxruntime.dll` +（可选）CUDA provider DLL（同一套 ORT 可以同时支持 CPU EP 与 CUDA EP）
- `faiss` 相关 DLL（若你静态链接也可以不单独带）
- 模型/索引文件：`*.onnx` + `*.index`

**部署到用户机器时不需要安装 Python。**

> 但：你可能会在“离线准备阶段”用 Python 把 `.pth/.pt` 转成 `.onnx`（一次性工具链）；这不影响“运行时无 Python”。

---

## 2. 与 libtorch 路线相比：ORT 为什么更适合你要的“更小码”

如果只看“推理运行时库体积”：

- `libtorch` 通常非常大（CPU 版就不小，CUDA 版更大，且常常需要 CPU/GPU 两套分发）
- `onnxruntime` 通常显著更小，并且 **更容易做成单 DLL 同时支持 CPU/GPU**

但注意：最终包体最大的往往是：
- 你的 **FAISS index（`.index`）**（几十 MiB 到数 GiB 都可能）
- 内容特征模型（HuBERT/ContentVec）与生成网模型文件

---

## 3. 必须对齐的“实时基准链路”（来自本仓库）

本仓库实时推理核心逻辑（含检索）：

- `tools/rvc_for_realtime.py`：`RVC.infer()`（内容特征 → 检索融合 → F0 → 生成网 → 输出）
- `api_240604.py`：实时音频 block 缓冲 + 重采样 + SOLA 对齐与 crossfade

你做 SDK 时建议把这条链路作为“行为对齐基准”，否则实时延迟/边界伪影/音质会和当前差异很大。

---

## 4. ORT SDK 必须拆成哪些组件

### 4.1 内容特征模型（Content Encoder）

现状（Python）：HuBERT（fairseq）从 `assets/hubert/hubert_base.pt` 加载。  

ORT SDK 建议（C++）：使用 **ONNX 的内容特征模型**（ContentVec/Hubert ONNX 二选一）：

- 本仓库提供的 ONNX 推理示例类：`infer/lib/infer_pack/onnx_inference.py` 中 `ContentVec`，默认找 `pretrained/vec-768-layer-12.onnx`  
  - 注意：本仓库当前**未内置**该 ONNX 文件，你需要自行准备（离线下载/转换）。

### 4.2 检索（FAISS，必须保留）

对齐本仓库检索策略（`tools/rvc_for_realtime.py:RVC.infer`）：

- `k=8`
- `weight = (1/score)^2`，再归一化
- 仅对 `skip_head//2` 之后的帧进行检索融合
- 融合：`feats = index_rate*retrieved + (1-index_rate)*feats`

建议 C++ 初始化时一次性：
- `faiss::read_index(index_path)`
- `index->reconstruct_n(0, index->ntotal, big_npy)` 常驻内存，避免每 block 重建

### 4.3 F0（音高）提取

你要求 CPU-only + 实时，因此建议分两条：

- CPU-only：**WORLD（DIO/Harvest）C++**（最现实，速度/稳定性好）
- GPU：可选 RMVPE ONNX（效果更好，但要额外模型与工程）

### 4.4 生成模型（Synthesizer / net_g）

关键挑战：本仓库实时走的是 `net_g.infer(skip_head, return_length)`（见 `infer/lib/infer_pack/models.py` 的 `infer`）。  
而仓库自带 ONNX 导出（`infer/modules/onnx/export.py`）导出的是 `SynthesizerTrnMsNSFsidM.forward(...)`，更偏“离线整段生成”，并不直接暴露实时 `skip_head/return_length` 逻辑。

因此 ORT 路线建议分阶段：

#### 阶段 A（先跑通，可实时但可能更费算）

- ONNX 生成网只做“整段/窗口”推理
- 在 C++ SDK 里做滑窗：
  - 每次把 `extra_sec + block_sec + crossfade/search` 对应的窗口送进生成网
  - 生成后在 C++ 中把前面的 `skip_head` 对应音频裁掉，只取当前 block 对应的部分
  - 再交给 SOLA 做对齐

优点：最容易落地，运行时仍无 Python。  
缺点：计算量更大，CPU-only 可能难以达到低延迟；GPU 较可接受。

#### 阶段 B（优化版，真正对齐本仓库实时裁剪）

目标：导出/构建一个 ONNX 图，显式支持类似：

- 输入：`feats, pitch, pitchf, sid, skip_head, return_length`
- 输出：仅生成需要的片段（减少计算）

这需要对模型导出做定制（通常要改导出代码/模型结构，避免 `.item()`/Python 控制流阻碍 ONNX）。

---

## 5. SDK 接口契约（建议用 C ABI，跨语言最稳）

```c
typedef void* rvc_handle_t;

typedef enum { RVC_EP_CPU = 0, RVC_EP_CUDA = 1 } rvc_ep_t;

typedef enum { RVC_F0_WORLD = 0, RVC_F0_RMVPE = 1 } rvc_f0_method_t;

typedef struct {
  rvc_ep_t ep;                 // CPU 或 CUDA
  int sample_rate;             // 宿主 I/O 采样率（例如 48000）
  float block_time_sec;        // 例如 0.25
  float crossfade_sec;         // 例如 0.05
  float extra_sec;             // 例如 2.5
  float index_rate;            // 必须保留检索：0~1
  int sid;                     // 说话人 ID（通常 0）
  int f0_up_key;               // 音高变调（半音）
  rvc_f0_method_t f0_method;   // CPU-only 建议 WORLD
} rvc_config_t;

typedef struct { int code; char message[512]; } rvc_error_t;

__declspec(dllexport) rvc_handle_t rvc_create(const rvc_config_t* cfg, rvc_error_t* err);
__declspec(dllexport) void rvc_destroy(rvc_handle_t h);

__declspec(dllexport) int rvc_load(
  rvc_handle_t h,
  const char* content_encoder_onnx,
  const char* synthesizer_onnx,
  const char* faiss_index,
  const char* rmvpe_onnx,           // 可选（WORLD 模式可传 null）
  rvc_error_t* err
);

// 输入：单声道 float32 PCM（[-1,1]），长度为 block 对齐值；输出同长度（可由宿主复制成双声道）。
__declspec(dllexport) int rvc_process_block(
  rvc_handle_t h,
  const float* in_mono,
  int in_frames,
  float* out_mono,
  int out_frames,
  rvc_error_t* err
);
```

实现要点：
- `rvc_create` 内初始化环形缓冲、fade 窗、SOLA buffer、重采样器状态
- `rvc_process_block` 内执行“输入更新→重采样→encoder→faiss→f0→synthesizer→SOLA→输出”

---

## 6. ORT 运行时选择 CPU/GPU（同一套 DLL）

建议逻辑：

- 如果用户选择 `RVC_EP_CUDA`：
  - 创建 ORT Session 时优先启用 CUDA EP（并保留 CPU EP 作为 fallback）
- 如果用户选择 `RVC_EP_CPU`：
  - 只启用 CPU EP

这样你只需要分发一套 `rvc_sdk_ort.dll`，并随包携带 ORT 的 CUDA provider（以及其依赖）。

---

## 7. 资产清单（你最终要随 SDK 分发什么）

最小可用（建议）：

- `rvc_sdk_ort.dll`
- `onnxruntime.dll`
- `onnxruntime_providers_cuda.dll`（如果要 GPU）
- （可能还需要）ORT CUDA provider 依赖的 NVIDIA DLL（视你打包策略）
- `faiss` DLL（或静态链接）
- `content_encoder.onnx`（你选择的内容特征模型）
- `synthesizer.onnx`（生成网 ONNX）
- `model.index`（FAISS index，必须）

可选：
- `rmvpe.onnx`（如果 GPU 路线要 RMVPE）

---

## 8. 你现在就能做的“最短落地路径”（建议顺序）

1) 先确定内容特征模型用哪个 ONNX：
   - 优先：ContentVec/Hubert ONNX（便于 ORT）  
   - 不建议第一版就“自己把 fairseq hubert_base.pt 转 ONNX”，成本高且踩坑多

2) 先落地“阶段 A 滑窗+裁剪”的生成网 ONNX（保证可运行）

3) 把 FAISS 检索按 `tools/rvc_for_realtime.py` 逻辑在 C++ 侧复刻（必须常驻 index）

4) CPU-only 的 F0 先用 WORLD（DIO/Harvest）C++ 实现

5) 把 `api_240604.py` 的 SOLA/缓冲参数照抄到 C++（避免边界伪影）

6) 再做性能优化：阶段 B 的实时裁剪 ONNX、index 压缩（IVF-PQ/OPQ）、FP16 等

---

## 9. 关键仓库参考位置（便于你对照实现）

- 实时推理（含检索）：`tools/rvc_for_realtime.py:347`  
- 实时音频 block + SOLA：`api_240604.py:296`  
- 生成网实时裁剪接口：`infer/lib/infer_pack/models.py:746`  
- 现有 ONNX 导出脚本：`infer/modules/onnx/export.py:6`  
- 现有 ONNX 推理示例（ContentVec + OnnxRVC）：`infer/lib/infer_pack/onnx_inference.py:64`

