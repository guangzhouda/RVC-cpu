# RVC Windows 实时变声 C++ SDK（DLL）落地方案（运行时无 Python）

日期：2026-01-31  
执行者：Codex  
项目：`Retrieval-based-Voice-Conversion-WebUI`（本仓库）

> 目标：把本仓库的实时变声能力做成“给别的程序用”的 `rvc_sdk.dll`（可选也产出 `.lib/.pdb`），要求：  
> 1) Windows 平台；2) 同时支持 GPU 与 CPU-only；3) 运行时必须去除 Python；4) 必须保留检索（FAISS）；5) 主要场景：实时变声。

---

## 0. 结论先行（推荐路线）

本仓库原生是 Python 推理（HuBERT + RVC net_g + F0 + FAISS）并带实时链路示例（`api_240604.py` + `tools/rvc_for_realtime.py`）。  
要做到“运行时无 Python”，**不要**尝试把整个项目“翻译成 C/C++”，而是做一个 **C++ 推理 SDK**，复用成熟运行时与库：

- 推理后端（推荐）：`libtorch`（TorchScript）+ `FAISS`（C++）  
  - 优点：最容易做到与当前实时路径一致（`net_g.infer(skip_head, return_length)` 这套流式接口在 TorchScript / C++ 侧可直接跑）。  
  - 缺点：体积大（libtorch 很重），CPU/GPU 通常要做 **两套 DLL**（一个 CPU build，一个 CUDA build）。

- 推理后端（备选）：`onnxruntime`（CPU + CUDA provider）+ `FAISS`（C++）  
  - 优点：**更容易做成单个 DLL 同时支持 CPU/GPU**（同一个 ORT GPU 包含 CPU EP），部署更“像 SDK”。  
  - 难点：你需要额外产出“实时流式可用”的 ONNX 图（目前仓库自带的 ONNX 导出 `infer/modules/onnx/export.py` 主要面向离线/整段推理，不包含实时 `skip_head/return_length` 这套裁剪逻辑）。

下面文档以 **TorchScript + libtorch（推荐）** 为主线，同时在关键点给出 ORT 的替代做法。

### 0.1 运行时是否还需要 Python？

不需要。

- 本方案里提到的 **`libtorch`** 是 “PyTorch 的 C++ 发行版/运行时”，不是 Python 包。  
- C++ SDK 的运行时依赖是：`rvc_sdk*.dll` +（libtorch/onnxruntime/faiss 等）动态库 +（模型/索引文件）。  
- 你可能会在“离线准备阶段”用 Python 把 `.pth/.pt` 导出成 TorchScript/ONNX，但这一步只做一次；**部署到用户机器后，不需要安装 Python**。

> 如果你看到文档/代码里出现 `librosa`、`fairseq` 等，那是指“仓库当前 Python 实现如何做”；在 C++ SDK 里应由 C++ 依赖替代（例如重采样用 `soxr`，RMS/窗函数/SOLA 用 C++ 实现）。

### 0.2 体积会不会很大？为什么？

会偏大，主要原因不是“Python”，而是你要同时支持 **CPU-only + GPU** 且要保留 **HuBERT + RVC + FAISS index**：

- 模型文件本身就很大：  
  - `assets/hubert/hubert_base.pt` 约 **180.7 MiB**（本仓库已有）  
  - `assets/rmvpe/rmvpe.pt` 约 **172.8 MiB**（本仓库已有，若你 CPU-only 不走 RMVPE 可不带）  
  - 你的音色模型 `assets/weights/*.pth`：大小取决于训练与版本（通常几十到数百 MiB）
- 检索索引（FAISS `.index`）大小与训练集/特征量强相关：从几十 MiB 到数 GiB 都可能（且你要求“必须保留检索”，所以这部分不可省）
- 运行时库体积：  
  - `libtorch`（CPU 版通常“数百 MiB级”，CUDA 版通常“GiB级”）  
  - `onnxruntime` 相对会小一些，但 GPU 版同样会引入 CUDA/cuDNN 依赖

因此，“运行时无 Python”并不等于“体积小”。要显著降体积，通常要动两类东西：  
1) **索引压缩（IVF-PQ/OPQ）**；2) **模型侧 FP16/量化/换更轻的 encoder**（但要接受质量/开发成本）。

---

## 1. 本仓库实时推理链路（作为对齐基准）

实时链路核心参考：

- 实时音频处理/缓冲/SOLA：`api_240604.py`（`AudioAPI.audio_callback`）  
- 实时 RVC 推理（含检索）：`tools/rvc_for_realtime.py`（`RVC.infer`）

实时每个 block 的数据流（简化版）：

1) 输入设备采样率音频 block（例如 48k）→ 维护环形缓冲（含 extra_time 前置上下文）  
2) 重采样到 16k（供内容特征模型/音高算法使用）  
3) 内容特征（HuBERT）提取：`model.extract_features(..., output_layer=9/12)`  
4) 检索（FAISS）融合：`index.search(k=8)` + `1/score^2` 加权融合（必须保留）  
5) 特征上采样 x2（20ms → 10ms 帧率对齐）  
6) F0 提取（rmvpe/harvest 等）并维护 pitch/pitchf 缓存  
7) 生成模型推理：`net_g.infer(..., skip_head, return_length)`（流式裁剪）  
8) SOLA 对齐 + crossfade（降低块边界伪影），输出到设备

你做 C++ SDK 时，建议“对齐这个链路”，否则实时效果/延迟/伪影会和当前差异很大。

---

## 2. 需要拆成哪些“可部署组件”

### 2.1 必须组件

1) **内容特征模型（Content Encoder）**  
   - 当前：fairseq HuBERT，权重：`assets/hubert/hubert_base.pt`（由 `infer/modules/vc/utils.py` / `tools/rvc_for_realtime.py` 加载）  
   - C++ 目标：TorchScript（推荐）或 ONNX（备选）

2) **RVC 生成模型（net_g / Synthesizer）**  
   - 当前：从 `assets/weights/*.pth` 加载（`infer/lib/jit/get_synthesizer.py`）  
   - 实时推理依赖其 `infer(skip_head, return_length)` 接口：`infer/lib/infer_pack/models.py` 中 `SynthesizerTrnMs*NSFsid.infer`  
   - C++ 目标：TorchScript（推荐）；ONNX 需要额外产出“实时可裁剪图”

3) **检索（FAISS）**（必须保留）  
   - 索引文件：`*.index`（通常是 `added_*.index`，仓库里明确提示不要用 `trained_*.index`）  
   - 运行时操作：`read_index` + `search` + `reconstruct_n` 得到 `big_npy`（并常驻内存）

4) **F0（音高）提取**  
   - 当前实时脚本支持：`rmvpe/crepe/harvest/pm/fcpe`  
   - C++ 目标（建议分层）：
     - CPU-only：用 WORLD（DIO/Harvest）C++ 实现（更现实）  
     - GPU：可选 RMVPE（TorchScript/ONNX）提升效果

5) **实时块处理（缓冲 + SOLA）**  
   - 需要状态：输入缓冲、输出缓冲、sola_buffer、fade_in/out 窗、pitch 缓存等  
   - 目标：SDK 内实现“吃一块 input → 吐一块 output”，让宿主程序易接入

### 2.2 可选组件（按需）

- 输入/输出噪声抑制（`TorchGate`）：`api_240604.py` 里是可选项；若做 SDK 可先不做，先把主链路稳定。  
- RMS 包络混合（`rms_mix_rate`）：可做成可选参数。  
- 立体声/设备枚举：建议让宿主做；SDK 只处理单声道 float PCM。

---

## 3. SDK 形态与接口契约（建议）

### 3.1 为什么用 C ABI（而不是 C++ 类导出）

DLL 给“别的程序”用时，**C ABI 最稳**（跨语言/跨编译器）。C++ ABI 在 MSVC/Clang/MinGW 之间不稳定。

### 3.2 建议导出接口（最小可用 + 可扩展）

建议约束：宿主每次喂固定 block（例如 10~40ms），SDK 内部完成缓冲与对齐，输出同采样率同长度 block。

```c
// rvc_sdk.h（示意）
typedef void* rvc_handle_t;

typedef enum {
  RVC_DEVICE_CPU = 0,
  RVC_DEVICE_CUDA = 1
} rvc_device_t;

typedef enum {
  RVC_F0_WORLD = 0,   // CPU-only 推荐
  RVC_F0_RMVPE = 1    // 可选（需要 rmvpe TorchScript/ONNX）
} rvc_f0_method_t;

typedef struct {
  rvc_device_t device;
  int sample_rate;          // 宿主 I/O 采样率，如 48000
  int channels;             // 宿主输出声道数（建议 1 或 2）
  float block_time_sec;     // 例如 0.25（参考 api_240604.py）
  float crossfade_sec;      // 例如 0.05
  float extra_sec;          // 例如 2.5
  int  enable_index;        // 必须为 1（本需求必须保留检索）
  float index_rate;         // 0~1
  int  protect;             // 若后续要做“保护清辅音”的逻辑才需要，否则可忽略
  rvc_f0_method_t f0_method;
} rvc_config_t;

typedef struct {
  int code;                 // 0 ok
  char message[512];
} rvc_error_t;

// 生命周期
__declspec(dllexport) rvc_handle_t rvc_create(const rvc_config_t* cfg, rvc_error_t* err);
__declspec(dllexport) void rvc_destroy(rvc_handle_t h);

// 加载模型/索引（建议初始化后调用一次）
__declspec(dllexport) int rvc_load_models(
  rvc_handle_t h,
  const char* hubert_ts_path,
  const char* synthesizer_ts_path,
  const char* index_path,
  rvc_error_t* err
);

// 实时处理：输入单声道 float32 PCM（[-1,1]），输出同长度
__declspec(dllexport) int rvc_process_block(
  rvc_handle_t h,
  const float* in_mono,
  int in_frames,
  float* out_interleaved,
  int out_frames,
  rvc_error_t* err
);
```

说明：
- `hubert_ts_path` / `synthesizer_ts_path` 选择 TorchScript（`.pt/.ts` 均可，本质是 torchscript 文件）。  
- 可以提供 `rvc_get_required_block_size()` 之类的接口，让宿主知道必须喂多少帧。  
- CPU-only 与 CUDA 可以做成两套 DLL（推荐）或一个 DLL 内部动态选择（实现更复杂）。

---

## 4. 模型导出与资产清单（离线准备阶段，可用 Python 做工具，但运行时不带 Python）

> 你要求“必须去除 Python”，我这里按行业惯例解释为：**运行时不依赖 Python**。  
> 离线导出（一次性）用 Python 把 `.pth/.pt` 转为 TorchScript/ONNX 是合理且常见的工程做法。

### 4.1 你最终要随 SDK 一起分发的文件（建议）

每个 voice 模型实例（一个音色）至少需要：

- `hubert.ts`：内容特征模型 TorchScript（CPU 用 fp32；GPU 用 fp16 可选）  
- `synthesizer.ts`：RVC 生成模型 TorchScript（CPU fp32；GPU fp16）  
- `model.index`：FAISS 索引（必须保留）  
- （可选）`rmvpe.ts`：RMVPE TorchScript（若你选择 RVC_F0_RMVPE）

### 4.2 从仓库现有权重导出 TorchScript（推荐做法）

仓库已有 JIT 导出工具：`infer/lib/jit/__init__.py`（`synthesizer_jit_export` / `rmvpe_jit_export` / `to_jit_model(hubert)`）。

下面给出“把导出的 bytes 直接写成 TorchScript 文件”的示例（你可在有 Python 环境的机器上执行一次）：

#### 4.2.1 导出 Synthesizer（RVC 生成网络）

CPU（fp32）：
```powershell
python - <<'PY'
import torch
from infer.lib import jit

pth = r"assets/weights/你的模型.pth"
cpt = jit.synthesizer_jit_export(pth, mode="script", device=torch.device("cpu"), is_half=False)
open("synthesizer.fp32.ts", "wb").write(cpt["model"])
print("ok:", "synthesizer.fp32.ts")
PY
```

GPU（fp16，CUDA 环境导出更稳）：
```powershell
python - <<'PY'
import torch
from infer.lib import jit

pth = r"assets/weights/你的模型.pth"
cpt = jit.synthesizer_jit_export(pth, mode="script", device=torch.device("cuda:0"), is_half=True)
open("synthesizer.fp16.ts", "wb").write(cpt["model"])
print("ok:", "synthesizer.fp16.ts")
PY
```

#### 4.2.2 导出 HuBERT（内容特征模型）

CPU（fp32）：
```powershell
python - <<'PY'
import torch
from infer.lib import jit

hubert_pt = r"assets/hubert/hubert_base.pt"
model, model_jit = jit.to_jit_model(hubert_pt, model_type="hubert", mode="script", device=torch.device("cpu"), is_half=False)
torch.jit.save(model_jit, "hubert.fp32.ts")
print("ok:", "hubert.fp32.ts")
PY
```

GPU（fp16）：
```powershell
python - <<'PY'
import torch
from infer.lib import jit

hubert_pt = r"assets/hubert/hubert_base.pt"
model, model_jit = jit.to_jit_model(hubert_pt, model_type="hubert", mode="script", device=torch.device("cuda:0"), is_half=True)
torch.jit.save(model_jit, "hubert.fp16.ts")
print("ok:", "hubert.fp16.ts")
PY
```

#### 4.2.3（可选）导出 RMVPE

CPU（fp32）：
```powershell
python - <<'PY'
import torch
from infer.lib import jit

rmvpe_pt = r"assets/rmvpe/rmvpe.pt"
cpt = jit.rmvpe_jit_export(rmvpe_pt, mode="script", device=torch.device("cpu"), is_half=False)
open("rmvpe.fp32.ts", "wb").write(cpt["model"])
print("ok:", "rmvpe.fp32.ts")
PY
```

GPU（fp16）：
```powershell
python - <<'PY'
import torch
from infer.lib import jit

rmvpe_pt = r"assets/rmvpe/rmvpe.pt"
cpt = jit.rmvpe_jit_export(rmvpe_pt, mode="script", device=torch.device("cuda:0"), is_half=True)
open("rmvpe.fp16.ts", "wb").write(cpt["model"])
print("ok:", "rmvpe.fp16.ts")
PY
```

> 备注：如果你选择“CPU-only 也必须 RMVPE”，那 CPU 实时压力会很大；一般建议 CPU-only 走 WORLD。

---

## 5. 检索（FAISS）在 C++ 侧如何保持与仓库一致

仓库实时检索逻辑（你必须对齐）在：`tools/rvc_for_realtime.py` 的 `infer()`，关键点：

- 只对 `skip_head // 2` 之后的内容特征做检索融合（跳过 extra 上下文）  
- `k=8`  
- 权重：`weight = (1 / score)^2`，再按行归一化  
- 用 `big_npy[ix]` 加权求和得到“检索特征”，再按 `index_rate` 和原特征线性混合

建议的 C++ 实现方式：

1) 初始化时一次性加载：
   - `faiss::read_index(index_path)`  
   - `index->reconstruct_n(0, index->ntotal, big_npy)`（常驻内存）

2) 每个 block 推理时：
   - 从 HuBERT 特征取出 `[skip_head/2 : end]` 的矩阵 `Q`（float32）  
   - `index->search(Q, k=8, distances, labels)`  
   - 按上面权重公式融合成 `R`  
   - `Q = index_rate * R + (1-index_rate) * Q`，再写回到特征张量

性能提示（实时很关键）：
- `big_npy` 内存占用可能很大；可以考虑：
  - 用 float16 存 `big_npy`（GPU 模式下更合适）  
  - 或把 index 改成 IVF-PQ/OPQ（见“量化/压缩”部分），减少常驻内存  
- 检索只做 CPU 也可以；GPU 主要用于神经网络推理。

---

## 6. 实时块处理：关键状态与参数（建议对齐 api_240604.py）

SDK 内建议直接复刻 `api_240604.py` 的实时参数计算，这样行为和现有 WebUI/Realtime API 一致：

- `zc = sample_rate / 100`（10ms 单位）  
- `block_frame`：对齐到 `zc` 的整数倍（由 `block_time_sec` 决定）  
- `block_frame_16k = 160 * block_frame / zc`（把 block 映射到 16k 的帧数）  
- `crossfade_frame`、`sola_buffer_frame`、`sola_search_frame`、`extra_frame` 同理  
- `skip_head = extra_frame / zc`  
- `return_length = (block_frame + sola_buffer_frame + sola_search_frame) / zc`

SOLA 实现要点（仓库逻辑）：
- 在输出 `infer_wav` 上做归一化相关搜索（conv1d 等价于滑动相关）找 `sola_offset`  
- 按 offset 裁剪后做淡入淡出 crossfade（或可选 phase vocoder）  
- 更新 `sola_buffer`

SDK 建议输出：
- 默认输出立体声/单声道都可，但建议由宿主决定；SDK 内部保持单声道，最后再复制到双声道。

---

## 7. C++ 工程依赖与构建（Windows）

### 7.1 推荐依赖（尽量标准化复用）

- `libtorch`（CPU 版 / CUDA 版）  
- `FAISS`（faiss-cpu，C++）  
- 重采样：`soxr`（或自实现简单线性重采样，但实时音质建议用成熟库）  
- FFT/卷积：可以用 Eigen/IPP/自实现（SOLA 相关开销不大）  
- 数学：Eigen  
- 日志：spdlog（可选）

### 7.2 构建策略（满足 CPU-only + GPU）

推荐产出两个 DLL，接口一致：

1) `rvc_sdk_cpu.dll`：链接 CPU 版 libtorch（不依赖 CUDA）  
2) `rvc_sdk_cuda.dll`：链接 CUDA 版 libtorch（需要匹配 CUDA/cuDNN 版本）

宿主程序策略：
- 先尝试加载 `rvc_sdk_cuda.dll`（若无 CUDA 环境则失败）  
- 失败则 fallback 到 `rvc_sdk_cpu.dll`

这样能最大化“可部署性”和“排障便利”，同时满足“支持 GPU + CPU-only”。

---

## 8. 量化/压缩路线（按实时优先级排序）

你要的是“实时变声”，优先级应是：**降低延迟 > 稳定性 > 体积**。

### 8.1 最有效且风险最低：FP16（GPU）

对 GPU 版本：
- HuBERT：导出 `hubert.fp16.ts`  
- Synthesizer：导出 `synthesizer.fp16.ts`  
收益：通常延迟明显下降，且音质风险低。

### 8.2 CPU-only 的现实建议：不要强推 INT8（先保证能实时）

CPU-only 实时最大的瓶颈通常是内容特征模型（HuBERT/ContentVec）。  
INT8 量化理论可加速，但工程成本高、算子兼容与音质风险不可控，建议分阶段：

阶段1：CPU fp32 能稳定跑到你目标延迟（哪怕质量略降、参数更保守）。  
阶段2：如果仍不够，再考虑：
- 把内容特征模型替换为更轻的 content encoder（ONNX + INT8）  
- 或只对部分层做动态量化（需要严格 AB 对比）

### 8.3 检索侧压缩（强烈建议做，收益大）

FAISS 侧可以做“索引量化/压缩”来降低内存与 cache miss：
- IVF（倒排）+ PQ（乘积量化）/ OPQ  
这通常对音色影响小于“把生成网 INT8”，但对内存占用改善很明显，利于实时稳定。

---

## 9. 验证与对齐（必须做，不然容易“能跑但不像 RVC”）

建议建立三类验证：

1) **离线一致性验证**（同一段 wav 输入）  
   - Python（`api_240604.py`/`tools/rvc_for_realtime.py` 路径）输出 vs C++ SDK 输出  
   - 对齐指标：波形能量/延迟/听感，允许少量差异但不能出现明显爆音、抖动

2) **实时稳定性验证**  
   - 连续运行 30~60 分钟  
   - 观察：延迟抖动、内存增长、是否出现卡顿/爆音

3) **性能剖析**  
   - 分段计时：内容特征 / 检索 / F0 / 生成网 / SOLA  
   - 对齐仓库打印：`tools/rvc_for_realtime.py` 会打印 `fea/index/f0/model` 耗时

---

## 10. 你下一步该做什么（最短路径）

1) 选定你要支持的 F0 方案：
   - CPU-only：WORLD  
   - GPU：RMVPE（可选）

2) 先用 Python 离线导出 TorchScript（见第 4 节），得到：
   - `hubert.fp32.ts` / `hubert.fp16.ts`
   - `synthesizer.fp32.ts` / `synthesizer.fp16.ts`
   - `model.index`

3) 搭一个最小 C++ POC（不是完整 SDK）：
   - 能加载 torchscript + faiss
   - 能对一段缓冲跑一轮 `tools/rvc_for_realtime.py:RVC.infer` 等价逻辑

4) 再把 POC 封装成 DLL（按第 3 节 C ABI）

---

## 附：本方案引用的仓库关键实现位置（便于你回查）

- 实时推理（含检索、F0 缓存、net_g.infer 裁剪）：`tools/rvc_for_realtime.py:347`  
- 实时音频块处理（缓冲/重采样/SOLA）：`api_240604.py:296`  
- 生成模型 `infer(skip_head, return_length)` 的实现：`infer/lib/infer_pack/models.py:746`  
- JIT 导出工具：`infer/lib/jit/__init__.py:1`、`infer/lib/jit/get_hubert.py:266`、`infer/lib/jit/get_synthesizer.py:1`
