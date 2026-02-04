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
    --block-sec 0.25 --extra-sec 0.2 --crossfade-sec 0.05 `
    --prefill-blocks 1 `
    --index-rate 0.1 --up-key 6 --threads 8 `
    --rms-mix-rate 1 --noise-scale 0.2 `
    --vad-rms 0.02 --vad-floor 1 `
    --print-levels --print-latency --max-queue-sec 0.3
```

### 参数怎么选：它们分别在解决什么问题（按你遇到过的症状整理）

> 说明：上面命令的参数分两类：  
> - **SDK 推理参数**（会进入 `rvc_sdk_ort_config_t`）：影响音质/延时/性能。  
> - **demo 管线参数**（只在 `rvc_sdk_ort_realtime.exe` 里生效）：影响采集/播放/队列与调试输出。  

**先排除“根本没采到声音 / 播不出来”**

- `--cap-id / --pb-id`：修复“听不到声音 / 输入 RMS=0”  
  - 先用 `--list-devices` 选对麦克风和播放设备。  
  - 再用 `--passthrough --print-levels --prefill-blocks 0` 验证：`raw_rms` 应该随你说话明显变化；一直是 `0.0000` 基本就是设备/权限问题（Windows 麦克风隐私权限也要放行桌面应用）。
- `--print-levels`（debug）：修复“我不知道到底有没有信号进来/出去”  
  - `raw_*` 是“刚采到的原始电平”；`in_*` 是 gate/VAD 之后送进推理的电平；`out_*` 是最终播放电平。

**降低延时 / 防止延时越跑越大**

- `--block-sec`：基础延时与稳定性的核心旋钮  
  - 过小（例如 0.1）容易出现“赫赫/颗粒感/抖动”，本质通常是 **F0 抖动 + SOLA 拼接更频繁 + 偶发跑慢**。  
  - 你这台机器的甜点区是 `0.25`（`0.5` 更稳但延时更大）。
- `--extra-sec`：上下文 padding（稳定音色/减少断裂），也是“延时和算力”的主要来源之一  
  - 太小：更容易“口齿不清/颗粒感/拼接感”。  
  - 太大：延时上去、CPU 更吃紧。  
  - 你实测 `0.2` 是比较平衡的点。
- `--prefill-blocks`（demo）：修复“刚启动时噼啪/断续/underflow”  
  - 这是启动阶段的“先攒一点输出再开播”，会增加启动等待，但能明显减少开头爆音。你这里 `1` 合适。
- `--max-queue-sec`（demo）：修复“刚开始还好，过一会延时越来越长”  
  - 当推理偶发慢一拍，`in_q` 会越积越大；这个参数会在积压过大时 **丢弃最旧音频并 reset 状态**，把延时上限钳住（代价是偶发轻微丢帧感）。
- `--print-latency`（debug）：看你当前“软件队列延时”有没有在漂  
  - `[lat] est = in_q + out_q` 只是应用内部队列估算，不含声卡/系统额外缓冲；真实听感通常更大一点。

**解决“静音也在说话/喘声/环境声触发怪音”**

- `--vad-rms / --vad-floor`（demo）：轻量 RMS-VAD（逐 10ms 帧）  
  - 你遇到的现象：不开口也会有声音、或者有旁人说话/键盘把模型“触发”。  
  - `--vad-rms 0.02`：阈值，低于它的帧会被当成“无声帧”。  
  - `--vad-floor 1`：你这里等价于“**只用 VAD 判整块是否全静音**，不对你说话时的输入做衰减”，能避免把咬字削糊。  
  - 如果你想“更强抑制背景”，可以把 `vad-floor` 调到 `0.01~0.1`（会更干净，但更容易口齿不清，需要配合 hold/attack/release 细调；见 `sdk/rvc_sdk_ort/demo_realtime/main.cpp` 的参数说明）。
- `--rmvpe / --rmvpe-threshold`：修复“音色不稳/跑调/变声效果差”  
  - 你离线对比已经验证：加 RMVPE 后音色明显更对。  
  - 阈值偏低可能更容易被噪声误触发；遇到抖动可试 `--rmvpe-threshold 0.05`。

**解决“电流声 / 金属感 / 颗粒感”**

- `--noise-scale`：修复“电流声/金属感”  
  - 你已实测：`0.1` 会出现电流感，`0.2` 明显正常。  
  - 一般不建议把它压得太低。
- `--rms-mix-rate`：修复“喘声/音量不稳”或“颗粒感”  
  - 值越小越会把输出能量往输入拉（可减静音段喘声），但也可能带来“颗粒/抽动”。  
  - 你这条链路里 `1`（关闭混合）反而更自然。
- `--crossfade-sec`：修复“拼接感/爆点”  
  - 太小更容易有拼接颗粒；太大可能让清辅音变糊。你这里 `0.05` 是稳妥值。
- `--index-rate`：修复“奇怪的声音/不是自己的内容/检索引入伪影”  
  - 检索强度越大越“像目标音色”，但也越容易引入伪影。你实测 `0.1` 比较平衡。  
  - 排查杂音来源时可先设 `--index-rate 0`（完全关闭检索融合）做 A/B。

**其它常用参数**

- `--up-key`：修复“男女声转换不对/音高不匹配”（半音数；男->女常见 +6~+12）。
- `--threads`：修复“CPU 偶发卡顿/rt 接近 1 导致抖动”  
  - 线程太多也可能引起调度抖动；你机器上 `8` 可用，遇到抖动可试 `4` 做对比。

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

### 点击即用（把模型一起打包成“可双击运行”的文件夹）

说明：
- 由于模型/索引通常是几十~几百 MB，不建议“硬塞进 exe”；本仓库做的是 **portable 文件夹**：exe + dll + models/。
- `rvc_sdk_ort_realtime.exe` 已支持：不传 `--enc/--syn/--index` 时，自动读取同目录 `rvc_realtime.ini`（或 `models/` 固定命名文件）。

一键打包（示例：你当前验证最好的那套）：
```powershell
powershell -ExecutionPolicy Bypass -File scripts/package_rvc_realtime_portable.ps1 `
  -Runtime cuda `
  -Enc "E:\RVC_models\test-rvc-onnx\vec-768-layer-12.onnx" `
  -Syn "E:\RVC_models\YaeMiko\bachongshenzi_synthesizer.onnx" `
  -Index "E:\RVC_models\YaeMiko\added_IVF256_Flat_nprobe_1_bachongshenzi_v2.index" `
  -Rmvpe "E:\RVC_models\test-rvc-onnx\rmvpe.onnx" `
  -CapId 1 -PbId 2
```

产物目录：
- `dist_realtime_portable\win-x64-cuda\`

双击运行：
- `dist_realtime_portable\win-x64-cuda\rvc_sdk_ort_realtime.exe`

### C/C++ 最小调用流程（示意）

```cpp
#include "rvc_sdk_ort.h"
#include <vector>

int main() {
  rvc_sdk_ort_config_t cfg{};
  cfg.io_sample_rate = 48000;
  cfg.model_sample_rate = 40000;     // 48k 模型请改成 48000
  cfg.block_time_sec = 0.25f;
  cfg.crossfade_sec = 0.05f;
  cfg.extra_sec = 0.2f;
  cfg.index_rate = 0.1f;
  cfg.sid = 0;
  cfg.f0_up_key = 6;
  cfg.vec_dim = 768;
  cfg.ep = RVC_SDK_ORT_EP_CPU;       // 或 RVC_SDK_ORT_EP_CUDA / RVC_SDK_ORT_EP_DML
  cfg.intra_op_num_threads = 8;
  cfg.f0_min_hz = 50.0f;
  cfg.f0_max_hz = 1100.0f;
  cfg.noise_scale = 0.2f;
  cfg.rms_mix_rate = 1.0f;
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
