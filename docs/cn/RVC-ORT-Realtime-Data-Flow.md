# RVC ORT C++ 实时变声：数据路由图与使用方法

日期：2026-02-03  
执行者：Codex  
仓库：`E:\Projects\RVC-cpu`  

本文档描述本仓库 `sdk/rvc_sdk_ort/` 当前实现的 **Windows 实时变声**（ONNX Runtime + FAISS）数据流、关键参数含义与排查入口。  
目标用途：当你听到“赫赫/电音/静音也出声”等伪影时，用此文档快速定位是哪一段链路产生了问题。

---

## 1. Realtime Demo（`rvc_sdk_ort_realtime.exe`）数据路由图

入口实现：`sdk/rvc_sdk_ort/demo_realtime/main.cpp`

```
[WASAPI Capture 设备 @ io_sr, f32, mono]
        |
        v
CaptureCallback() 仅写 ring buffer
sdk/rvc_sdk_ort/demo_realtime/main.cpp:270
        |
        v
in_rb (ma_pcm_rb)
        |
        v
WorkerLoop() 推理线程（不在音频回调里跑模型）
sdk/rvc_sdk_ort/demo_realtime/main.cpp:306
        |
        +--> (可选) --gate-rms：整 block RMS < gate -> 直接输出 0
        |               静音->有声边界调用 rvc_sdk_ort_reset_state()
        |
        +--> (可选) --passthrough：不跑模型，直接播输入（调试 I/O）
        |
        +--> 否则：rvc_sdk_ort_process_block()
        |        -> SDK 内部完整推理+SOLA
        v
out_rb (ma_pcm_rb)
        |
        v
PlaybackCallback() 从 out_rb 读，不足则补 0（underflow）
sdk/rvc_sdk_ort/demo_realtime/main.cpp:279
        |
        v
[WASAPI Playback 设备 @ io_sr, f32, mono]
```

### 1.1 日志字段含义（你看到的 `[perf]` / `[lvl]`）

- `[perf] avg process_block=... (block=..., rt=...x)`：
  - `avg process_block`：推理线程处理一个 block 的平均耗时（ms）
  - `block`：该 block 在音频时间上的长度（ms）
  - `rt = block_ms / avg_ms`：实时倍率（**>1 表示能跑过实时**；例如 5x 表示比实时快 5 倍）
- `[lvl] in_rms/in_peak/out_rms/out_peak`：输入/输出电平，用于确认“到底有没有采到声音/到底有没有输出”。

---

## 2. SDK 内部（`rvc_sdk_ort_process_block`）数据路由图

调用入口：
- `sdk/rvc_sdk_ort/src/rvc_sdk_ort.cpp:110`（C ABI）
- `sdk/rvc_sdk_ort/src/rvc_engine.cpp:923`（`RvcEngine::ProcessBlock`）

### 2.1 关键长度/对齐（10ms frame 体系）

在 `RvcEngine::InitPlan_()`（`sdk/rvc_sdk_ort/src/rvc_engine.cpp:156`）里，时间单位统一为 **10ms 一帧**：

- `zc_io = io_sr / 100`，`zc_model = model_sr / 100`
- `block_frames = round(block_sec * 100)`
- `extra_frames = round(extra_sec * 100)`
- `crossfade_frames = round(crossfade_sec * 100)`
- `sola_buffer_frames = min(crossfade_frames, 4)`（最多 40ms）
- `sola_search_frames = 1`（固定 10ms）
- `total_frames = extra + block + sola_buffer + sola_search`
- `return_frames = block + sola_buffer + sola_search`
- `skip_head_frames = extra`

最终：

- **宿主每次喂给 SDK 的采样点数**：`block_size_io = block_frames * zc_io`
- **一次推理窗口的 16k 输入长度**：`total_size_16k = total_frames * 160`
- **一次推理输出到 io_sr 的波形长度**：`return_frames * zc_io`

### 2.2 推理主链路（从输入 block 到输出 block）

```
in_mono[block_size_io] @ io_sr
   |
   v
UpdateInputBuffers_()
- input_io_  : [total_frames * zc_io]  (io_sr)
- input_16k_ : [total_frames * 160]    (16k)
sdk/rvc_sdk_ort/src/rvc_engine.cpp:469
   |
   v
InferWindow_()
sdk/rvc_sdk_ort/src/rvc_engine.cpp:794
   |
   +--> ExtractAndRetrieve_()  (内容特征 + 检索)
   |    sdk/rvc_sdk_ort/src/rvc_engine.cpp:620
   |    - encoder.onnx 输入: [1,1,T16]  (T16 = total_frames*160)
   |    - encoder.onnx 输出: 归一化成 feats20:[T20,C]
   |    - feats20 末尾 cat(last) -> T20+1
   |    - FAISS search(k=8) + (1/d)^2 加权 -> blend(index_rate)
   |    - 上采样到 10ms：当前实现是“2x repeat”（不是线性插值）
   |      -> feats10:[p_len,C]  (p_len = total_frames)
   |
   +--> ComputeF0WithCache_()  (F0)
   |    sdk/rvc_sdk_ort/src/rvc_engine.cpp:533
   |    - 默认：16k 上做 YIN，hop=160(10ms)，带 cache 更新（对齐 tools/rvc_for_realtime.py）
   |    - 可选：RMVPE（需加载 rmvpe.onnx），通常比 YIN 更稳（短 block 下更重要）
   |    - 输出 pitch[int64 T] / pitchf[float T]，T=p_len
   |
   +--> synthesizer.onnx
        sdk/rvc_sdk_ort/src/rvc_engine.cpp:801
        输入：
        - phone     [1,T,C]  (T=p_len, C=vec_dim)
        - lengths   [1]
        - pitch     [1,T]
        - pitchf    [1,T]
        - sid       [1]
        - rnd       [1,192,rnd_len]
          rnd_len = (full: T) / (stream: T - max(skip_head-24,0))
        rnd = N(0,1) * noise_scale
        输出：
        - wav_model: [N] @ model_sr
        - full 模式：C++ 侧裁剪 skip_head -> skip_head+return_frames
        - stream 模式：图内已裁剪，必须严格等于 return_frames
        - 之后 model_sr -> io_sr 线性重采样
   |
   v
infer_wav_io[return_frames * zc_io] @ io_sr
   |
   +--> (可选) rms_mix_rate：按 10ms 分段，把输出 RMS 拉回输入 RMS
   |    sdk/rvc_sdk_ort/src/rvc_engine.cpp:966
   |
   v
SOLA 对齐 + crossfade（相关性搜索 offset 0~10ms）
sdk/rvc_sdk_ort/src/dsp/sola.cpp:38
   |
   v
out_mono[block_size_io] @ io_sr  -> clamp[-1,1]
```

---

## 3. 当前实现与 WebUI realtime 的关键差异（用于定位“赫赫/电音”）

当你感觉音质不稳定或出现“赫赫/喘声/电音”，优先对照这些差异点：

1) **F0 方法**：当前 C++ 支持 `YIN@16k` 与 `RMVPE(onnx)` 两种路径：  
   - YIN：`sdk/rvc_sdk_ort/src/f0/yin_f0.h` / `sdk/rvc_sdk_ort/src/f0/yin_f0.cpp`  
   - RMVPE：`sdk/rvc_sdk_ort/src/f0/rmvpe_f0.h` / `sdk/rvc_sdk_ort/src/f0/rmvpe_f0.cpp`（需要 `--rmvpe rmvpe.onnx`）  
   WebUI realtime 常用 `rmvpe/harvest/fcpe`，短 block 下通常更稳；F0 抖动会直接导致“电音/跑调/胡言乱语”等伪影。

2) **重采样质量**：当前 C++ 用线性重采样（`sdk/rvc_sdk_ort/src/dsp/linear_resampler.h`）。  
   WebUI 用 torchaudio resampler，通常更高质量。

3) **静音门控**：WebUI 可用 **逐 10ms 的 dB threshold** 置零静音帧；当前 demo 的 `--gate-rms` 是整 block RMS 门限，抑制不够“细”。  
   当你的麦克风底噪较大、或开了“AI 降噪/回声消除”导致底噪不为 0 时，block 级 gate 很容易失效，从而出现“静音也在说话/赫赫声”。

4) **stream synthesizer 是“固定配置导出”**：本仓库支持两类 synthesizer.onnx  
   - full：输出整窗（C++ 侧裁剪）  
   - stream：图内已裁剪（输出严格等于 `return_frames`）  
   其中 stream 版本通常在 onnx 元数据里写入 `rvc_stream_skip_head_frames / rvc_stream_return_length_frames`，**运行时必须与导出时的 block/extra/crossfade 对齐**，否则可能直接在图内报 Reshape 错误，或音质异常。

---

## 4. 使用方法（跑/排查用）

### 4.1 先确认 I/O 链路（不跑模型）

```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --cap-id 1 --pb-id 2 `
  --passthrough --print-levels --prefill-blocks 0
```

### 4.2 跑模型（最小必要参数）

```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --enc "...\vec-768-layer-12.onnx" `
  --syn "...\xxx_stream.onnx" `
  --index "...\added_*.index" `
  --cap-id 1 --pb-id 2
```

### 4.2.1 可选：启用 RMVPE F0（推荐用于减少短 block “赫赫/电音”）

RMVPE 需要额外的 `rmvpe.onnx`，realtime demo 通过 `--rmvpe` 加载：

```powershell
build_rvc_sdk_ort/Release/rvc_sdk_ort_realtime.exe `
  --enc "...\vec-768-layer-12.onnx" `
  --syn "...\xxx_stream.onnx" `
  --index "...\added_*.index" `
  --rmvpe "...\rmvpe.onnx" `
  --cap-id 1 --pb-id 2
```

### 4.3 按链路逐段排除“赫赫”来源（建议按顺序试）

1) 关检索：`--index-rate 0`（排除 FAISS 融合带来的“飘”）  
2) 关随机：`--noise-scale 0`（排除随机噪声引入；若变电音，通常意味着你更需要稳定 F0/特征）  
3) 开门限：`--gate-rms 0.03`（压掉静音底噪触发）  
4) 开包络混合：`--rms-mix-rate 0.3~0.7`（压喘声/怪声）

---

## 5. 附：原生 GUI realtime 入口说明

这两个 bat 仅负责启动 Python GUI，本身不包含任何实时逻辑：

- `go-realtime-gui.bat:1`：`runtime\\python.exe gui_v1.py`
- `go-realtime-gui-dml.bat:1`：`runtime\\python.exe gui_v1.py --pycmd runtime\\python.exe --dml`

真正的“实时处理”在：
- `gui_v1.py:848` 的 `audio_callback()`（RMS 门限、rms_mix_rate、SOLA 等核心逻辑）。
