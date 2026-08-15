# RVC 实时变声：纯 CPU / 核显 DirectML（ONNX）实现总结

> 目标：把 RVC 实时变声跑成"不依赖 NVIDIA 独立显卡"的实时链路，并实测验证。
> 结论：**纯 CPU（RTF 1.16）与核显 DirectML（RTF 3.0+）两条路线均达到实时**，音质与 PyTorch 原版数值一致。

## 一、性能实测（本机：Ryzen 7 7840H + RTX 4060 Laptop，250ms 块，fcpe）

| 路线 | 单块耗时 | RTF（越高越好） | 状态 |
|---|---|---|---|
| PyTorch 纯 CPU（原始） | ~540ms+（HuBERT 多线程出 NaN） | 0.46 | 不实时 |
| **ONNX 纯 CPU（extra 1.5s）** | **215ms**（p95 222ms） | **1.16** | 实时 |
| **ONNX + 核显 DirectML（extra 2.5s / 1.5s）** | **75 / 77ms** | **3.33 / 3.23** | 大余量 |
| ONNX + DML + 150ms 块 + rmvpe | 49ms | 3.04 | 低延迟 |
| PyTorch CUDA（4060，官方 GUI） | ~60ms | ~4 | 参照基准 |

各环节耗时（DML，250ms 块）：HuBERT 特征 30ms、声码器解码 25ms、fcpe 音高 48ms、faiss 检索 ~5ms。

## 二、关键发现

1. **PyTorch 多线程 CPU 推理有严重问题**：本机 HuBERT 在 torch.set_num_threads(>1) 时输出 NaN 且更慢（425~448ms vs 单线程 298ms）；fcpe、rmvpe 同样"线程越多越慢"。→ 全链路改用 **ONNX Runtime**，多线程正常、无 NaN、快 4~13 倍。
2. **DirectML 是零依赖加速**：onnxruntime-directml 通过 DirectX 12 驱动任何显卡（含核显），Windows 10 1809+ 的机器基本都能用；无 DML 时自动回退纯 CPU（同一套代码）。
3. **声码器解码耗时与上下文窗口几乎无关**（131/181/281 帧都是 ~158ms CPU）——大头在 48k 升采样合成；extra_time 2.5s→1.5s 是安全的（TextEncoder 感受野 ±~0.7s + 流模型预热 ~0.24s）。
4. **音高算法**：fcpe 最快（CPU 27ms）；rmvpe 更准但 torch 版 CPU 需 167ms——导出 ONNX 后 CPU 15.6ms / DML 11.9ms，与 torch 版逐帧一致（maxdiff 6e-5 Hz）。
5. **曾踩过的坑**（都已在代码中规避）：
   - rtrvc.py 模块级 multiprocessing.Manager() 会让无 __main__ 保护的脚本 spawn 时无限递归挂死；
   - faiss 必须先于 onnxruntime 导入（Windows DLL 冲突）；
   - torch stft(return_complex=True) 无法导出 ONNX，改用 return_complex=False 等价实现；
   - 旧基准脚本 rvc_cpu_bench.py / rvc_onnx_bench.py 把 48k 缓冲误当 16k 喂给 fcpe（F0 指标虚高），rvc_onnx_live.py 已按 gui_v1.py 的正确语义实现。

## 三、新增文件

| 文件 | 说明 |
|---|---|
| rvc_onnx_live.py | 实时变声入口：麦克风 → HuBERT-ONNX → f0(fcpe/rmvpe) → faiss → 流式声码器 ONNX → SOLA → 扬声器；自动 DML/CPU |
| go-realtime-onnx.bat | 双击启动 |
| tools/export_streaming_onnx.py + infer/modules/onnx/export_streaming.py | 按 (block/crossfade/extra) 参数导出固定形状流式声码器 ONNX |
| tools/export_hubert_onnx.py | 导出 HuBERT ONNX（动态时间轴，v2 用 output_layer=12） |
| tools/export_rmvpe_onnx.py | 导出 RMVPE ONNX（音频→360 音分显著性） |
| tools/verify_streaming_onnx.py | 流式声码器 ONNX 与 torch 数值对拍 |
| tools/rvc_cpu_bench.py / rvc_onnx_bench.py / llvc_cpu_bench.py | 离线基准（滑动缓冲 + SOLA 全流程） |
| tools/run_realtime_eval_suite.py / summarize_realtime_eval.py / render_realtime_eval_report.py / compare_audio_metrics.py | 批量评估套件 |
| requirements-onnx-realtime.txt | ONNX 实时环境依赖清单 |

模型文件（已 gitignore，需本地生成）：
- assets/hubert/hubert_base.onnx（377MB，由 tools/export_hubert_onnx.py 生成）
- assets/rmvpe/rmvpe.onnx（362MB，由 tools/export_rmvpe_onnx.py 生成）
- assets/weights/*.stream*.onnx + .json（由 tools/export_streaming_onnx.py 生成）

## 四、使用方法

环境：在已有 .venv（RVC 完整依赖）基础上建 .venv-onnx-stream，安装 requirements-onnx-realtime.txt（onnxruntime 用 directml 版）。

```
:: 列出设备
.venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --list-devices

:: 不开声卡自测（测 RTF）
.venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --selftest

:: 实时变声（核显 DML，150ms 低延迟 + rmvpe 音高）
.venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --input-device 1 --output-device 7 --pitch 8 --f0method rmvpe --latency low --onnx assets/weights/hatsune_miku_model.stream150.onnx --meta assets/weights/hatsune_miku_model.stream150.onnx.json

:: 强制纯 CPU（无 DML 机器）
.venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py --provider cpu
```

关键参数：--provider auto|dml|cpu、--f0method fcpe|rmvpe、--index-rate（0=关闭检索）、--pitch（半音）、--latency low|high、--wasapi-exclusive。

## 五、质量验证

- HuBERT ONNX vs torch：maxdiff 1.4e-5（1.25s 窗口）；
- 流式声码器 ONNX vs torch：corr > 0.95（verify_streaming_onnx.py）；
- DML vs CPU（同一 ONNX）：解码器 maxdiff 9e-7、HuBERT 6e-5，corr 1.0；
- RMVPE ONNX vs torch：f0 逐帧 maxdiff 6e-5 Hz，corr 1.0。

## 七、跨机器使用注意（重要）

DML 的提速幅度**取决于机器的核显/独显算力**：本机 Radeon 780M（较强核显）解码器从 158ms 降到 14ms，但老旧 Intel UHD / 入门 Vega 核显可能只有 1.5~2x，甚至因 GPU 不支持部分算子（STFT、ConvTranspose、动态形状）导致大量 CPU 回退而更慢。

--provider auto（默认）会在启动时用解码器小模型快速对拍 CPU 与 DML 的实际速度（约几秒，不需加载大模型），自动选择更快的后端并打印基准结果。必要时也可手动指定 --provider dml 或 --provider cpu。

建议在其他机器上首次运行时保持 --provider auto，观察打印的 [auto] 解码器基准: cpu=X dml=Y，若 Y 明显更大说明该卡不适应 DML，程序会自动用 CPU（纯 CPU ONNX 路径在本机 RTF 1.16，仍为实时）。

## 六、已知限制与后续

- 延迟体感 = 块大小（150/250ms）+ 声卡缓冲；更低延迟需更小块重导出并验证。
- extra_time 下限 ~1.0s（感受野约束），1.0s 配置音质略降，推荐 1.5s。
- 解码器 int8 量化（QDQ）可再降 CPU 耗时，但动态量化会撞 ORT 缺 ConvInteger 内核，未完成。
- fcpe 在 DML 链路上仍是最大单项（48ms，CPU torch）；导出 fcpe→ONNX/DML 可再压 ~40ms。
- 旧 gui_v1.py 的 GPU 路线保持不变；本方案是独立命令行入口。
