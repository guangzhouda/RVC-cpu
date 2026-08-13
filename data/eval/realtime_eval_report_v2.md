# Realtime Eval Report

本报告比较三条链路：
- `RVC PyTorch GPU` 作为 baseline
- `RVC PyTorch CPU`
- `LLVC CPU streaming`

客观指标统一以 `RVC GPU baseline` 输出为参考。

## 短句集

样本数: `5`，平均时长: `4.35s`，中位时长: `4.22s`。

### 实时性能

| Engine | Budget | N | Avg ms | P95 ms | RTF | Realtime OK Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RVC GPU baseline | 250ms block | 5 | 75.52 | 90.34 | 3.62 | 5 |
| RVC CPU | 250ms block | 5 | 1111.92 | 1146.73 | 0.24 | 0 |
| LLVC CPU | 26ms chunk | 5 | 75.95 | 80.37 | 0.36 | 0 |

### 相对 RVC GPU baseline 的客观指标

| Candidate | N | Wave RMSE | Log-mel L1 | MCD | F0 RMSE (Hz) | F0 Corr | SI-SDR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RVC CPU | 5 | 0.0498 | 1.2104 | 83.77 | 393.62 | 0.774 | -9.22 |
| LLVC CPU | 5 | 0.0614 | 16.4124 | 1212.82 | 681.57 | 0.156 | -28.08 |

## 长句集

样本数: `2`，平均时长: `11.01s`，中位时长: `11.01s`。

### 实时性能

| Engine | Budget | N | Avg ms | P95 ms | RTF | Realtime OK Count |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RVC GPU baseline | 250ms block | 2 | 40.76 | 43.72 | 6.13 | 2 |
| RVC CPU | 250ms block | 2 | 697.54 | 706.88 | 0.36 | 0 |
| LLVC CPU | 26ms chunk | 2 | 59.09 | 63.21 | 0.44 | 0 |

### 相对 RVC GPU baseline 的客观指标

| Candidate | N | Wave RMSE | Log-mel L1 | MCD | F0 RMSE (Hz) | F0 Corr | SI-SDR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RVC CPU | 2 | 0.0444 | 0.7502 | 49.56 | 307.67 | 0.865 | -9.45 |
| LLVC CPU | 2 | 0.0590 | 13.2596 | 970.08 | 700.61 | 0.190 | -30.89 |

## 已知限制

- 当前输入集来自 `LibriSpeech dev-clean-2`，为英文干净语音，不代表中文场景。
- 本机上 `RVC CPU` 的 HuBERT 多线程会产出非有限值，因此 benchmark 自动回退到 `threads=1`。
- `LLVC` 与 `RVC` 不是同一目标模型架构，因此客观指标仅表示“接近 RVC GPU baseline 的程度”，不是同架构等价替换证明。
- `MCD` 这里是基于直接对齐后的 MFCC 均值距离，适合做相对比较，不适合和其他论文数值横向对齐。

