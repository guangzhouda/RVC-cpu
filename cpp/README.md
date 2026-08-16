# RVC 实时变声 C++ 插件（DirectML）

纯 C++/ONNX/DirectML 实现，零 Python 运行时。与 Python 版（apps/live_onnx.py）
逐块对拍一致（f0Corr 0.995 / melCos 0.978 / f0RMSE 4.6Hz）。

## 构建产物

| 产物 | 说明 |
|---|---|
| `rvc_selftest.exe` | 主程序：convert / devices / live 三种模式 |
| `rvc_plugin.dll` | 插件 DLL，导出 C API |

## 构建（Windows + MSVC 2022）

```
cd cpp
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

依赖已 vendored：`external/ortdml/`（onnxruntime DirectML C++ SDK）
和 `src/third_party/miniaudio.h`（WASAPI 实时音频，单头文件）。

## 用法

### 1. 文件转换（离线变声）
```
rvc_selftest.exe convert 输入.wav 输出.wav [--onnx p] [--meta p] [--cpu]
```

### 2. 实时变声（WASAPI 麦克风→耳机）
```
rvc_selftest.exe devices                          # 列出音频设备（含索引）
rvc_selftest.exe live                             # 用系统默认输入/输出
rvc_selftest.exe live --input 0 --output 1        # 指定麦克风 0、耳机 1
```

### 可调参数（live / convert 通用）

| 参数 | 作用 | 示例 |
|---|---|---|
| `--input N` / `--output N` | 选择音频设备 | `--input 0 --output 1` |
| `--rms-mix 0~1` | 响度因子：输出响度跟随输入起伏（0.5=自然） | `--rms-mix 0.5` |
| `--pitch N` | 固定半音变调（男→女约 +8~+10 最自然） | `--pitch 8` |
| `--no-auto-pitch` | 关闭自动音高（配合 `--pitch` 纯手动控制） | `--no-auto-pitch --pitch 8` |
| `--auto-pitch` | 开自动音高（默认已开） | — |
| `--target Hz` | auto-pitch 目标基准 f0（女声 200~240，男声 110~140） | `--target 220` |

> ⚠️ `--pitch` 若同时开着 auto-pitch（默认）会**叠加**。想纯手动固定音高，用 `--no-auto-pitch --pitch N`。
| `--cpu` | 全部走 CPU（无 DML 时） | `--cpu` |

设备索引来自 `devices` 命令输出（不是 Python sounddevice 的索引，miniaudio 独立枚举）。

自动音高（auto-pitch）默认开启 continuous 模式，按输入 f0 中位数自动映射到 `--target`。

### 3. 插件 DLL（C API，可被 OBS/游戏/通话宿主调用）
```c
#include "plugin.h"
void* h = rvc_engine_create(hubert, rmvpe, dec, meta, 0);
int block = rvc_block_frames(h);
float in[block], out[block];
rvc_process(h, in, out, block);
rvc_destroy(h);
```

## 架构

```
hubert → DML（卷积+Transformer，快且正确）
rmvpe  → CPU（含 GRU，DML 输出全零，实测）
dec    → DML（纯卷积，13.7ms/块）
```

## 性能（DML device_id=0，150ms 块）

- 解码器单独：13.7ms/块（10.9x 实时）
- 全链路文件转换：15s 处理 43.5s（2.9x 实时）

## 关键实现对拍点（与 Python 逐块一致）

- 重采样：torchaudio kaiser-sinc 精确复刻（3:1 抽取，×orig 坐标）
- RMVPE 解码：cents 用 pad 后索引 `20*(k-4)+1997.379`
- HuBERT 特征：帧数 `min(ceil(T/320), 实际输出帧)` + 复制末帧
- auto-pitch：先算 pitch（旧 shift）再更新 shift