# 大饼式管线 SOP（DML 实时变声版）

> 目标：把「大饼AI变声」的公开管线经验，映射到本仓库的 DirectML（DML）实时变声路线上。
> 本机硬件：RTX 4060 Laptop 8GB + AMD 780M 核显（双显卡笔记本）。
> 明确约定：**不使用 CUDA**，推理全部走 onnxruntime DirectML。

## 一、本机实测数据（2026-07-06，初音 v2 48k 模型）

| 配置 | 解码器(ms/step) | 整体 RTF |
|---|---|---|
| DML 默认（未指定设备） | 20.9 | 2.35x |
| DML device_id=0（推荐） | **15.4** | — |
| DML device_id=1 | 55.4 | — |
| 纯 CPU | 165.4 | — |
| f0method=fcpe（默认 DML） | — | 2.35x (107ms/250ms) |
| f0method=rmvpe（默认 DML） | — | 4.34x (58ms/250ms) |
| **rmvpe + device_id=0（现默认）** | — | **4.94x (51ms/250ms)** |
| stream150（150ms 块）+ rmvpe + device 0 | — | 约 2x（仍实时，端到端延迟更低） |

结论：4060 跑 DML 完全够实时；**双显卡机器必须显式指定 device_id**，且本机 rmvpe 比 fcpe 又快又准。

## 二、快速开始

双击 `scripts\live-dml.bat`，或手动：

```bat
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0
```

### 选择麦克风/扬声器

1. 查看设备列表（名称+索引）：
   ```bat
   .venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --list-devices
   ```
2. 用名称子串或索引指定（子串即可，如 `麦克风`、`耳机`、`VoiceMeeter`）：
   ```bat
   .venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 --input-device 麦克风 --output-device 耳机
   ```
   - 缺省 = 系统默认设备；
   - `--check-device` 可只打印解析结果不启动；
   - 名称写错会报错并列出全部可用设备。
3. 或把参数直接追加到 `scripts\live-dml.bat` 的 `%*` 位置（`--input-device 麦克风 --output-device 耳机`），双击即用。

典型路由（大饼同款虚拟声卡思路）：
- 自己听效果：输入=物理麦克风，输出=耳机；
- 变声进游戏/语音软件：输出选 VoiceMeeter/VB-CABLE 虚拟输入，软件里把麦克风选成对应虚拟输出。

常用参数（大饼参数对照）：

| 参数 | 大饼对应 | 说明 |
|---|---|---|
| `--pitch 9~12` | 音调 9-12（官方男变女推荐） | 半音变调，男→女建议 12 |
| `--index-rate 0.75` | Any-to-One 检索替换 | 默认已 0.75；1.0 更像但易糊 |
| `--dml-device 0` | CUDA 加速模块 | 双显卡必须指定，本机实测 0 最快 |
| `--latency low --wasapi-exclusive` | 低延迟模式（80ms 级宣传） | 爆音则退回默认 high |
| `--f0method rmvpe` | 高质量音高提取 | 已设为默认 |

## 三、大饼管线 → 本仓库映射

| 大饼环节 | 本仓库对应 |
|---|---|
| 麦克风 → AI降噪 → 引擎 | 输入侧加 RNNoise / 系统降噪（外部插件，管线暂未内置） |
| 本地模型推理（CUDA） | onnxruntime DirectML（device_id=0） |
| 虚拟声卡输出（Dubbing Virtual Device） | VoiceMeeter / VB-CABLE（config 里已配 VoiceMeeter） |
| 音调 9-12、共振峰 0.5-2.0 | `--pitch`（共振峰 RVC 管线无 DSP 等价物，靠检索混合补偿） |
| 统一 44100/48000Hz | 模型 48k，流固定 48k，设备不符由 WASAPI 重采样 |
| 30 分钟音频复刻 | 见第五节自训练 |

## 四、实时效果 checklist（大饼 FAQ 同款建议）

1. **必须用耳机**——扬声器声音会串回麦克风造成回声；
2. 笔记本插电，避免省电降频；
3. Windows 声音设置里关闭「音频增强」；
4. 麦克风 15-25cm 距离，环境噪音 <45dB；
5. 有回音：先关耳返思路（本管线无耳返），降低输入增益至 70% 以下；
6. 音质发飘/断句：加 `--latency high`（默认即 high）；
7. 想要更低延迟：用 stream150 模型 + `--latency low --wasapi-exclusive`。

## 五、30 分钟复刻（训练自己的模型）

> 大饼官方**没有公开训练方法**："30 分钟复刻"只是宣传口径，定制服务仅对主播/企业开放。
> 本仓库用开源 RVC 实现等价流程，一键 CLI：`tools/train_cli.py`

**数据规范（大饼同款）**：30 分钟以上目标音色干声（无伴奏、无混响、低底噪），15-25cm 收音、环境 <45dB。带伴奏素材先用 UVR5 分离人声。可整段放一个目录，脚本会自动切片。

**完整训练（约 1-2 小时，4060）**：

```bat
.venv\Scripts\python.exe train\train_cli.py --data "D:\干声目录" --exp 我的音色 --stage all --epochs 150 --batch 8
```

分步执行（断点续跑）：
```bat
.venv\Scripts\python.exe train\train_cli.py --data "D:\干声目录" --exp 我的音色 --stage preprocess   :: 切片
.venv\Scripts\python.exe train\train_cli.py --exp 我的音色 --stage extract       :: RMVPE 音高 + HuBERT 特征
.venv\Scripts\python.exe train\train_cli.py --exp 我的音色 --stage train         :: 训练
.venv\Scripts\python.exe train\train_cli.py --exp 我的音色 --stage index         :: 构建检索索引
```

**关键参数**：
- `--sr 48k`（默认，与实时 ONNX 管线一致，导出免重采样）
- `--epochs`：30 分钟数据 100~200；10 分钟数据建议 200~300
- `--batch 8`：4060 8GB 建议 8~12
- 训练完成后：
  - 推理格式权重：`assets/weights/<exp>.pth`（最终版）与 `assets/weights/<exp>_eN_sM.pth`（每 `--save-every` epoch 的中间版）——**导出 ONNX 用这个**；**导出后建议把权重和索引移入 `assets/weights/<角色名>/` 目录**（目录规范见 assets/weights/README.md）
  - 训练检查点：`logs/<exp>/G_xxx.pth`（含优化器状态，仅供断点续训，不能直接导出）
  - 检索索引：`logs/<exp>/added_*.index`

**导出实时 ONNX 并接入**（以 exp=我的音色 为例）：

```bat
.venv\Scripts\python.exe export\export_streaming_onnx.py --model assets\weights\我的音色.pth --output logs\我的音色\model.stream150.onnx --block-time 0.15
.venv\Scripts\python.exe export\verify_streaming_onnx.py --model assets\weights\我的音色.pth --onnx logs\我的音色\model.stream150.onnx --block-time 0.15
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 --onnx logs\我的音色\model.stream150.onnx --meta logs\我的音色\model.stream150.onnx.json --index logs\我的音色\added_*.index --pitch 12
```

（已实测全链路：预处理→提取→训练→索引→导出→verify PASS 0.996→离线推理正常。）

**调参**：先 `--pitch 0` 听底子，再按目标性别调 9-12，最后 index-rate 0.3 起步（糊就降、不像就升）。

> 注：训练用 .venv（CUDA，一次性离线任务，约比 DML 快 5-10 倍，不影响实时管线的 DML 路线）；
> 坚持纯 DML 训练可另建 `torch-directml` 环境（官方 environment_dml.yaml，慢很多）。

## 六、客观质量评估（PESQ 等）

评估工具与结论（2026-08 诊断实录，样本按角色分目录：`logs/offline_test/miku/`、`logs/offline_test/furina/`，参考音频在 `reference/`）：

| 工具 | 位置 | 用途 |
|---|---|---|
| PESQ-WB / STOI / SI-SNR / LSD / melCos | `eval/evaluate_quality.py` | 相对参考的损伤程度 |
| RMVPE f0 轨迹（f0Corr/f0RMSE） | 同上（内置） | 音高轮廓保留度 |
| Whisper ASR WER | `eval/asr_transcribe.py` | 可懂度终裁 |
| ONNX vs torch 解码器一致性 | `export/verify_streaming_onnx.py` | 模型导出正确性 |
| 离线变声 | `convert/offline_convert.py` | 排除声卡干扰、生成评估样本 |

**评估要点（血泪教训）**：
1. **变声 vs 原声的 PESQ/STOI 天然低**（说话人已改变），PESQ≈1.0/STOI≈0.25 不代表损坏；
2. **f0 指标必须用 RMVPE 提取**：torchaudio detect_pitch 对明亮女声音色会锁谐波假频
   （原声 124Hz 被 detect 成 178Hz，变声输出被 detect 成 213Hz，得出 f0Corr≈0 的假结论；
   换 RMVPE 后 f0Corr=0.845、RMSE=17Hz，音高完全跟随）；
3. **可懂度看 ASR**：Whisper base 转写变声输出，关键短语完整保留 = 输出正常。

## 七、本次诊断修复记录

| 问题 | 结论 |
|---|---|
| 滑动窗口前移量误用 n16(4160) 应为 block_frame_16k(4000) | 已修（`apps\live_onnx.py`），每块 10ms 漂移消除 |
| `hatsune_miku_model.stream.onnx`(250ms) 导出损坏 | verify 仅 pearson 0.90；已换用 stream150（0.994 PASS）；250ms 版已重新导出为 stream_fixed.onnx |
| DML 双显卡设备选择 | 实测 device_id=0 最快（15ms vs 默认 21ms vs device 1 的 55ms），默认参数已带 `--dml-device 0` |
| f0method 默认 fcpe → rmvpe | DML 上 rmvpe 更快更准（58ms vs 107ms/块） |
| pip 走阿里云镜像 SSL 被拦 | 依赖包从 PyPI 直链下载到 `tools/_wheels/` 后本地安装 |

## 八、已做修改（本次）

- `apps\live_onnx.py`：
  - 新增 `--dml-device` 参数（默认 None=系统默认；本机建议 0）；
  - `_pick_providers` 支持 DML device_id 指定；
  - 默认 f0method 由 fcpe 改为 rmvpe（更快更准）。
- `scripts\live-dml.bat`：默认追加 `--dml-device 0`，附参数速查注释。

## 参考来源

- 量子位·30分钟复刻报道: https://cloud.tencent.cn/developer/article/2307717
- 大饼官网 FAQ / 使用教程: https://www.dabing-china.com/faq.html , https://www.dabing-china.com/guide.html
- ZEGO 大饼 AI 变声实现文档: https://doc-zh.zego.im/ai-voice-changer-windows/quick-start/implementation
- 网信办第四批深度合成算法备案（第45条 格子互动大饼语音合成类算法）: https://www.ncwxw.gov.cn/sys-nd/329.html
- B站大饼新人必读: https://m.bilibili.com/opus/812840088542117909
