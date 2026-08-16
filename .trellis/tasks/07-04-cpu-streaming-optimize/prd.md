# CPU 流式推理优化 — 非对称 Encoder-Decoder

## Goal

将实时变声的 CPU 推理延迟从 **540ms → < 150ms**，通过非对称 encoder-decoder 架构：Encoder 用大上下文低频运行，Decoder 用小窗口高频运行。

## Background

### 当前 CPU 性能瓶颈

```
GPU 推理:   fea=0.010s  f0=0.005s  model=0.024s  总计 0.05s  ✅ 实时
CPU 推理:   fea=0.295s  f0=0.018s  model=0.219s  总计 0.54s  ❌ 断裂

CPU 瓶颈: fea(54%) + model(41%) = 95%
```

### 非对称 Encoder-Decoder 构想

```
当前（对称）:
  每帧: [10240样本] → Encoder [0.295s] → Decoder [0.219s]
         ↑ 94%重叠，反复编码相同内容

优化后（非对称）:
  Encoder（低频大窗口）: [~32000样本, 2s] → 每 N 帧跑一次 [0.3s/N 摊销]
  Decoder（高频小窗口）:  [~2560样本]       → 每帧都跑    [保持实时]

  Encoder 特征缓存 → Decoder 按需取小窗口
```

### 为什么可行

1. **Encoder（Hubert）**: 需要大上下文才能提取准确的语音/发音特征。但特征变化缓慢，不需要每帧重算。
2. **Decoder（Synthesizer/NSF-HiFiGAN）**: 只依赖当前时刻附近的特征，用小窗口即可合成高质量音频。
3. **F0 提取**: fcpe 在 CPU 上已足够快（0.018s），不需要优化。

## Requirements

1. **Encoder 大窗口缓存** — Hubert 编码器处理 2s 窗口，结果缓存，每 4-8 帧更新一次
2. **Decoder 小窗口推理** — Decoder 只处理当前帧附近的特征片段
3. **窗口管理** — 正确管理 encoder 缓存与 decoder 窗口的边界对齐
4. **总延迟 < 150ms @ block_time=0.15s** — 满足实时要求

## Architecture

```
音频输入流
    │
    ├─→ Encoder(大窗口 2s, 低频 1/6帧)
    │      ↓
    │   cached_feats[T: T+2s]  ← 缓存，新音频只更新尾部
    │      │
    │      └─→ Decoder(小窗口 0.15s)
    │             ↓
    │         合成音频 → 输出
    │
    └─→ F0 提取(fcpe) → 直接送入 Decoder
```

## Acceptance Criteria

- [ ] CPU 推理总延迟 < 150ms（block_time=0.15s 下不出现断裂）
- [ ] Encoder 缓存正确管理，窗口滑动无特征错位
- [ ] 音频质量不退化（与 GPU 版本对比无明显差异）
- [ ] 不破坏现有 GPU 路径

## Out of Scope

- 不修改 Hubert/Fcpe/Decoder 模型本身（只改调度）
- 不优化单次 inference 的计算效率（算子优化）

## Technical Notes

### 需改动的文件

| 文件 | 改动 |
|------|------|
| `infer/lib/rtrvc.py` | 新增 encoder 缓存机制、非对称 infer 逻辑 |
| `gui_v1.py` | 可能需调整 buffer 大小配合新架构 |

### Encoder 缓存设计（核心）

```python
class RVC:
    def __init__(self, ...):
        self.cached_feats = None      # [1, T_enc, 256/768] 缓存的 Hubert 特征
        self.cached_feats_len = 0     # 当前缓存长度（帧数）
        self.encoder_stride = 6       # 每 N 帧更新一次 encoder
        self.encoder_window = 200      # encoder 上下文窗口（帧数 ≈ 2s @ 100fps）

    def infer(self, input_wav, ...):
        # 1. F0 提取（不变，已经够快）
        pitch, pitchf = self.get_f0(...)

        # 2. Encoder（低频大窗口）
        if self.frame_count % self.encoder_stride == 0:
            # 拿大窗口跑 Hubert
            feats = self.model.extract_features(input_wav_big)
            self.cached_feats = feats  # 缓存
        else:
            feats = self.cached_feats  # 复用缓存

        # 3. Decoder（高频小窗口）
        # 从 cached_feats 里取当前帧附近的小窗口
        local_feats = feats[:, current_pos:current_pos+decoder_window, :]

        # 4. 合成
        audio = self.net_g.infer(local_feats, ...)
        return audio
```
