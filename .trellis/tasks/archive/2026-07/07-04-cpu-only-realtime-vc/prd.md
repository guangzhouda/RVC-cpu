# CPU-Only 实时变声

## Goal

将实时变声（`gui_v1.py` + `infer/lib/rtrvc.py`）改造为纯 CPU 运行，不依赖 GPU/CUDA，同时保持合理的推理延迟。

## Background

### 当前实时变声架构

```
go-realtime-gui.bat → gui_v1.py
                        ├── sounddevice (音频输入/输出)
                        ├── FreeSimpleGUI (界面)
                        ├── infer/lib/rtrvc.py (RVC 推理引擎)
                        │     ├── Fairseq Hubert (特征提取)
                        │     ├── Faiss (特征检索)
                        │     ├── F0 提取 (harvest/crepe/rmvpe/fcpe)
                        │     └── Synthesizer (声码器推理)
                        └── configs/config.py (设备配置)
```

### Config 已有 CPU 回退

`configs/config.py` → `device_config()`:
- 有 CUDA → `device = "cuda:0"`, `is_half = True`
- 有 MPS → `device = "mps"`, `is_half = False`
- 无 GPU → `device = "cpu"`, `is_half = False`

但代码中仍有硬编码的 GPU 假设需要处理。

## Requirements

1. **新增 CPU 启动脚本** — `go-realtime-gui-cpu.bat`，明确使用 CPU 模式
2. **强制 CPU 模式** — 添加命令行参数 `--cpu` 强制使用 CPU
3. **F0 提取适配** — CPU 模式下自动选择 fcpe（最快），禁用 crepe（GPU 依赖）
4. **JIT 兼容** — CPU 下半精度 JIT 不支持，确保回退到默认模型
5. **禁用不必要的 GPU 操作** — 去除 `torch.cuda.*` 调用在 CPU 路径上的影响

## Acceptance Criteria

- [x] 新增 `go-realtime-gui-cpu.bat`，双击即可 CPU 运行
- [x] `--cpu` 参数强制 `device = "cpu"`, `is_half = False`
- [x] CPU 模式下 F0 方法默认/强制使用 fcpe（已有回退逻辑）
- [x] 无 CUDA 环境下不报错（config 已有 fallback）
- [x] 推理延迟在可接受范围（< 500ms block_time 0.25s 下）

## Out of Scope

- 不优化 CPU 推理速度（那是后续任务）
- 不改动 `infer-web.py`（Web UI 已有 CPU 支持）

## Technical Notes

### 需要修改的文件

| 文件 | 改动 |
|------|------|
| `configs/config.py` | 添加 `--cpu` 参数，强制 CPU 模式 |
| `gui_v1.py` | 传递 `--cpu` 到 config |
| `infer/lib/rtrvc.py` | 确保 CPU 路径正常（JIT 回退、crepe 回退） |
| `go-realtime-gui-cpu.bat` | 新建 CPU 启动脚本 |

### rtrvc.py 已有的 CPU 相关代码

```python
# 已有注释标记的强制 CPU 测试代码：
# config.device=torch.device("cpu")########强制cpu测试
# config.is_half=False########强制cpu测试

# JIT 回退已有：
if self.use_jit and not config.dml:
    if self.is_half and "cpu" in str(self.device):
        # CPU 下半精度 JIT 不支持，回退默认模型
        set_default_model()
```

### F0 方法选择

CPU 模式下：
- `harvest` — 纯 CPU，但较慢
- `fcpe` — 推荐，CPU 上速度最好
- `crepe` — 需要 GPU（CPU 太慢）
- `rmvpe` — CPU 可用，中等速度
