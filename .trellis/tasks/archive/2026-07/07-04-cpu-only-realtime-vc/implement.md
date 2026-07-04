# Implement Plan: CPU-Only 实时变声

## Step 1: 修改 configs/config.py — 添加 `--cpu` 参数

```python
parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
```

在 `device_config()` 最前面添加 CPU 强制检查。

## Step 2: 新建 go-realtime-gui-cpu.bat

```batch
runtime\python.exe gui_v1.py --cpu
pause
```

## Step 3: 验证 rtrvc.py CPU 路径

- JIT + CPU + half → 自动回退 default model（已有）
- crepe + CPU → 自动回退 fcpe（已有，但检查 DML 判断逻辑）
- 确认 `.to(self.device)` 在 `device="cpu"` 时正常

## Step 4: 测试

- 无 GPU 环境启动不报错
- F0 提取正常工作
- 音频推理正常输出

## Validation

```bash
# 启动 CPU 模式
python gui_v1.py --cpu

# 检查设备检测日志
# 应输出: "Use cpu instead"
# 应输出: "Half-precision floating-point: False, device: cpu"
```
