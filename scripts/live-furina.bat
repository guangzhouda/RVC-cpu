@echo off
rem ============ Furina Realtime - Plan E (int8, no index, auto-pitch continuous) ============
rem 自动音高（continuous 自适应，无性别假设）：
rem   shift = 12*log2(目标f0 / 输入f0中位数)
rem   男声 123Hz -> 自动 +10~12 半音；女声 228Hz -> 自动 ~0；0.5 秒内收敛
rem   无需知道使用者性别，男女通用（Meloie 同款 smart pitch 思路）
rem
rem 常用参数（追加到行尾）:
rem   --denoise               前置降噪（GTCRN；仅嘈杂环境开；安静环境开了反而吞轻声、口齿不清）
rem   --denoise-mix 0.5       降噪干湿比（默认 0.7；口齿不清就调低，如 0.5）
rem   --agc                   自动增益：小声说话时自动放大（与 --denoise 配合）
rem   --silence-dbfs -60      瞬态/静音门限：低于 -60dBFS 输出静音（对应大饼枪声过滤/静音抑制）
rem   --auto-pitch-target 220   目标基准 f0（Hz）：女声目标 200~240；男声目标 110~140
rem   --auto-pitch-max 14       补偿上限（半音）
rem   --auto-pitch-mode binary  可选：男+12/女+0（需要使用者性别假设时）
rem   --auto-pitch-mode cdf     可选：分布级映射（需 --target-quantiles）
rem   --input-device 麦克风 --output-device 耳机   设备选择（名称子串）
rem   --list-devices            查看设备列表
cd /d %~dp0\..
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 --onnx assets\weights\furina\onnx\stream150.int8.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index-rate 0 --auto-pitch --auto-pitch-mode continuous %*
pause
