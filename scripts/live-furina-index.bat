@echo off
rem ============ Furina Realtime - Plan F (fp32 + index + RMS, 对齐 GPU 引擎听感) ============
rem 与 live-furina.bat（E 方案 int8 无 index）并存，互不影响。
rem 本方案模拟 GPU 官方引擎（live_gui）的配置：
rem   fp32 模型（非 int8）          -> 细节/辅音更清晰
rem   压缩索引 + index-rate 0.3     -> 音色更贴训练集（更"像"芙宁娜）
rem   rms-mix 0.5                  -> 输出响度跟随输入起伏（更自然）
rem   auto-pitch continuous        -> 男女自动收敛
rem
rem 常用参数（追加到行尾）:
rem   --denoise --denoise-mix 0.5 --agc   嘈杂环境降噪组合
rem   --silence-dbfs -60                  静音门限
rem   --input-device 麦克风 --output-device 耳机   设备选择
cd /d %~dp0\..
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 --onnx assets\weights\furina\onnx\stream150.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index assets\weights\furina\added_IVF1358_furina.small4.index --index-rate 0.3 --rms-mix 0.5 --auto-pitch --auto-pitch-mode continuous --input-device 1 --output-device 6 %*
pause