@echo off
rem ============ RVC Realtime Voice Changer - DirectML (recommended) ============
rem Requires: .venv-onnx-stream (onnxruntime-directml)
rem Optional args: --pitch 12 --index-rate 0.3 --input-device 麦克风 --output-device 耳机
cd /d %~dp0\..
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 %*
pause
