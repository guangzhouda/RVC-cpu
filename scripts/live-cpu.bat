@echo off
rem ============ RVC Realtime Voice Changer - CPU ============
rem Requires: .venv-onnx-stream; no GPU needed, slower
cd /d %~dp0\..
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --provider cpu %*
pause
