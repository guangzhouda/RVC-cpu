@echo off
cd /d %~dp0
.venv-onnx-stream\Scripts\python.exe rvc_onnx_live.py %*
