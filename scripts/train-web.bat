@echo off
rem ============ RVC WebUI - Train / Inference (CUDA) ============
rem Requires: .venv (torch cu124). Open http://127.0.0.1:7897
cd /d %~dp0\..
.venv\Scripts\python.exe apps\webui.py --pycmd .venv\Scripts\python.exe --port 7897 %*
pause
