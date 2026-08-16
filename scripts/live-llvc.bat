@echo off
rem ============ LLVC Realtime Voice Changer ============
rem Requires: E:/Projects/LLVC and llvc_models checkpoints
cd /d %~dp0\..
.venv\Scripts\python.exe apps\live_llvc.py %*
pause
