@echo off
rem ============ RVC Realtime Voice Changer - CUDA (PySimpleGUI) ============
rem Requires: .venv (torch cu124); w-okada style GUI, ASIO support
cd /d %~dp0\..
.venv\Scripts\python.exe apps\live_gui.py %*
pause
