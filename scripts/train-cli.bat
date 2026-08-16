@echo off
rem ============ RVC 30-min Replica CLI training ============
rem Usage: scripts\train-cli.bat --data D:\your_dry_vocals --exp my_voice --stage all
rem Requires: .venv (CUDA) for fast training
cd /d %~dp0\..
.venv\Scripts\python.exe train\train_cli.py %*
pause
