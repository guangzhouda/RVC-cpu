@echo off
rem ===== Inject compare: record ONCE, then feed same WAV to E and F =====
rem Usage: put your original voice at logs\record\E\input.wav (or edit below)
rem   step1: double-click record-E.bat (speak, Ctrl+C)
rem   step2: double-click this file -> generates inject_E.wav / inject_F.wav + report
chcp 65001 >nul
cd /d %~dp0\..
set PY=.venv-onnx-stream\Scripts\python.exe
if not exist logs\record\E\input.wav (
    echo [error] record first: double-click record-E.bat, speak, Ctrl+C
    pause
    exit /b 1
)
echo === inject E (int8 no-index) ===
%PY% apps\live_onnx.py --dml-device 0 --onnx assets\weights\furina\onnx\stream150.int8.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index-rate 0 --auto-pitch --auto-pitch-mode continuous --inject-wav logs\record\E\input.wav --inject-out logs\record\inject_E.wav
echo.
echo === inject F (fp32 + index + rms) ===
%PY% apps\live_onnx.py --dml-device 0 --onnx assets\weights\furina\onnx\stream150.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index assets\weights\furina\added_IVF1358_furina.small4.index --index-rate 0.3 --rms-mix 0.5 --auto-pitch --auto-pitch-mode continuous --inject-wav logs\record\E\input.wav --inject-out logs\record\inject_F.wav
echo.
echo === comparison report ===
%PY% scripts\compare_plans.py --e-out logs\record\inject_E.wav --f-out logs\record\inject_F.wav --ref logs\record\E\input.wav
echo.
echo listen: logs\record\inject_E.wav (E)  /  logs\record\inject_F.wav (F)
pause
