@echo off
rem ===== Fair comparison E vs F (same input) =====
rem 1) offline convert logs\record\E\input.wav with E config -> fair_E.wav
rem 2) offline convert same input with F config -> fair_F.wav
rem 3) report: E-vs-F gap + closeness to original
chcp 65001 >nul
cd /d %~dp0\..
set PY=.venv-onnx-stream\Scripts\python.exe
if not exist logs\record\E\input.wav (
    echo [error] record first: double-click record-E.bat, speak, Ctrl+C
    pause
    exit /b 1
)
echo === generate E output (same input) ===
%PY% convert\offline_convert.py logs\record\E\input.wav logs\record\fair_E.wav --dml-device 0 --onnx assets\weights\furina\onnx\stream150.int8.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index-rate 0 --auto-pitch --auto-pitch-mode continuous
echo.
echo === generate F output (same input) ===
%PY% convert\offline_convert.py logs\record\E\input.wav logs\record\fair_F.wav --dml-device 0 --onnx assets\weights\furina\onnx\stream150.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index assets\weights\furina\added_IVF1358_furina.small4.index --index-rate 0.3 --rms-mix 0.5 --auto-pitch --auto-pitch-mode continuous
echo.
echo === comparison report ===
%PY% scripts\compare_plans.py --e-out logs\record\fair_E.wav --f-out logs\record\fair_F.wav --ref logs\record\E\input.wav
echo.
echo listen: logs\record\fair_E.wav (E)  /  logs\record\fair_F.wav (F)
pause
