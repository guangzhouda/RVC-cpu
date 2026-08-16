@echo off
rem ===== OPTIONAL: live record with Plan F engine =====
rem Same input is not required (fair compare uses E input offline).
rem Outputs: logs\record\F\input.wav + output.wav (F live, listenable)
cd /d %~dp0\..
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 --onnx assets\weights\furina\onnx\stream150.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index assets\weights\furina\added_IVF1358_furina.small4.index --index-rate 0.3 --rms-mix 0.5 --auto-pitch --auto-pitch-mode continuous --input-device 1 --output-device 6 --record-in logs\record\F\input.wav --record-out logs\record\F\output.wav %*
pause
