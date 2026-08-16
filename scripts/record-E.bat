@echo off
rem ===== Record original voice (once) - Plan E engine =====
rem Speak 10-30s, then press Ctrl+C to stop.
rem Outputs: logs\record\E\input.wav (original) + output.wav (live converted, listenable)
rem Then double-click compare-record.bat for fair E-vs-F report.
rem
rem TIP: raise mic volume in Windows Sound Settings (Recording > Mic > Level 100).
rem      A -50dBFS input is too quiet and shrinks rms-mix output.
cd /d %~dp0\..
.venv-onnx-stream\Scripts\python.exe apps\live_onnx.py --dml-device 0 --onnx assets\weights\furina\onnx\stream150.int8.onnx --meta assets\weights\furina\onnx\stream150.onnx.json --index-rate 0 --auto-pitch --auto-pitch-mode continuous --input-device 1 --output-device 6 --record-in logs\record\E\input.wav --record-out logs\record\E\output.wav %*
pause
