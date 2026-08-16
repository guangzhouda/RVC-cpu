@echo off
rem ============ RVC Inference API Server (FastAPI) ============
rem Requires: .venv; see api_server.py for endpoints
cd /d %~dp0\..
.venv\Scripts\python.exe apps\api_server.py %*
pause
