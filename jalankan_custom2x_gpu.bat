@echo off
title Custom2x - Local GPU (Real-ESRGAN)
cd /d "%~dp0"

set "GPU_NAME="
for /f "skip=1 delims=" %%i in ('wmic path win32_VideoController get name 2^>nul') do (
    if not defined GPU_NAME set "GPU_NAME=%%i"
)

if not defined GPU_NAME (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do set "GPU_NAME=%%i"
)

if not defined GPU_NAME set "GPU_NAME=Unknown GPU"

echo ============================================================
echo  Menjalankan Custom2x Local GPU (Real-ESRGAN)
echo  GPU: %GPU_NAME% (CUDA)
echo ============================================================
echo.
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe custom2x_local_gpu.py %*
) else (
    python custom2x_local_gpu.py %*
)
pause
