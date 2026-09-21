@echo off
title Custom2x - Local GPU (Real-ESRGAN)
cd /d "%~dp0"
echo ============================================================
echo  Menjalankan Custom2x Local GPU (Real-ESRGAN)
echo  GPU: NVIDIA GeForce GTX 1060 6GB (CUDA)
echo ============================================================
echo.
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe custom2x_local_gpu.py %*
) else (
    python custom2x_local_gpu.py %*
)
pause
