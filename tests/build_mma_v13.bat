@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" -arch=sm_86 -O3 -std=c++17 "D:\dev\viper\tests\mma_ptx_test.cu" -o "D:\dev\viper\build\mma_ptx_v13.exe"
if errorlevel 1 (echo CUDA 13 MMA FAILED & exit /b 1)
D:\dev\viper\build\mma_ptx_v13.exe
