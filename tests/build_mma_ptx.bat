@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 -O3 -std=c++17 "D:\dev\viper\tests\mma_ptx_test.cu" -o "D:\dev\viper\build\mma_ptx_test.exe"
if errorlevel 1 (echo PTX MMA FAILED & exit /b 1)
D:\dev\viper\build\mma_ptx_test.exe
