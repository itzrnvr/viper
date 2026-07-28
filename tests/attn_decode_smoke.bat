@echo off
setlocal
set "VC=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CU=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "P=D:\dev\viper"
call "%VC%" >nul 2>&1
if errorlevel 1 ( echo vcvars64 FAILED & exit /b 1 )
if not exist "%P%\build" mkdir "%P%\build"
"%CU%\bin\nvcc.exe" -arch=sm_86 -O3 --use_fast_math -std=c++20 ^
    -I "%P%" -I "%P%\kernels" ^
    "%P%\tests\attn_decode_smoke.cu" "%P%\kernels\ops\attn_decode_kernel.cu" ^
    -o "%P%\build\attn_decode_smoke.exe"
if errorlevel 1 ( echo BUILD FAILED & exit /b 1 )
"%P%\build\attn_decode_smoke.exe"
