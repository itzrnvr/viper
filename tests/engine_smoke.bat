@echo off
setlocal
set PROJECT=D:\dev\viper
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 FAILED & exit /b 1 )
if not exist "%PROJECT%\build" mkdir "%PROJECT%\build"
"%CUDA%\bin\nvcc.exe" -arch=sm_86 -O3 --use_fast_math -std=c++20 ^
    -I "%PROJECT%" ^
    -I "%PROJECT%\include" ^
    -I "%PROJECT%\kernels" ^
    "%PROJECT%\tests\engine_smoke.cu" ^
    "%PROJECT%\kernels\ops\rmsnorm_kernel.cu" ^
    "%PROJECT%\kernels\ops\rope_kernel.cu" ^
    "%PROJECT%\kernels\ops\embedding_kernel.cu" ^
    "%PROJECT%\kernels\ops\swiglu_kernel.cu" ^
    "%PROJECT%\kernels\ops\residual_kernel.cu" ^
    "%PROJECT%\kernels\ops\linear_kernel.cu" ^
    "%PROJECT%\kernels\ops\sdpa_kernel.cu" ^
    "%PROJECT%\kernels\ops\sampling_kernel.cu" ^
    -o "%PROJECT%\build\engine_smoke.exe"
if errorlevel 1 ( echo nvcc FAILED & exit /b 1 )
"%PROJECT%\build\engine_smoke.exe"
exit /b %errorlevel%
