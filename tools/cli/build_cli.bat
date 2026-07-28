@echo off
setlocal
set PROJECT=D:\dev\viper
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 FAILED & exit /b 1 )
if not exist "%PROJECT%\build" mkdir "%PROJECT%\build"

REM Set INCLUDE so both cl.exe and nvcc's host pass find viper headers.
REM vcvars64 already set INCLUDE for MSVC system headers; append ours.
set "INCLUDE=%INCLUDE%;%PROJECT%\include;%PROJECT%\kernels"

set "INCLUDE=%INCLUDE%;%PROJECT%;%PROJECT%\kernels"
    -std=c++17 ^
    -Xcompiler "/std:c++17" ^
    -I "%PROJECT%\include" ^
    -I "%PROJECT%\kernels" ^
    "%PROJECT%\tools\cli\main.cpp" ^
    "%PROJECT%\src\viper\model.cpp" ^
    "%PROJECT%\kernels\ops\rmsnorm_kernel.cu" ^
    "%PROJECT%\kernels\ops\rope_kernel.cu" ^
    "%PROJECT%\kernels\ops\embedding_kernel.cu" ^
    "%PROJECT%\kernels\ops\swiglu_kernel.cu" ^
    "%PROJECT%\kernels\ops\residual_kernel.cu" ^
    "%PROJECT%\kernels\ops\linear_kernel.cu" ^
    "%PROJECT%\kernels\ops\linear_bf16_kernel.cu" ^
    "%PROJECT%\kernels\ops\sdpa_kernel.cu" ^
    "%PROJECT%\kernels\ops\sampling_kernel.cu" ^
    -o "%PROJECT%\build\viper_cli.exe"
if errorlevel 1 ( echo nvcc FAILED & exit /b 1 )
echo [OK  ] viper_cli.exe built
exit /b 0
