@echo off
setlocal
set PROJECT=D:\dev\viper
set CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo vcvars64 FAILED & exit /b 1 )
if not exist "%PROJECT%\build" mkdir "%PROJECT%\build"
set "INCLUDE=%INCLUDE%;%PROJECT%;%PROJECT%\kernels"
"%CUDA%\bin\nvcc.exe" -arch=sm_86 -O3 --use_fast_math ^
    -std=c++17 ^
    -Xcompiler "/std:c++17" ^
    -I "%PROJECT%\include" ^
    -I "%PROJECT%\kernels" ^
    "%PROJECT%\tools\cli\main.cu" ^
    "%PROJECT%\kernels\ops\rmsnorm_kernel.cu" ^
    "%PROJECT%\kernels\ops\rope_kernel.cu" ^
    "%PROJECT%\kernels\ops\embedding_kernel.cu" ^
    "%PROJECT%\kernels\ops\swiglu_kernel.cu" ^
    "%PROJECT%\kernels\ops\residual_kernel.cu" ^
    "%PROJECT%\kernels\ops\linear_kernel.cu" ^
    "%PROJECT%\kernels\ops\linear_bf16_kernel.cu" ^
    "%PROJECT%\kernels\ops\attn_decode_kernel.cu" ^
    "%PROJECT%\kernels\ops\sampling_kernel.cu" ^
    "%PROJECT%\kernels\persistent_forward.cu" ^
    "%PROJECT%\kernels\ops\fused_dp4a_kernel.cu" ^
    "%PROJECT%\kernels\ops\rmsnorm_quantize.cu" ^
    "%PROJECT%\kernels\ops\dp4a_smem_kernel.cu" ^
    "%PROJECT%\kernels\ops\swiglu_quantize.cu" ^
    "%PROJECT%\kernels\ops\linear_multim.cu" ^
    -o "%PROJECT%\build\viper_cli.exe"
if errorlevel 1 ( echo nvcc FAILED & exit /b 1 )
echo [OK  ] viper_cli.exe built
exit /b 0
