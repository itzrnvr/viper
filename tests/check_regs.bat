@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 -O3 --use_fast_math -std=c++17 -Xcompiler "/std:c++17" -I "D:\dev\viper\include" -I "D:\dev\viper\kernels" -Xptxas=-v -c "D:\dev\viper\kernels\ops\linear_kernel.cu" -o nul 2>&1
