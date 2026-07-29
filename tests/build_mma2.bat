@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 -O3 -std=c++17 "D:\dev\viper\tests\mma_fp16_vs_s8.cu" -o "D:\dev\viper\build\mma_test2.exe" 2>&1
if errorlevel 1 (
    echo === FP16 compiled but S8 failed — testing FP16 only ===
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 -O3 -std=c++17 -DFP16_ONLY "D:\dev\viper\tests\mma_fp16_vs_s8.cu" -o "D:\dev\viper\build\mma_test2.exe" 2>&1
)
if errorlevel 1 exit /b 1
D:\dev\viper\build\mma_test2.exe
