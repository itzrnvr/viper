@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul

echo === Testing WMMA FP16 ===
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 -O3 -std=c++17 "D:\tmp\wmma_fp16_test.cu" -o "D:\dev\viper\build\wmma_fp16.exe" 2>&1
if errorlevel 1 (echo WMMA FP16: FAILED) else (echo WMMA FP16: SUCCESS)

echo.
echo === Testing WMMA with -arch=sm_80 ===
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_80 -O3 -std=c++17 "D:\tmp\wmma_fp16_test.cu" -o "D:\dev\viper\build\wmma_fp16_80.exe" 2>&1
if errorlevel 1 (echo WMMA FP16 sm_80: FAILED) else (echo WMMA FP16 sm_80: SUCCESS)

echo.
echo === Testing with -gencode ===
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -gencode arch=compute_86,code=sm_86 -O3 -std=c++17 "D:\tmp\wmma_fp16_test.cu" -o "D:\dev\viper\build\wmma_gencode.exe" 2>&1
if errorlevel 1 (echo WMMA gencode: FAILED) else (echo WMMA gencode: SUCCESS)
