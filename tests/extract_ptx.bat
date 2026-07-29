@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -arch=sm_86 -ptx "D:\tmp\wmma_simple.cu" -o "D:\tmp\wmma_simple.ptx"
findstr /i "mma" "D:\tmp\wmma_simple.ptx"
