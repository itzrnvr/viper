@echo off
REM viper comprehensive benchmark script
REM Run this when GPU is free for full A/B + quality comparison

set VIPER_Scalar=D:\dev\viper\build\viper_cli_scalar.exe
set VIPER_DP4A=D:\dev\viper\build\viper_cli_dp4a.exe
set MODEL=D:\dev\viper\artifacts\Nanbeige4.2-3B.viper
set VOCAB=D:\dev\viper\artifacts\vocab.bin
set LLAMA=C:\Users\babys\Documents\llama_cpp_default_path\llama-cpp\llama-cli.exe
set GGUF=D:\tmp\nbg_gguf\Nanbeige4.2-3B-Q4_K_M.gguf
set PROMPT=What is 15 multiplied by 37? Show your work.

echo ==========================================
echo  VIPER COMPREHENSIVE BENCHMARK
echo ==========================================
echo.

echo === 1. SCALAR BASELINE (3 runs) ===
for /L %%i in (1,1,3) do (
    echo Run %%i:
    "%VIPER_Scalar%" --model "%MODEL%" --vocab "%VOCAB%" --prompt "%PROMPT%" --max-tokens 64 --spec-k 0 --prefill-batch 0 2>&1 | findstr "tok/s"
)
echo.

echo === 2. DP4A L1-CACHED (3 runs) ===
for /L %%i in (1,1,3) do (
    echo Run %%i:
    "%VIPER_DP4A%" --model "%MODEL%" --vocab "%VOCAB%" --prompt "%PROMPT%" --max-tokens 64 --spec-k 0 --prefill-batch 0 2>&1 | findstr "tok/s"
)
echo.

echo === 3. DP4A + PREFILL BATCH 8 ===
"%VIPER_DP4A%" --model "%MODEL%" --vocab "%VOCAB%" --prompt "Write a Python function to sort a list. Include docstring." --max-tokens 64 --spec-k 0 --prefill-batch 8 2>&1 | findstr "tok/s ttft"
echo.

echo === 4. DP4A + LM_HEAD PRUNE 32768 ===
"%VIPER_DP4A%" --model "%MODEL%" --vocab "%VOCAB%" --prompt "%PROMPT%" --max-tokens 64 --spec-k 0 --lm-head-prune 32768 2>&1 | findstr "tok/s"
echo.

echo === 5. SPEC DECODE K=8 (repetitive text) ===
"%VIPER_DP4A%" --model "%MODEL%" --vocab "%VOCAB%" --prompt "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog." --max-tokens 64 --spec-k 8 2>&1 | findstr "tok/s"
echo.

echo === 6. LLAMA.CPP COMPARISON ===
"%LLAMA%" -m "%GGUF%" -ngl 99 -p "%PROMPT%" -n 64 --temp 0 -no-cnv 2>&1 | findstr "tokens per second"
echo.

echo === 7. QUALITY COMPARISON ===
echo Viper:
"%VIPER_DP4A%" --model "%MODEL%" --vocab "%VOCAB%" --prompt "Explain quantum computing in simple terms." --max-tokens 128 --spec-k 0 --prefill-batch 0 2>&1 | findstr /V "\["
echo.
echo llama.cpp:
"%LLAMA%" -m "%GGUF%" -ngl 99 -p "Explain quantum computing in simple terms." -n 128 --temp 0 -no-cnv 2>&1 | findstr /V "eval"
echo.

echo ==========================================
echo  BENCHMARK COMPLETE
echo ==========================================
pause
