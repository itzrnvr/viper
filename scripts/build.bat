@echo off
REM Viper build wrapper — sources MSVC 2019 + CUDA 12.8, then CMake + build.
REM Usage: scripts\build.bat [Release|Debug|RelWithDebInfo] [Ninja|NMake]

setlocal EnableExtensions EnableDelayedExpansion

set "CFG=%~1"
if "%CFG%"=="" set "CFG=Release"

set "GEN=%~2"
if "%GEN%"=="" (
    where ninja >nul 2>nul
    if %ERRORLEVEL%==0 (
        set "GEN=Ninja"
    ) else (
        set "GEN=NMake Makefiles"
    )
)

set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
    echo [ERR ] vcvars64.bat not found at "%VCVARS%"
    exit /b 1
)

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo [ERR ] nvcc.exe not found at "%CUDA_PATH%\bin\nvcc.exe"
    exit /b 1
)

call "%VCVARS%" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERR ] vcvars64.bat failed
    exit /b 1
)

set "PATH=%CUDA_PATH%\bin;%PATH%"

pushd "%~dp0\.."

set "BUILD_DIR=build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [INFO] Configuring with %GEN% (%CFG%)
cmake -S . -B "%BUILD_DIR%" -G "%GEN%" -DCMAKE_BUILD_TYPE=%CFG% ^
    -DCUDA_TOOLKIT_ROOT_POINT="%CUDA_PATH%" 2>&1 | findstr /v "^--"
if %ERRORLEVEL% NEQ 0 (
    echo [ERR ] CMake configure failed
    popd
    exit /b 1
)

echo [INFO] Building
cmake --build "%BUILD_DIR%" --config %CFG% 2>&1 | findstr /v "^--"
set "RC=%ERRORLEVEL%"
popd

if %RC% NEQ 0 (
    echo [ERR ] Build failed
    exit /b %RC%
)

echo [OK  ] Build complete: %BUILD_DIR%\bin
exit /b 0