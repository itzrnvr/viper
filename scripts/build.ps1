# Viper build wrapper — PowerShell variant. Sources MSVC 2019 + CUDA 12.8,
# then CMake + build.
# Usage: scripts\build.ps1 [Release|Debug|RelWithDebInfo] [Ninja|NMake]

[CmdletBinding()]
param(
    [string]$Config = "Release",
    [string]$Generator = ""  # auto-detect: Ninja if available, else NMake Makefiles
)

$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found at $vcvars" }
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if (-not (Test-Path "$cudaPath\bin\nvcc.exe")) { throw "nvcc.exe not found at $cudaPath" }

if (-not $Generator) {
    $ninja = Get-Command ninja -ErrorAction SilentlyContinue
    $Generator = if ($ninja) { "Ninja" } else { "NMake Makefiles" }
}

# Source vcvars64 in this shell.
cmd /c "`"$vcvars`" && set PATH=$cudaPath\bin;%PATH%" | Out-Null
$env:PATH = "$cudaPath\bin;$env:PATH"

Push-Location (Join-Path $PSScriptRoot "..")
try {
    if (-not (Test-Path "build")) { New-Item -ItemType Directory -Path "build" | Out-Null }

    Write-Host "[INFO] Configuring with $Generator ($Config)"
    cmake -S . -B build -G $Generator -DCMAKE_BUILD_TYPE=$Config `
        -DCUDA_TOOLKIT_ROOT_POINT="$cudaPath" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

    Write-Host "[INFO] Building"
    cmake --build build --config $Config 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    Write-Host "[OK  ] Build complete: build\bin"
} finally {
    Pop-Location
}