@echo off
setlocal
set PROJECT=D:\dev\viper
set VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat
call "%VCVARS%" >nul
if errorlevel 1 ( echo vcvars64 FAILED & exit /b 1 )
if not exist "%PROJECT%\build" mkdir "%PROJECT%\build"
cl /nologo /std:c++20 /EHsc /O2 /W3 ^
   "%PROJECT%\tools\convert\convert.cpp" ^
   /Fe"%PROJECT%\build\viper_convert.exe"
if errorlevel 1 ( echo cl FAILED & exit /b 1 )
echo [OK  ] viper_convert.exe built
exit /b 0
