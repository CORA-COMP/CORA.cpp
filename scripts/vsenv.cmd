@echo off
rem vsenv - runs a command in the Visual Studio (MSVC x64) environment
rem
rem Syntax:   scripts\vsenv.cmd <command> [arguments...]
rem Example:  scripts\vsenv.cmd cmake --build --preset msvc-cuda
rem The Visual Studio or Build Tools installation is found with vswhere.
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -property installationPath`) do set "VSROOT=%%i"
if not defined VSROOT (echo vsenv: no Visual Studio or Build Tools found 1>&2 & exit /b 1)
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
%*
