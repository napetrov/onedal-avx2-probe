@echo off
setlocal EnableDelayedExpansion

set ROOT=%~dp0..
set OUT=%ROOT%\out
if not exist "%OUT%" mkdir "%OUT%"

set BIN_AVX2=%OUT%\avx2_full_probe.exe
set BIN_ISA=%OUT%\isa_dispatch_probe.exe
set SUMMARY=%OUT%\summary.txt

:: Detect compiler
where cl >nul 2>&1
if %ERRORLEVEL%==0 (
    set COMPILER=cl
    set CFLAGS=/O2 /arch:AVX2 /EHsc
) else (
    where g++ >nul 2>&1
    if %ERRORLEVEL%==0 (
        set COMPILER=g++
        set CFLAGS=-O2 -msse4.2 -mavx2 -mfma -mbmi -mbmi2
    ) else (
        echo ERROR: No C++ compiler found. Install MSVC or MinGW.
        exit /b 1
    )
)

echo ================================================
echo  Probe 1: AVX2 Full Suite
echo ================================================
if "%COMPILER%"=="cl" (
    cl %CFLAGS% /Fe:"%BIN_AVX2%" "%ROOT%\src\avx2_full_probe.cpp" >nul 2>&1
    if errorlevel 1 (
        echo [build] Trying without /arch:AVX2 flag...
        cl /O2 /EHsc /Fe:"%BIN_AVX2%" "%ROOT%\src\avx2_full_probe.cpp"
    )
) else (
    g++ %CFLAGS% -o "%BIN_AVX2%" "%ROOT%\src\avx2_full_probe.cpp"
)
"%BIN_AVX2%" > "%OUT%\avx2_probe.log" 2>&1
set RC_AVX2=%ERRORLEVEL%
type "%OUT%\avx2_probe.log"

echo.
echo ================================================
echo  Probe 2: ISA Dispatch (SSE2 / SSE4.2 / AVX2)
echo ================================================
if "%COMPILER%"=="cl" (
    cl %CFLAGS% /Fe:"%BIN_ISA%" "%ROOT%\src\isa_dispatch_probe.cpp" >nul 2>&1
    if errorlevel 1 (
        cl /O2 /EHsc /Fe:"%BIN_ISA%" "%ROOT%\src\isa_dispatch_probe.cpp"
    )
) else (
    g++ %CFLAGS% -o "%BIN_ISA%" "%ROOT%\src\isa_dispatch_probe.cpp"
)
"%BIN_ISA%" > "%OUT%\isa_probe.log" 2>&1
set RC_ISA=%ERRORLEVEL%
type "%OUT%\isa_probe.log"

:: Summary
for /f "tokens=*" %%L in ('findstr /B "CPU:" "%OUT%\isa_probe.log" 2^>nul') do set CPU_LINE=%%L
for /f "tokens=*" %%L in ('findstr "USE AVX2\|USE SSE4.2\|USE SSE2\|NO SIMD" "%OUT%\isa_probe.log" 2^>nul') do set DISPATCH=%%L

(
echo cpu=%CPU_LINE%
echo rc_avx2=%RC_AVX2%
echo rc_isa=%RC_ISA%
echo dispatch=%DISPATCH%
) > "%SUMMARY%"

echo.
echo ================================================
echo  SUMMARY
echo ================================================
type "%SUMMARY%"

if %RC_AVX2% neq 0 exit /b %RC_AVX2%
exit /b %RC_ISA%
