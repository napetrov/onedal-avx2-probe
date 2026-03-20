@echo off
setlocal

set ROOT=%~dp0..
set OUT=%ROOT%\out
if not exist "%OUT%" mkdir "%OUT%"

set BIN=%OUT%\avx2_full_probe.exe
set LOG=%OUT%\probe.log
set SUMMARY=%OUT%\summary.txt

:: Try MSVC cl.exe first, then fall back to g++ (MinGW/LLVM)
where cl >nul 2>&1
if %ERRORLEVEL%==0 (
    echo [build] cl /O2 /arch:AVX2 /EHsc
    cl /O2 /arch:AVX2 /EHsc /Fe:"%BIN%" "%ROOT%\src\avx2_full_probe.cpp" /link /OUT:"%BIN%"
) else (
    where g++ >nul 2>&1
    if %ERRORLEVEL%==0 (
        echo [build] g++ -O2 -mavx2 -mfma -mbmi -mbmi2
        g++ -O2 -mavx2 -mfma -mbmi -mbmi2 -o "%BIN%" "%ROOT%\src\avx2_full_probe.cpp"
    ) else (
        echo ERROR: No C++ compiler found. Install MSVC or MinGW.
        exit /b 1
    )
)

echo [run] %BIN%
"%BIN%" > "%LOG%" 2>&1
set RC=%ERRORLEVEL%
type "%LOG%"

for /f "tokens=*" %%L in ('findstr /B "CPU:" "%LOG%"') do set CPU_LINE=%%L
for /f "tokens=*" %%L in ('findstr "RESULTS:" "%LOG%"') do set RESULTS_LINE=%%L
for /f "tokens=*" %%L in ('findstr "ALL PASS\|FAILURES" "%LOG%"') do set VERDICT_LINE=%%L

(
echo cpu=%CPU_LINE%
echo rc=%RC%
echo results=%RESULTS_LINE%
echo verdict=%VERDICT_LINE%
) > "%SUMMARY%"

echo.
echo [summary]
type "%SUMMARY%"
exit /b %RC%
