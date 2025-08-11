@echo off
REM Trading Automation Batch Script
REM Alternative to PowerShell script for simpler Task Scheduler setup

setlocal

REM Set script directory
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Create log file with timestamp
set LOG_FILE=%SCRIPT_DIR%automation_batch_log.txt
echo [%date% %time%] Trading Automation Started >> "%LOG_FILE%"

REM Determine day of week (1=Monday, 7=Sunday)
for /f "tokens=1" %%i in ('powershell -command "& {(Get-Date).DayOfWeek.value__}"') do set DAY_NUM=%%i

echo [%date% %time%] Day number: %DAY_NUM% >> "%LOG_FILE%"

REM Execute based on day
if %DAY_NUM% GEQ 1 if %DAY_NUM% LEQ 5 (
    echo [%date% %time%] Weekday detected - Running mobile signals >> "%LOG_FILE%"
    goto :weekday
)

if %DAY_NUM%==6 (
    echo [%date% %time%] Saturday detected - Running optimizations >> "%LOG_FILE%"
    goto :saturday
)

if %DAY_NUM%==0 (
    echo [%date% %time%] Sunday detected - Running optimal strategy finder >> "%LOG_FILE%"
    goto :sunday
)

echo [%date% %time%] Unknown day: %DAY_NUM% >> "%LOG_FILE%"
goto :end

:weekday
echo Running mobile signals generation...
python mobile_signals.py >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    echo [%date% %time%] Mobile signals failed >> "%LOG_FILE%"
    goto :end
)

REM Check if server is running (simple check)
netstat -an | find ":3001" >nul
if errorlevel 1 (
    echo Starting mobile server...
    echo [%date% %time%] Starting mobile server >> "%LOG_FILE%"
    start /min python mobile_server.py
) else (
    echo [%date% %time%] Mobile server already running >> "%LOG_FILE%"
)
goto :end

:saturday
echo Running optimization suite...
echo [%date% %time%] Starting optimization scripts >> "%LOG_FILE%"

REM Start optimization scripts in parallel
start /b python examples\bollinger_bands_optimization.py >> "%LOG_FILE%_bb.txt" 2>&1
start /b python examples\ma_optimization.py >> "%LOG_FILE%_ma.txt" 2>&1
start /b python examples\rsi_optimization.py >> "%LOG_FILE%_rsi.txt" 2>&1

echo [%date% %time%] All optimization scripts started >> "%LOG_FILE%"
goto :end

:sunday
echo Running optimal strategy finder...
echo [%date% %time%] Starting optimal strategy finder >> "%LOG_FILE%"
python examples\optimal_strategy_finder.py >> "%LOG_FILE%" 2>&1
goto :end

:end
echo [%date% %time%] Trading Automation Completed >> "%LOG_FILE%"
