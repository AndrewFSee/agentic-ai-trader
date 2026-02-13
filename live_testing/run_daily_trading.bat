@echo off
REM Forward Paper Trading - Daily Runner
REM This script runs the paper trader and logs output

cd /d "%~dp0"

REM Disable the venv_research auto-execution (it's missing dependencies)
set SKIP_VENV_REEXEC=1

REM Activate conda base environment (has all dependencies)
call conda activate base

REM Run paper trader
echo Running paper trader at %date% %time%
python paper_trader.py >> trading_logs\daily_runner.log 2>&1

REM Check exit code
if %ERRORLEVEL% EQU 0 (
    echo Success at %date% %time% >> trading_logs\daily_runner.log
) else (
    echo Error at %date% %time% - Exit Code: %ERRORLEVEL% >> trading_logs\daily_runner.log
)
