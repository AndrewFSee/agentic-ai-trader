@echo off
REM Overnight sentiment database builder for Batch 2
REM Run this and leave it overnight - takes ~12 hours

echo ========================================
echo Sentiment Database Builder - Batch 2
echo ========================================
echo.
echo This will process 25 more S&P 500 stocks
echo Estimated time: 10-12 hours
echo.
echo Starting at: %date% %time%
echo.

cd /d C:\Users\Andrew\projects\agentic_ai_trader\ml_models

REM Activate conda environment if needed
call conda activate base

REM Run the script and log output
python build_sentiment_batch2.py > sentiment_batch2_log.txt 2>&1

echo.
echo ========================================
echo COMPLETE
echo ========================================
echo Finished at: %date% %time%
echo Check sentiment_batch2_log.txt for results
echo.
pause
