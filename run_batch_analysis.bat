@echo off
title NSTL Acoustic - Batch PRI Correlation Peak Analyzer
cd /d "%~dp0"

echo =======================================================================
echo   NSTL ACOUSTIC SIGNAL PROCESSING - BATCH PRI CORRELATION ANALYZER
echo   Automated Matched-Filter Analysis across 74 PRI Folders (5000+ Files)
echo =======================================================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please ensure Python is installed and added to PATH on this laptop.
    pause
    exit /b 1
)

python run_batch_analysis.py

echo.
echo =======================================================================
echo   Process completed. Check 'batch_pri_results' for Excel and CSV files!
echo =======================================================================
pause
