@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   Hetero Launcher (finalcuda.cu)
echo ============================================
echo.

REM Ask for Query folder
set /p query_folder="Enter QUERY folder path: "
if "!query_folder!"=="" (
    echo Error: Query folder cannot be empty!
    pause
    exit /b 1
)
if not exist "!query_folder!" (
    echo Error: Query folder does not exist!
    pause
    exit /b 1
)

REM Ask for Reference folder
set /p ref_folder="Enter REFERENCE folder path: "
if "!ref_folder!"=="" (
    echo Error: Reference folder cannot be empty!
    pause
    exit /b 1
)
if not exist "!ref_folder!" (
    echo Error: Reference folder does not exist!
    pause
    exit /b 1
)

echo.
echo Compiling finalcuda.cu...
nvcc -O3 -diag-suppress=177 -arch=sm_75 finalcuda.cu cpu_utils.c -o finalcuda.exe
if errorlevel 1 (
    echo Error: Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!
echo.
echo Running finalcuda with:
echo   Query folder: !query_folder!
echo   Reference folder: !ref_folder!
echo.

REM Run the program
set "HYRRO_QUERY_FOLDER=!query_folder!"
set "HYRRO_REFERENCE_FOLDER=!ref_folder!"
finalcuda.exe "!query_folder!" "!ref_folder!"

echo.
echo Process completed!
pause
