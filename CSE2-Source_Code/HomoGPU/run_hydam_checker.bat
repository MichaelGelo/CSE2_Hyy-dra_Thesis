@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   HomoGPU Launcher (homocuda.cu)
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
echo Compiling homocuda.cu...
nvcc -O3 -diag-suppress=177 -arch=sm_75 homocuda.cu -o homocuda.exe
if errorlevel 1 (
    echo Error: Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!
echo.
echo Running homocuda with:
echo   Query folder: !query_folder!
echo   Reference folder: !ref_folder!
echo.

REM Run the program
homocuda.exe "!query_folder!" "!ref_folder!"

echo.
echo Process completed!
pause
