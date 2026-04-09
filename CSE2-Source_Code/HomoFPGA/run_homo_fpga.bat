@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo         HomoFPGA Launcher (homoFPGA.c)
echo ============================================
echo.

REM Ask for Query folder
set /p query_folder="Enter QUERY folder path: "
if "!query_folder!"=="" (
    echo Error: Query folder cannot be empty!
    pause
    exit /b 1
)
if not exist "!query_folder!\" (
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
if not exist "!ref_folder!\" (
    echo Error: Reference folder does not exist!
    pause
    exit /b 1
)

echo.
echo Compiling homoFPGA...
gcc homoFPGA.c -o homoFPGA.exe -std=c99 -Wall -Wextra
if errorlevel 1 (
    echo Error: Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!
echo.
echo Running homoFPGA with:
echo   Query folder: !query_folder!
echo   Reference folder: !ref_folder!
echo.

REM Run the program
homoFPGA.exe "!query_folder!" "!ref_folder!"

echo.
echo Process completed!
pause
