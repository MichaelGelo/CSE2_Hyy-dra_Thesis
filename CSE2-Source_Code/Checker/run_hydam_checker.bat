@echo off
setlocal enabledelayedexpansion

echo ============================================
echo   HyDam Checker - Interactive Launcher
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
echo Compiling HyDam_checker...
gcc [HyDam_checker].c -o [HyDam_checker].exe
if errorlevel 1 (
    echo Error: Compilation failed!
    pause
    exit /b 1
)

echo Compilation successful!
echo.
echo Running HyDam_checker with:
echo   Query folder: !query_folder!
echo   Reference folder: !ref_folder!
echo.

REM Run the program
REM Note: Update the code to accept command line arguments if needed
[HyDam_checker].exe "!query_folder!" "!ref_folder!"

echo.
echo Process completed!
pause
