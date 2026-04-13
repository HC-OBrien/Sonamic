@echo off
setlocal

:: ── locate this script's directory ──────────────────────────────────────────
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

:: ── check Java is on the PATH ────────────────────────────────────────────────
where java >nul 2>&1
if errorlevel 1 (
    echo ERROR: Java not found.  Please install a JDK and add it to PATH.
    pause
    exit /b 1
)

:: ── activate the virtual environment ────────────────────────────────────────
:: venv lives one level up at the project root
call "%ROOT%\..\.venv\Scripts\activate.bat"

:: ── launch the app ──────────────────────────────────────────────────────────
python "%ROOT%\Sonamic.py"

if errorlevel 1 (
    echo.
    echo  App exited with an error -- see above.
    pause
)

endlocal
