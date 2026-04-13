@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"

echo.
echo  DynSonOpt -- PyInstaller build
echo  ================================
echo.

:: ── step 1: get the JRE ─────────────────────────────────────────────────────
::
::  We bundle Eclipse Temurin 21 LTS (JRE, not full JDK — much smaller).
::  The Adoptium API URL always resolves to the latest GA release.

:: venv, jre and dist all live one level up (project root, not SonamicCode\)
set "PROJECT=%ROOT%\.."

set "JRE_DIR=%PROJECT%\jre"
set "JRE_ZIP=%PROJECT%\jre.zip"
set "JRE_URL=https://api.adoptium.net/v3/binary/latest/21/ga/windows/x64/jre/hotspot/normal/eclipse"

if exist "%JRE_DIR%\bin\java.exe" (
    echo  [JRE] Found bundled JRE at jre\  --  skipping download.
    echo.
) else (
    if not exist "%JRE_ZIP%" (
        echo  [JRE] Downloading Eclipse Temurin 21 JRE ...
        echo        ^(this is ~55 MB, please wait^)
        echo.
        powershell -NoProfile -Command ^
            "Invoke-WebRequest -Uri '%JRE_URL%' -OutFile '%JRE_ZIP%' -UseBasicParsing"
        if errorlevel 1 (
            echo.
            echo  ERROR: Download failed.  Check your internet connection and try again.
            pause
            exit /b 1
        )
        echo  [JRE] Download complete.
        echo.
    ) else (
        echo  [JRE] Found jre.zip  --  skipping download.
        echo.
    )

    echo  [JRE] Extracting jre.zip ...
    powershell -NoProfile -Command ^
        "Expand-Archive -Path '%JRE_ZIP%' -DestinationPath '%PROJECT%\_jre_tmp' -Force"
    if errorlevel 1 (
        echo.
        echo  ERROR: Extraction failed.
        pause
        exit /b 1
    )

    :: The zip contains a single top-level folder (e.g. jdk-21.0.x+y-jre)
    :: Rename it to just "jre" so the spec and runtime hook can find it
    for /d %%D in ("%PROJECT%\_jre_tmp\*") do (
        move "%%D" "%JRE_DIR%" >nul
    )
    rmdir "%PROJECT%\_jre_tmp" 2>nul

    echo  [JRE] Extracted to jre\
    echo.
)

:: ── step 2: run PyInstaller ──────────────────────────────────────────────────

echo  [BUILD] Running PyInstaller ...
echo.

"%PROJECT%\.venv\Scripts\pyinstaller.exe" "%ROOT%\DynSonOpt.spec" --clean

if errorlevel 1 (
    echo.
    echo  Build FAILED -- check the output above.
    pause
    exit /b 1
)

echo.
echo  ============================================================
echo   Build complete!
echo   Output:  %PROJECT%\dist\DynSonOpt\DynSonOpt.exe
echo.
echo   The app is fully self-contained -- no Java install needed
echo   on any machine you copy the dist\DynSonOpt\ folder to.
echo  ============================================================
echo.
pause
endlocal
