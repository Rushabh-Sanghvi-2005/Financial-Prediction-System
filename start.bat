@echo off
setlocal EnableDelayedExpansion
title Ultimate Universal Launcher

echo ===============================================================================
echo                           UNIVERSAL LAUNCHER
echo ===============================================================================
echo.

REM --- CONFIGURATION ---
set "MIN_PY_VER_MAJOR=3"
set "MIN_PY_VER_MINOR=9"
REM Update this with your repo ZIP!
set "GITHUB_REPO_ZIP=https://github.com/Rushabh-Sanghvi-2005/Financial-Prediction-System/archive/refs/heads/main.zip"

REM --- STEP 1: CHECK PYTHON ---
echo [CHECK] Looking for Python...

REM A. Check PATH
python --version >NUL 2>&1
if %errorlevel% equ 0 (
    python -c "import sys; exit(0 if sys.version_info >= (%MIN_PY_VER_MAJOR%, %MIN_PY_VER_MINOR%) else 1)" >NUL 2>&1
    if !errorlevel! equ 0 (
        echo [OK] System Python found.
        set "PYTHON_CMD=python"
        goto :CHECK_BOOTSTRAP
    )
)

REM B. Check Portable
if exist "python_embed\python.exe" (
    echo [OK] Portable Python found.
    set "PYTHON_CMD=python_embed\python.exe"
    goto :CHECK_BOOTSTRAP
)

REM C. Install Portable (Last Resort)
echo [MISSING] No compatible Python found.
echo [INSTALL] Generating Portable Installer...
call :GENERATE_INSTALLER
echo [INSTALL] Running Installer (Please Wait)...
powershell -ExecutionPolicy Bypass -File install_portable.ps1
if %errorlevel% neq 0 (
    echo [ERROR] Installation failed.
    pause
    exit /b 1
)
set "PYTHON_CMD=python_embed\python.exe"

REM --- STEP 2: CHECK BOOTSTRAPPER ---
:CHECK_BOOTSTRAP
if exist "bootstrap.py" (
    echo [OK] Project files found locally.
    goto :RUN
)

echo.
echo [MISSING] Project files not found.
echo [DOWNLOAD] Downloading from Cloud Repository...
echo URL: %GITHUB_REPO_ZIP%
echo.

REM Generate Downloader Script
(
echo $url = "%GITHUB_REPO_ZIP%"
echo $zip = "repo.zip"
echo Write-Host "Downloading..."
echo try { Invoke-WebRequest -Uri $url -OutFile $zip } catch { Write-Error $_; exit 1 }
echo Write-Host "Extracting..."
echo Expand-Archive -Path $zip -DestinationPath . -Force
echo $extracted = Get-ChildItem -Directory ^| Sort-Object LastWriteTime -Descending ^| Select -First 1
echo Write-Host "Detected Folder: $($extracted.Name)"
echo Get-ChildItem -Path $extracted.FullName ^| Move-Item -Destination . -Force
echo Remove-Item $extracted.FullName -Recurse -Force
echo Remove-Item $zip -Force
) > download_repo.ps1

powershell -ExecutionPolicy Bypass -File download_repo.ps1
if %errorlevel% neq 0 (
    echo [ERROR] Download failed. Check URL or Internet.
    pause
    del download_repo.ps1
    exit /b 1
)
del download_repo.ps1
echo [OK] Project Restored.

REM --- STEP 3: RUN ---
:RUN
echo.
echo [LAUNCH] Starting Application...
"%PYTHON_CMD%" bootstrap.py
pause
exit /b

REM --- SUBROUTINE: GENERATE INSTALLER ---
:GENERATE_INSTALLER
(
echo $ErrorActionPreference = "Stop"
echo $pythonVersion = "3.11.9"
echo $zipName = "python-$pythonVersion-embed-amd64.zip"
echo $downloadUrl = "https://www.python.org/ftp/python/$pythonVersion/$zipName"
echo $extractPath = "python_embed"
echo $getPipUrl = "https://bootstrap.pypa.io/get-pip.py"
echo Write-Host "[1/2] Downloading Python..."
echo if ^(-not ^(Test-Path $zipName^)^) { Invoke-WebRequest -Uri $downloadUrl -OutFile $zipName }
echo Write-Host "[2/2] Extracting..."
echo Expand-Archive -Path $zipName -DestinationPath $extractPath -Force
echo $pth = Get-ChildItem -Path $extractPath -Filter "*._pth" ^| Select -First 1
echo $c = Get-Content $pth.FullName; $n = $c -replace "#import site", "import site"; Set-Content $pth.FullName $n
echo Invoke-WebRequest -Uri $getPipUrl -OutFile "get-pip.py"
echo ^& "$extractPath\python.exe" get-pip.py --no-warn-script-location
echo Remove-Item "get-pip.py"
echo Remove-Item $zipName
) > install_portable.ps1
exit /b
