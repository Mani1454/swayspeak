@echo on
setlocal enabledelayedexpansion

echo ============================================
echo        AURASPEAK - Starting Up...
echo ============================================
echo.
echo Press any key to continue starting...
pause >nul

REM Ensure we're at repo root
pushd %~dp0

if not exist environment.yml (
  echo [ERROR] environment.yml not found in %~dp0
  goto :error
)

REM Initialize Conda
echo [INFO] Initializing Conda...
set CONDA_PATH=%USERPROFILE%\miniconda3
if not exist "%CONDA_PATH%\Scripts\activate.bat" (
  echo [ERROR] Conda not found at %CONDA_PATH%
  echo Please install Miniconda or set CONDA_PATH correctly.
  goto :error
)

call "%CONDA_PATH%\Scripts\activate.bat" "%CONDA_PATH%"
if errorlevel 1 (
  echo [ERROR] Failed to initialize Conda.
  goto :error
)

echo [INFO] Creating/updating conda environment "auraspeak"...
call conda env update --file environment.yml --name auraspeak --prune
if errorlevel 1 (
  echo [INFO] Environment may not exist, creating new one...
  call conda env create --file environment.yml --name auraspeak
  if errorlevel 1 (
    echo [ERROR] Failed to create/update conda environment.
    goto :error
  )
)

call conda activate auraspeak
if errorlevel 1 (
  echo [ERROR] Failed to activate conda environment.
  goto :error
)

echo [INFO] Installing/updating Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Pip install failed.
  goto :error
)

echo [INFO] Starting server at http://127.0.0.1:8000 ...
pushd code
python server.py
popd

goto :done

:error
echo.
echo ============================================
echo        AN ERROR OCCURRED - SEE ABOVE
echo ============================================
echo.
pause
popd
endlocal
exit /b 1

:done
echo.
echo ============================================
echo        Server stopped.
echo ============================================
pause
popd
endlocal
