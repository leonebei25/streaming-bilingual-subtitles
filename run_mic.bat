@echo off
rem ---------------------------------------------------------------------------
rem  run_mic.bat - start the live microphone subtitle demo (Kroko ASR + Opus MT)
rem
rem  Usage:
rem      run_mic.bat                       live mic -> on-screen subtitles
rem      run_mic.bat --video x.mp4         also play the video in a window
rem      run_mic.bat --max-seconds 60      stop after 60 s
rem      run_mic.bat --punct-model models\sherpa-onnx-online-punct-en-2024-08-06
rem      run_mic.bat devices               list input devices, then exit
rem
rem  Any extra arguments are passed straight through to scripts\live_demo.py.
rem ---------------------------------------------------------------------------
setlocal
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [run_mic] No .venv found. Run activate.bat once to create it.
    exit /b 1
)
call ".venv\Scripts\activate.bat" >nul

if /i "%~1"=="devices" (
    python scripts\live_demo.py --mic-list
    exit /b %errorlevel%
)

python scripts\live_demo.py --mic --mt opus --mt-model models\opus-mt-en-zh %*
exit /b %errorlevel%
