@echo off
rem Activate the project virtual environment from cmd.exe.
rem Usage:  activate.bat
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo No virtual environment at .venv; creating it...
    py -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\activate.bat" (
        echo Failed to create .venv. Install Python 3.12 first.
        exit /b 1
    )
    echo Created .venv. Install dependencies before running the pipeline.
)
call ".venv\Scripts\activate.bat"
echo venv active.
python -c "import sys; print('python:', sys.executable)"
