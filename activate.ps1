# Activate the project virtual environment (PowerShell).
#
# Usage (the leading dot matters - it keeps the venv active in your shell):
#   . .\activate.ps1
#
# If .venv is missing, it is created with py/python and dependencies are NOT
# installed automatically; run:
#   python -m pip install -r requirements-desktop.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
if (-not $root) { $root = (Get-Location).Path }
Set-Location $root

$venv = Join-Path $root ".venv"
$activate = Join-Path $venv "Scripts\Activate.ps1"

if (-not (Test-Path $activate)) {
    Write-Host "No virtual environment at $venv; creating it..." -ForegroundColor Yellow
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        & py -m venv $venv
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if (-not $python) { throw "Neither 'py' nor 'python' is on PATH; install Python 3.12 first." }
        & python -m venv $venv
    }
    $activate = Join-Path $venv "Scripts\Activate.ps1"
    if (-not (Test-Path $activate)) { throw "venv creation failed: $activate not found" }
    Write-Host "Created $venv. Install dependencies before running the pipeline." -ForegroundColor Yellow
}

. $activate

$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
Write-Host "venv active: $pythonPath" -ForegroundColor Green
Write-Host "project:    $root" -ForegroundColor Green
