#!/usr/bin/env bash
# Activate the project virtual environment from Git Bash / WSL-on-Windows.
#
# Usage:
#   source ./activate.sh
#
# The leading "source" (or ". ./activate.sh") is required: running the script
# as a child process would activate the venv only inside that subprocess.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -d ".venv" ]; then
    echo "No virtual environment at $ROOT/.venv; creating it..."
    if command -v py >/dev/null 2>&1; then
        py -m venv .venv
    elif command -v python >/dev/null 2>&1; then
        python -m venv .venv
    else
        echo "Neither 'py' nor 'python' found on PATH; install Python 3.12 first." >&2
        exit 1
    fi
    echo "Created .venv. Install dependencies before running the pipeline:"
    echo "  python -m pip install -r requirements-desktop.txt"
fi

# Git Bash on Windows keeps venv scripts under Scripts/, POSIX under bin/.
if [ -f ".venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "No activate script found under .venv (Scripts/ or bin/)." >&2
    exit 1
fi

echo "venv active: $(command -v python)"
echo "project:    $ROOT"
