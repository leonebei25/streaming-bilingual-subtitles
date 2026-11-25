#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# jetson_setup.sh - one-shot setup / verification for running the bilingual
# subtitle demo on a Jetson (Linux for Tegra, aarch64).
#
# Usage (from the project root on the Jetson):
#   ./scripts/jetson_setup.sh info     # what system is this? (no changes)
#   ./scripts/jetson_setup.sh deps     # apt + venv + pip installs
#   ./scripts/jetson_setup.sh check    # imports, model dirs, devices
#   ./scripts/jetson_setup.sh          # info + deps + check
#
# Design notes:
#  * OpenCV comes from apt (python3-opencv). PyPI opencv-python has no
#    aarch64 wheel, so `pip install opencv-python` would try to build from
#    source for ~an hour. The venv is created with --system-site-packages so
#    the apt cv2 is visible.
#  * numpy is pinned <2 because the apt-built cv2 was compiled against numpy
#    1.x; numpy 2 raises "compiled using NumPy 1.x cannot be run in NumPy 2".
#  * torch is NOT installed: the demo uses the CTranslate2 MT backend
#    (models/opus-mt-en-zh-ct2), which needs only ctranslate2 + transformers
#    + sentencepiece. NVIDIA's torch wheel is a separate, much bigger install.
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")/.."
ROOT="$PWD"

PY="${PYTHON:-}"
if [ -z "$PY" ] && [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
fi
if [ -z "$PY" ]; then
    PY="python3"
fi

say() { printf '\n=== %s ===\n' "$*"; }

do_info() {
    say "system"
    if [ -r /etc/nv_tegra_release ]; then
        head -1 /etc/nv_tegra_release
    else
        echo "not a Jetson / no /etc/nv_tegra_release"
    fi
    . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
    echo "kernel : $(uname -r)"
    echo "arch   : $(uname -m)"
    echo "python3: $(python3 -V 2>&1)"
    echo "cores  : $(nproc)"
    free -h | head -2
    df -h "$ROOT" | tail -1

    say "network"
    if ping -c1 -W2 192.168.55.100 >/dev/null 2>&1 || \
       getent hosts archive.ubuntu.com >/dev/null 2>&1; then
        echo "internet/DNS: looks OK"
    else
        echo "WARNING: no DNS/internet. pip and apt will fail."
        echo "         Configure WiFi first (README-wifi):"
        echo "           sudo nmcli device wifi connect 'SSID' password 'PASS'"
    fi
}

do_deps() {
    say "apt packages"
    sudo apt-get update
    sudo apt-get install -y \
        python3-venv python3-pip python3-opencv \
        libportaudio2 libgl1 libglib2.0-0 \
        ffmpeg fonts-noto-cjk pulseaudio-utils

    say "virtualenv"
    if [ ! -x ".venv/bin/python" ]; then
        python3 -m venv --system-site-packages .venv
    else
        echo ".venv already exists"
    fi
    PY=".venv/bin/python"
    "$PY" -m pip install --upgrade pip setuptools wheel

    say "pip packages"
    # numpy<2: apt cv2 is built against numpy 1.x (see header note).
    "$PY" -m pip install "numpy<2" "scipy>=1.10" "pillow>=9.5" \
        "sounddevice>=0.5" "sherpa-onnx>=1.10,<3" \
        "transformers>=4.40" "sentencepiece>=0.1.99" "sacremoses>=0.1.1"

    # CTranslate2 4.x needs Python >= 3.9; JetPack 5 (Ubuntu 20.04) has 3.8.
    if "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,9) else 1)'; then
        "$PY" -m pip install "ctranslate2>=4.0"
    else
        echo "python < 3.9: installing ctranslate2 3.x"
        "$PY" -m pip install "ctranslate2>=3.24,<4"
    fi
}

do_check() {
    [ -x ".venv/bin/python" ] && PY=".venv/bin/python"
    say "imports ($PY)"
    "$PY" - <<'EOF'
import importlib, sys
mods = ["numpy", "scipy", "cv2", "PIL", "sounddevice",
        "sherpa_onnx", "ctranslate2", "transformers", "sentencepiece"]
bad = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  ok   {m:16s} {getattr(mod, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        bad.append(m)
        print(f"  FAIL {m:16s} {type(e).__name__}: {e}")
print("python:", sys.version.split()[0])
raise SystemExit(1 if bad else 0)
EOF

    say "model directories"
    "$PY" - <<'EOF'
from pathlib import Path
need = {
    "models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06": ["encoder", "decoder", "joiner", "tokens"],
    "models/opus-mt-en-zh-ct2": ["model.bin", "source.spm", "target.spm", "vocab.json"],
    "models/sherpa-onnx-online-punct-en-2024-08-06": ["model", "bpe.vocab"],
}
missing = 0
for d, pats in need.items():
    p = Path(d)
    if not p.is_dir():
        print(f"  MISSING dir  {d}")
        missing += 1
        continue
    for pat in pats:
        hits = list(p.glob(pat + "*"))
        if not hits:
            print(f"  MISSING file {d}/{pat}*")
            missing += 1
    if missing == 0:
        print(f"  ok           {d}")
raise SystemExit(1 if missing else 0)
EOF

    say "audio devices"
    "$PY" scripts/live_demo.py --mic-list || true
    if command -v pactl >/dev/null 2>&1; then
        echo
        pactl list short sources || true
    fi

    say "MT timing"
    "$PY" scripts/check_mt.py --backend ct2 --threads "$(nproc)" || true

    say "next"
    echo "  ./run_mic.sh devices     # pick a device"
    echo "  ./run_mic.sh             # mic -> subtitles (needs a desktop session)"
}

case "${1:-all}" in
    info)  do_info ;;
    deps)  do_deps ;;
    check) do_check ;;
    all)   do_info; do_deps; do_check ;;
    *)     echo "usage: $0 [info|deps|check]" >&2; exit 2 ;;
esac
