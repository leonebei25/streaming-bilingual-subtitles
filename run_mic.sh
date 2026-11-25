#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_mic.sh - live microphone subtitle demo on Linux / Jetson.
#
# The bash counterpart of run_mic.bat. Speech from the microphone (or a
# PulseAudio monitor source, i.e. the sound the machine is playing) becomes
# on-screen English + Chinese subtitles in real time.
#
# Usage:
#   ./run_mic.sh                        mic -> on-screen subtitles
#   ./run_mic.sh devices                list input devices, then exit
#   ./run_mic.sh monitor                capture what the machine plays
#                                       (PulseAudio/PipeWire loopback)
#   ./run_mic.sh --video x.mp4          also play the video in a window
#   ./run_mic.sh --max-seconds 60       stop after 60 s
#   ./run_mic.sh --headless             no windows (SSH / smoke test)
#
# Any argument that is not one of the commands above is passed straight
# through to scripts/live_demo.py.
#
# Environment:
#   MT=ct2|opus|auto   translation backend (default: auto -- CTranslate2 when
#                      models/opus-mt-en-zh-ct2 exists, else opus)
#   PYTHON=...         interpreter to use (default: .venv, then python3)
#   PCM_SOURCE=...     monitor source name for the "monitor" command
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    if [ -x ".venv/bin/python" ]; then
        PY=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PY="python3"
        echo "[run_mic] no .venv/bin/python found; using $(command -v python3)" >&2
    else
        echo "[run_mic] no python3 and no .venv; create one with:" >&2
        echo "  python3 -m venv --system-site-packages .venv" >&2
        exit 1
    fi
fi

DEMO="scripts/live_demo.py"
MT_MODE="${MT:-auto}"

# No display: in a windowed run this would otherwise die inside cv2.imshow.
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
    case "${1:-}" in
        --headless|devices|monitor) ;;
        *)
            echo "[run_mic] no DISPLAY/WAYLAND_DISPLAY: the subtitle window needs"
            echo "           a desktop session. Add --headless to run without it,"
            echo "           or run this from the Jetson's local desktop." >&2
            ;;
    esac
fi

case "${1:-}" in
    devices)
        # sounddevice enumerates PortAudio devices; on PulseAudio that may not
        # include monitor sources, hence the hint below.
        "$PY" "$DEMO" --mic-list || true
        echo
        echo "PulseAudio / PipeWire sources (the '.monitor' one is system audio):"
        if command -v pactl >/dev/null 2>&1; then
            pactl list short sources || true
        else
            echo "  pactl not found; try: arecord -l" >&2
        fi
        exit 0
        ;;
    monitor)
        shift
        if ! command -v parec >/dev/null 2>&1; then
            echo "[run_mic] parec not found. Install pulseaudio-utils:" >&2
            echo "  sudo apt install pulseaudio-utils" >&2
            exit 1
        fi
        SRC="${PCM_SOURCE:-}"
        if [ -z "$SRC" ]; then
            # First monitor source that is not a ".monitor" of a null sink.
            SRC="$(pactl list short sources 2>/dev/null \
                   | awk '$2 ~ /\.monitor$/ {print $2; exit}')"
        fi
        if [ -z "$SRC" ]; then
            echo "[run_mic] no monitor source found. Use a microphone instead" >&2
            echo "           (./run_mic.sh) or list sources: ./run_mic.sh devices" >&2
            exit 1
        fi
        echo "[run_mic] capturing system audio from $SRC"
        # parec feeds the existing PCM-stdin path: same streaming contract as
        # --mic, no PortAudio in between. pipefail makes the pipeline status
        # reflect either end; errexit is paused so we can report it ourselves.
        set +e
        parec --device="$SRC" --format=s16le --rate=16000 --channels=1 \
             --latency-msec=40 \
            | "$PY" "$DEMO" --pcm-stdin --mt "$MT_MODE" "$@"
        rc=$?
        set -e
        exit "$rc"
        ;;
esac

exec "$PY" "$DEMO" --mic --mt "$MT_MODE" "$@"
