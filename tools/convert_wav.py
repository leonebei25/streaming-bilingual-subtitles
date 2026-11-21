#!/usr/bin/env python3
"""Convert PCM WAV files to 16 kHz mono 16-bit PCM.

Examples:
  python tools/convert_wav.py in.wav
  python tools/convert_wav.py in.wav -o out.wav
  python tools/convert_wav.py video_test --out-dir video_test/16k --recursive
  python tools/convert_wav.py in.wav --keep-stereo --normalize
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from a_line.audio_io import _cli

if __name__ == "__main__":
    raise SystemExit(_cli())
