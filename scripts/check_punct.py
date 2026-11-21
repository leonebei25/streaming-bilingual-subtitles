#!/usr/bin/env python3
"""Check a punctuation model without running the whole pipeline.

Loads the model, punctuates a few sentences, and reports per-call latency so
the real-time budget can be judged before the cuts are wired into the live
display.

Examples:
  python scripts/check_punct.py --model models/sherpa-onnx-online-punct-en-2024-08-06
  python scripts/check_punct.py --model path/to/model.onnx --mode offline
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from b_line.punct import PunctRestorer

SAMPLES = [
    "SO NOW WE SAMPLE THE SIGNAL AND RECONSTRUCT IT",
    "THE FOURIER TRANSFORM TELLS US WHICH FREQUENCIES ARE PRESENT",
    "THIS IS IMPORTANT BECAUSE ALIASING CANNOT BE UNDONE LATER",
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True,
                    help="model directory or a single .onnx file")
    ap.add_argument("--mode", choices=["auto", "online", "offline"],
                    default="auto", help="force a model family (default auto)")
    ap.add_argument("--text", action="append",
                    help="text to punctuate (repeatable)")
    ap.add_argument("--repeat", type=int, default=5,
                    help="timing repetitions per text (default 5)")
    args = ap.parse_args(argv)

    punct = PunctRestorer(args.model, mode=args.mode)
    print(f"kind: {punct.kind}")
    print(f"model: {punct.model_path}")
    if punct.bpe_vocab:
        print(f"bpe_vocab: {punct.bpe_vocab}")

    repeat = max(1, args.repeat)
    for text in (args.text or SAMPLES):
        out = punct.apply(text)          # apply(), not restore(), so errors show
        t0 = time.perf_counter()
        for _ in range(repeat):
            punct.apply(text)
        dt = (time.perf_counter() - t0) / repeat
        print()
        print("in :", text)
        print("out:", out)
        print(f"     {dt * 1000:.1f} ms/call ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
