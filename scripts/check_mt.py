#!/usr/bin/env python3
"""Load OpusMT, report the real device, and time a few English sentences.

This is the translation half of the smoke test. It answers three questions at
once: does the local model directory load, which device is actually used, and
how many milliseconds per sentence does that cost.

Examples:
  # desktop / CPU baseline
  python scripts/check_mt.py

  # Jetson, after the CUDA torch wheel is installed
  python scripts/check_mt.py --device cuda

  # compare beam settings
  python scripts/check_mt.py --device cuda --beams 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SENTENCES = [
    "Welcome to digital signal processing.",
    "Today we study streaming speech recognition.",
    "Please mind the packet loss gaps.",
    "The instructor will now derive the Wiener filter step by step.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="models/opus-mt-en-zh",
                    help="local model directory or hub id")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the sentence list this many times; later rounds "
                         "exclude load time and warm-up")
    ap.add_argument("--sentence", action="append",
                    help="use this sentence instead of the built-in list "
                         "(repeatable)")
    args = ap.parse_args(argv)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"torch is not installed: {exc}")

    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "this torch build has no CUDA (the version above ends in +cpu, or "
            "the driver is missing).\n"
            "CUDA can only be tested on the Jetson, with NVIDIA's wheel:\n"
            "  pip install torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126\n"
            "On this machine run without --device to get the CPU baseline."
        )
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    from b_line.mt import OpusMT

    t0 = time.perf_counter()
    try:
        mt = OpusMT(
            model_name=args.model,
            device=args.device,
            max_length=args.max_length,
            num_beams=args.beams,
        )
    except Exception as exc:
        raise SystemExit(f"cannot build OpusMT: {exc}")
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.2f}s  device={mt._device}  beams={mt.num_beams}")

    sentences = args.sentence or SENTENCES
    times = []
    for r in range(max(1, args.repeat)):
        for s in sentences:
            t = time.perf_counter()
            zh, conf = mt.translate(s)
            dt = time.perf_counter() - t
            times.append(dt)
            if r == 0 or args.repeat == 1:
                print(f"{dt * 1000:7.0f} ms  conf={conf:.3f}")
                print(f"     en: {s}")
                print(f"     zh: {zh or '<empty>'}")
            else:
                print(f"{dt * 1000:7.0f} ms  {zh}")
        if r == 0 and args.repeat > 1:
            times.clear()
            print("-- round 1 (includes warm-up) done, timing round 2+ --")

    if times:
        mean = sum(times) / len(times)
        worst = max(times)
        print(f"\nmean {mean * 1000:.0f} ms  max {worst * 1000:.0f} ms "
              f"over {len(times)} sentences")
        if mean > 0.5:
            print("WARNING: over 500 ms per sentence -- on-device this eats the "
                  "latency budget; try --beams 1 or --device cuda")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
