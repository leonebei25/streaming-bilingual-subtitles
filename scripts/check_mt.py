#!/usr/bin/env python3
"""Load the MT backend, report the real device, and time a few sentences.

Two backends are covered:

  ct2   CTranslate2 (int8 by default). This is the Jetson deployment path:
        no torch at inference time, only ctranslate2 plus the Marian
        tokenizer. The model must be converted once with
        scripts/convert_mt_ct2.py.
  opus  torch + transformers MarianMT. The desktop baseline, and the
        reference the ct2 numbers are compared against.

``--backend auto`` (default) picks ct2 when models/opus-mt-en-zh-ct2 exists,
otherwise opus. It answers three questions at once: does the model directory
load, which device is actually used, and how many milliseconds per sentence
does that cost.

Examples:
  python scripts/check_mt.py                             # auto on this machine
  python scripts/check_mt.py --backend ct2 --threads 6   # Jetson, 6 CPU cores
  python scripts/check_mt.py --backend ct2 --compute-type float32
  python scripts/check_mt.py --backend opus --device cuda --beams 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SENTENCES = [
    "Welcome to digital signal processing.",
    "Today we study streaming speech recognition.",
    "Please mind the packet loss gaps.",
    "The instructor will now derive the Wiener filter step by step.",
]


def _default_model(backend: str) -> str:
    if backend == "ct2":
        return str(ROOT / "models" / "opus-mt-en-zh-ct2")
    return str(ROOT / "models" / "opus-mt-en-zh")


def _resolve_backend(choice: str) -> str:
    if choice != "auto":
        return choice
    if Path(_default_model("ct2")).is_dir():
        return "ct2"
    return "opus"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backend", choices=["auto", "ct2", "opus"], default="auto")
    ap.add_argument("--model", default=None,
                    help="model directory (default: models/opus-mt-en-zh-ct2 "
                         "for ct2, models/opus-mt-en-zh for opus)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--compute-type", default="int8",
                    help="ct2 only: int8, int8_float32, float32, ... "
                         "(default: int8)")
    ap.add_argument("--threads", type=int, default=0,
                    help="ct2 only: intra-op threads (0 = let CTranslate2 "
                         "decide; use the physical core count on the Jetson)")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the sentence list this many times; later rounds "
                         "exclude load time and warm-up")
    ap.add_argument("--sentence", action="append",
                    help="use this sentence instead of the built-in list "
                         "(repeatable)")
    args = ap.parse_args(argv)

    backend = _resolve_backend(args.backend)
    model = args.model or _default_model(backend)

    if backend == "ct2":
        try:
            import ctranslate2
        except ImportError as exc:
            raise SystemExit(
                f"ctranslate2 is not installed: {exc}\n"
                "  pip install ctranslate2 transformers sentencepiece"
            )
        print(f"ctranslate2 {ctranslate2.__version__}  "
              f"device={args.device}  compute_type={args.compute_type}")
        from b_line.mt import CT2MT

        t0 = time.perf_counter()
        try:
            mt = CT2MT(
                model_dir=model,
                device=args.device,
                compute_type=args.compute_type,
                beam_size=args.beams,
                max_decoding_length=args.max_length,
                intra_threads=args.threads,
            )
        except Exception as exc:
            raise SystemExit(f"cannot build CT2MT: {exc}")
        load_s = time.perf_counter() - t0
        print(f"model loaded in {load_s:.2f}s  dir={model}")
        print(f"beam={mt.beam_size}  threads={mt.intra_threads or 'auto'}")
    else:
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(f"torch is not installed: {exc}")
        print(f"torch {torch.__version__}  "
              f"cuda_available={torch.cuda.is_available()}")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "this torch build has no CUDA (the version above ends in +cpu, "
                "or the driver is missing).\n"
                "On the Jetson use NVIDIA's wheel, e.g.:\n"
                "  pip install torch --index-url "
                "https://pypi.jetson-ai-lab.io/jp6/cu126\n"
                "Or use the ct2 backend, which needs no torch at all."
            )
        if torch.cuda.is_available():
            print(f"gpu: {torch.cuda.get_device_name(0)}")

        from b_line.mt import OpusMT

        t0 = time.perf_counter()
        try:
            mt = OpusMT(
                model_name=model,
                device=args.device,
                max_length=args.max_length,
                num_beams=args.beams,
            )
        except Exception as exc:
            raise SystemExit(f"cannot build OpusMT: {exc}")
        load_s = time.perf_counter() - t0
        print(f"model loaded in {load_s:.2f}s  device={mt._device}  "
              f"beams={mt.num_beams}  dir={model}")

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
                  "latency budget; try --beams 1 or a smaller/quantized model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
