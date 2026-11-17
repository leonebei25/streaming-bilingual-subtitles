#!/usr/bin/env python3
"""Desktop A/B smoke run.

Builds the same synthetic (or real WAV) stream and the same link trace twice,
runs the A chain (no enhancement) and the B chain (spectral subtraction), and
writes events, metrics and a subtitled MP4 for each. This is the check to run
before packaging anything for the Jetson: it exercises source -> link ->
enhancement -> ASR -> segmentation -> MT -> events -> video in one command.

Examples:
  python scripts/desktop_smoke.py --synth
  python scripts/desktop_smoke.py --wav classroom.wav --script script.json \
      --asr sherpa --asr-model-dir models/sherpa-en --mt opus
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_pipeline


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", action="store_true", help="use the synthetic demo stream")
    ap.add_argument("--wav", help="16 kHz 16-bit PCM WAV")
    ap.add_argument("--script", help="transcript JSON for ScriptedASR")
    ap.add_argument("--outdir", default="runs/desktop_smoke")
    ap.add_argument("--seconds", type=float, default=9.0,
                    help="synthetic stream length (ignored with --wav)")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0)

    ap.add_argument("--link-loss", type=float, default=0.05)
    ap.add_argument("--link-jitter-ms", type=float, default=100.0)
    ap.add_argument("--link-buffer-ms", type=float, default=120.0)
    ap.add_argument("--link-burst", default="")
    ap.add_argument("--link-seed", type=int, default=1)

    ap.add_argument("--asr", choices=["scripted", "sherpa"], default="scripted")
    ap.add_argument("--asr-model-dir")
    ap.add_argument("--asr-provider", default="cpu")
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--mt", choices=["identity", "opus"], default="identity")
    ap.add_argument("--mt-model", default="Helsinki-NLP/opus-mt-en-zh")
    ap.add_argument("--mt-device", default=None)
    ap.add_argument("--font")
    ap.add_argument("--no-video", action="store_true")
    return ap.parse_args(argv)


def common_args(args, outdir: Path):
    base = [
        "--link",
        "--link-loss", str(args.link_loss),
        "--link-jitter-ms", str(args.link_jitter_ms),
        "--link-buffer-ms", str(args.link_buffer_ms),
        "--link-seed", str(args.link_seed),
        "--speed", str(args.speed),
        "--asr", args.asr,
        "--mt", args.mt,
        "--mt-model", args.mt_model,
        "--asr-provider", args.asr_provider,
        "--asr-threads", str(args.asr_threads),
    ]
    if args.link_burst:
        base += ["--link-burst", args.link_burst]
    if args.asr_model_dir:
        base += ["--asr-model-dir", args.asr_model_dir]
    if args.mt_device:
        base += ["--mt-device", args.mt_device]
    if args.font:
        base += ["--font", args.font]
    if args.realtime:
        base += ["--realtime"]
    if args.synth or not args.wav:
        base += ["--synth"]
    else:
        base += ["--wav", args.wav]
        if args.script:
            base += ["--script", args.script]
    return base


def main(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, enhance in (("a", "none"), ("b", "spectral")):
        out_json = outdir / f"{name}.json"
        cmd = common_args(args, outdir) + [
            "--enhance", enhance,
            "--out", str(out_json),
        ]
        if not args.no_video:
            cmd += ["--video-out", str(outdir / f"{name}.mp4")]
        print(f"\n=== chain {name.upper()} ({enhance}) ===")
        run_pipeline.main(cmd)
        with open(out_json, encoding="utf-8") as f:
            payload = json.load(f)
        results[name] = {
            "enhance": enhance,
            "metrics": payload.get("metrics"),
            "video": payload.get("video"),
            "n_events": len(payload.get("events", [])),
        }

    summary = outdir / "summary.json"
    with open(summary, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    for name in ("a", "b"):
        m = results[name]["metrics"] or {}
        print(f"{name.upper()}: WER={m.get('wer')} RTF={m.get('rtf')} "
              f"segments={m.get('n_segments')} gaps={m.get('n_gaps')}")
    print("wrote", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
