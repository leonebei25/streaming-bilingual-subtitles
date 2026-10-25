#!/usr/bin/env python3
"""Run the B-line pipeline on a WAV file (or a synthesized demo signal).

Examples:
  python run_pipeline.py --wav in.wav --script script.json --out events.json
  python run_pipeline.py --wav in.wav --script script.json --loss 5.0-5.4 --realtime --speed 2
  python run_pipeline.py --synth --loss 4.55-4.95          # self-contained demo
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from b_line.audio import load_wav_mono16, save_wav_mono16, synthesize_test_audio
from b_line.asr import ScriptedASR
from b_line.config import BLineConfig
from b_line.metrics import evaluate
from b_line.mt import IdentityMT
from b_line.pipeline import SubtitlePipeline

# Demo transcript used by --synth
DEMO_SCRIPT = [
    (0.5, 1.9, "Welcome to digital signal processing.", "欢迎来到数字信号处理。", 0.95),
    (2.9, 4.3, "Today we study streaming speech recognition.", "今天我们学习流式语音识别。", 0.95),
    (5.3, 6.7, "Please mind the packet loss gaps.", "请注意丢包造成的缺口。", 0.95),
]
DEMO_TOTAL = 9.0


def parse_loss(s: str):
    if not s:
        return None
    out = []
    for part in s.split(","):
        a, b = part.split("-")
        out.append((float(a), float(b)))
    return out


def load_script(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    refs = []
    for u in data["utterances"]:
        entries.append((u["t0"], u["t1"], u["en"], u.get("conf", 0.95)))
        refs.append((u["t0"], u["t1"], u["en"], u.get("zh")))
    return entries, refs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="input 16-bit PCM WAV")
    ap.add_argument("--synth", action="store_true", help="synthesize a demo signal instead of --wav")
    ap.add_argument("--script", help="transcript JSON for ScriptedASR (required unless --synth)")
    ap.add_argument("--out", help="write events+metrics JSON here")
    ap.add_argument("--loss", help="comma-separated t0-t1 loss windows, e.g. 5.0-5.4")
    ap.add_argument("--realtime", action="store_true", help="replay at wall-clock rate (latency meaningful)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--chunk-ms", type=float, default=320.0)
    ap.add_argument("--stability-ms", type=float, default=240.0)
    args = ap.parse_args(argv)

    max_seg = 60 if args.synth else 40
    cfg = BLineConfig(chunk_ms=args.chunk_ms, stability_delay_ms=args.stability_ms, max_seg_chars=max_seg)

    if args.synth:
        utterances = [(t0, t1, 190.0 + 40.0 * i) for i, (t0, t1, _, _, _) in enumerate(DEMO_SCRIPT)]
        x = synthesize_test_audio(utterances, DEMO_TOTAL, sr=cfg.sample_rate, seed=7)
        save_wav_mono16("sample.wav", x, cfg.sample_rate)
        sr = cfg.sample_rate
        entries = [(t0, t1, en, conf) for (t0, t1, en, zh, conf) in DEMO_SCRIPT]
        refs = [(t0, t1, en, zh) for (t0, t1, en, zh, conf) in DEMO_SCRIPT]
        print("wrote sample.wav (synthesized demo)")
    else:
        if not args.wav:
            ap.error("provide --wav or --synth")
        x, sr = load_wav_mono16(args.wav, cfg.sample_rate)
        if args.script:
            entries, refs = load_script(args.script)
        else:
            entries, refs = [], None

    asr = ScriptedASR(entries, cfg)
    mt = IdentityMT()
    pipe = SubtitlePipeline(cfg, asr, mt)

    t0 = time.perf_counter()
    events = pipe.run(x, sr, loss_windows=parse_loss(args.loss),
                      realtime=args.realtime, speed=args.speed)
    proc = time.perf_counter() - t0

    met = evaluate(events, references=refs, duration_sec=pipe.duration_sec,
                   processing_sec=proc, realtime=args.realtime)

    print()
    for e in events:
        print(json.dumps(e.to_dict(), ensure_ascii=False))
    print("---")
    print("config_hash:", cfg.freeze())
    print("metrics:", json.dumps(met, ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "config": cfg.to_dict(),
                "config_hash": cfg.freeze(),
                "events": [e.to_dict() for e in events],
                "metrics": met,
            }, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
