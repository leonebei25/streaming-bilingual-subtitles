#!/usr/bin/env python3
"""Run the A/B subtitle pipeline on an audio stream.

A and B share one path and differ only in `--enhance`:
  A: --enhance none
  B: --enhance spectral (or wiener)

Examples:
  python run_pipeline.py --synth --enhance none --link --link-loss 0.05
  python run_pipeline.py --synth --enhance spectral --link --link-loss 0.05
  python run_pipeline.py --wav in.wav --script script.json --asr sherpa \
      --asr-model-dir models/sherpa-en --mt opus --enhance spectral \
      --video-out out.mp4 --out run.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from a_line.enhance import (
    BypassEnhancer,
    SpectralSubtractionEnhancer,
)
from a_line.enhance.source import EnhancedFrameSource
from a_line.link import LinkConfig, LinkSource
from a_line.video import make_test_video, render_subtitles
from b_line.asr import ScriptedASR, SherpaASR
from b_line.config import BLineConfig
from b_line.metrics import evaluate
from b_line.mt import IdentityMT, OpusMT
from b_line.pipeline import SubtitlePipeline
from b_line.sources import SyntheticFrameSource, WavFrameSource

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


def build_source(args, cfg, loss):
    """Return (source, script_entries, references) for the requested input."""
    if args.synth:
        utterances = [(t0, t1, 190.0 + 40.0 * i) for i, (t0, t1, _, _, _) in enumerate(DEMO_SCRIPT)]
        src = SyntheticFrameSource(
            cfg, utterances, DEMO_TOTAL,
            loss_windows=loss, realtime=args.realtime, speed=args.speed, seed=7,
        )
        entries = [(t0, t1, en, conf) for (t0, t1, en, zh, conf) in DEMO_SCRIPT]
        refs = [(t0, t1, en, zh) for (t0, t1, en, zh, conf) in DEMO_SCRIPT]
        return src, entries, refs

    if not args.wav:
        raise SystemExit("provide --wav or --synth")
    src = WavFrameSource(
        args.wav, cfg,
        loss_windows=loss, realtime=args.realtime, speed=args.speed,
    )
    if args.script:
        entries, refs = load_script(args.script)
    else:
        entries, refs = [], None
    return src, entries, refs


def build_enhancer(args, cfg: BLineConfig):
    if args.enhance == "none":
        return BypassEnhancer()
    if args.enhance == "spectral":
        return SpectralSubtractionEnhancer(
            sample_rate=cfg.sample_rate, method="spectral"
        )
    if args.enhance == "wiener":
        return SpectralSubtractionEnhancer(
            sample_rate=cfg.sample_rate, method="wiener"
        )
    raise SystemExit(f"unknown enhancement: {args.enhance}")


def build_link(args):
    if not args.link:
        return None
    return LinkConfig(
        packet_ms=args.link_packet_ms,
        base_delay_ms=args.link_base_delay_ms,
        jitter_ms=args.link_jitter_ms,
        loss_rate=args.link_loss,
        burst_windows=parse_loss(args.link_burst) or (),
        jitter_buffer_ms=args.link_buffer_ms,
        seed=args.link_seed,
    )


def build_asr(args, cfg: BLineConfig, entries):
    if args.asr == "scripted":
        if not entries:
            print("warning: ScriptedASR has no transcript; pass --script for real WAV input")
        return ScriptedASR(entries, cfg)
    if args.asr == "sherpa":
        if not args.asr_model_dir:
            raise SystemExit("--asr sherpa requires --asr-model-dir")
        return SherpaASR(
            model_dir=args.asr_model_dir,
            cfg=cfg,
            num_threads=args.asr_threads,
            provider=args.asr_provider,
        )
    raise SystemExit(f"unknown ASR backend: {args.asr}")


def build_mt(args):
    if args.mt == "identity":
        return IdentityMT()
    if args.mt == "opus":
        return OpusMT(
            model_name=args.mt_model,
            device=args.mt_device,
            use_context=args.mt_context,
        )
    raise SystemExit(f"unknown MT backend: {args.mt}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="16 kHz mono/stereo 16-bit PCM WAV")
    ap.add_argument("--synth", action="store_true", help="stream a synthetic demo signal instead of --wav")
    ap.add_argument("--script", help="transcript JSON for ScriptedASR (required unless --synth)")
    ap.add_argument("--out", help="write events+metrics JSON here")
    ap.add_argument("--loss", help="comma-separated t0-t1 loss windows, e.g. 5.0-5.4")
    ap.add_argument("--realtime", action="store_true", help="release audio at wall-clock rate")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--chunk-ms", type=float, default=320.0)
    ap.add_argument("--stability-ms", type=float, default=240.0)

    ap.add_argument("--enhance", choices=["none", "spectral", "wiener"],
                    default="none", help="A=none, B=spectral/wiener")
    ap.add_argument("--enhance-block-ms", type=float, default=320.0)

    ap.add_argument("--link", action="store_true", help="wrap the source in a packet link")
    ap.add_argument("--link-packet-ms", type=float, default=20.0)
    ap.add_argument("--link-base-delay-ms", type=float, default=40.0)
    ap.add_argument("--link-jitter-ms", type=float, default=0.0)
    ap.add_argument("--link-buffer-ms", type=float, default=120.0)
    ap.add_argument("--link-loss", type=float, default=0.0,
                    help="per-packet loss probability, e.g. 0.05")
    ap.add_argument("--link-burst", help="t0-t1 burst loss window, e.g. 8.0-8.2")
    ap.add_argument("--link-seed", type=int, default=0)

    ap.add_argument("--asr", choices=["scripted", "sherpa"], default="scripted")
    ap.add_argument("--asr-model-dir", help="extracted sherpa-onnx streaming model")
    ap.add_argument("--asr-provider", default="cpu")
    ap.add_argument("--asr-threads", type=int, default=2)

    ap.add_argument("--mt", choices=["identity", "opus"], default="identity")
    ap.add_argument("--mt-model", default="Helsinki-NLP/opus-mt-en-zh")
    ap.add_argument("--mt-device", default=None)
    ap.add_argument("--mt-context", action="store_true")

    ap.add_argument("--video-in", help="input video to subtitle")
    ap.add_argument("--video-out", help="write a subtitled MP4 here")
    ap.add_argument("--video-seconds", type=float, default=0.0,
                    help="length of the generated test video (0 = stream duration)")
    ap.add_argument("--font", help="path to a CJK font for burning subtitles")
    args = ap.parse_args(argv)

    max_seg = 60 if args.synth else 120
    cfg = BLineConfig(
        chunk_ms=args.chunk_ms,
        stability_delay_ms=args.stability_ms,
        max_seg_chars=max_seg,
    )

    loss = parse_loss(args.loss)
    source, entries, refs = build_source(args, cfg, loss)

    link_cfg = build_link(args)
    if link_cfg is not None:
        source = LinkSource(source, link_cfg, cfg.sample_rate)

    enhancer = build_enhancer(args, cfg)
    if args.enhance != "none":
        source = EnhancedFrameSource(
            source,
            enhancer,
            cfg.sample_rate,
            block_ms=args.enhance_block_ms,
        )
    enhance_delay_ms = float(getattr(enhancer, "algorithmic_delay_ms", 0.0))
    enhance_block_ms = args.enhance_block_ms if args.enhance != "none" else 0.0

    asr = build_asr(args, cfg, entries)
    mt = build_mt(args)
    pipe = SubtitlePipeline(cfg, asr, mt)

    t0 = time.perf_counter()
    events = pipe.run(source)
    proc = time.perf_counter() - t0

    met = evaluate(events, references=refs, duration_sec=pipe.duration_sec,
                   processing_sec=proc, realtime=args.realtime)

    video_stats = None
    if args.video_out:
        video_in = args.video_in
        if not video_in:
            seconds = args.video_seconds or max(pipe.duration_sec, 1.0)
            video_in = make_test_video(
                str(Path(args.video_out).with_name("_test_input.mp4")),
                seconds=seconds,
            )
        video_stats = render_subtitles(
            video_in, events, args.video_out, font_path=args.font
        )
        print("video:", json.dumps(video_stats, ensure_ascii=False))

    print()
    for e in events:
        print(json.dumps(e.to_dict(), ensure_ascii=False))
    print("---")
    print("enhance:", args.enhance,
          f"(algorithmic {enhance_delay_ms:.1f} ms, block {enhance_block_ms:.1f} ms)")
    if link_cfg is not None:
        print("link:", json.dumps(asdict(link_cfg), ensure_ascii=False))
    print("config_hash:", cfg.freeze())
    print("metrics:", json.dumps(met, ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "config": cfg.to_dict(),
                "config_hash": cfg.freeze(),
                "enhance": args.enhance,
                "enhance_delay_ms": enhance_delay_ms,
                "enhance_block_ms": enhance_block_ms,
                "link": asdict(link_cfg) if link_cfg is not None else None,
                "events": [e.to_dict() for e in events],
                "metrics": met,
                "video": video_stats,
            }, f, ensure_ascii=False, indent=2)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
