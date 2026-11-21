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
from b_line.asr import ScriptedASR, SherpaASR, default_model_dir
from b_line.config import BLineConfig
from b_line.metrics import evaluate
from b_line.mt import IdentityMT, OpusMT
from b_line.pipeline import SubtitlePipeline
from b_line.sources import (
    MicFrameSource,
    PcmStreamSource,
    SyntheticFrameSource,
    WavFrameSource,
)

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
    if args.mic_list:
        try:
            import sounddevice as sd
            apis = sd.query_hostapis()
            for i, d in enumerate(sd.query_devices()):
                if d["max_input_channels"] <= 0:
                    continue
                api = apis[d["hostapi"]]["name"] if d["hostapi"] < len(apis) else "?"
                try:
                    rate = f"{float(d.get('default_samplerate', 0)):.0f} Hz"
                except Exception:
                    rate = "?"
                print(f"{i:3d}  in={d['max_input_channels']}  {rate:>9}  "
                      f"[{api}]  {d['name']}")
        except Exception as exc:
            raise SystemExit(f"cannot list audio devices: {exc}")
        raise SystemExit(0)

    # Native streaming inputs: audio arrives frame by frame at wall-clock rate,
    # never preloaded. Both share MicFrameSource's streaming contract.
    if args.mic:
        return MicFrameSource(cfg, device=args.mic_device,
                              force_rate=args.mic_rate or None,
                              force_channels=args.mic_channels or None,
                              limit_sec=args.max_seconds), [], None
    if args.pcm_stdin:
        return PcmStreamSource(cfg, limit_sec=args.max_seconds), [], None

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
        raise SystemExit("provide one of --wav, --synth, --mic or --pcm-stdin")
    src = WavFrameSource(
        args.wav, cfg,
        loss_windows=loss, realtime=args.realtime, speed=args.speed,
    )
    if args.script:
        entries, refs = load_script(args.script)
    else:
        entries, refs = [], None
    return src, entries, refs


def probe_mic(src, seconds: float) -> None:
    """Record a few seconds from a live MicFrameSource, print the level, exit.

    Uses the same device/channel/rate path as ``--mic``, so a silent reading
    here means the device delivers silence (muted, blocked by Windows privacy,
    or the wrong default input) -- not a pipeline bug.
    """
    import time
    import numpy as np
    try:
        import sounddevice as sd
        info = sd.query_devices(getattr(src, "device", None), "input")
        print(f"mic probe: device={info['name']!r} "
              f"channels={getattr(src, '_channels', '?')} rate={src.capture_rate}")
    except Exception:
        pass
    t0 = time.monotonic()
    blocks = []
    try:
        while time.monotonic() - t0 < seconds:
            items = src.take()
            if not items:
                time.sleep(0.02)
                continue
            blocks += [it.data for it in items if hasattr(it, "data")]
    finally:
        stop = getattr(src, "stop", None)
        if stop:
            stop()
    if not blocks:
        print("mic probe: NO AUDIO arrived (0 frames)")
        return
    x = np.concatenate(blocks)
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x * x).mean()))
    print(f"mic probe: {len(x) / src.sr:.1f}s peak={peak:.4f} rms={rms:.4f}")
    print("  => SILENCE: fix Windows privacy / default input / mute"
          if peak < 0.01 else "  => audio arriving OK")


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
        model_dir = args.asr_model_dir or default_model_dir()
        if not model_dir:
            raise SystemExit(
                "--asr sherpa needs --asr-model-dir (no model found under "
                "models/; download "
                "sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06)")
        return SherpaASR(
            model_dir=model_dir,
            cfg=cfg,
            num_threads=args.asr_threads,
            provider=args.asr_provider,
            decoding_method=args.decoding_method,
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
    ap.add_argument("--wav", help="16 kHz mono/stereo 16-bit PCM WAV (file replay)")
    ap.add_argument("--synth", action="store_true", help="stream a synthetic demo signal instead of --wav")
    ap.add_argument("--mic", action="store_true",
                    help="capture the microphone in real time (native streaming input)")
    ap.add_argument("--mic-device", default=None,
                    help="input device index or name for --mic (see --mic-list)")
    ap.add_argument("--mic-rate", type=int, default=0,
                    help="force the capture sample rate for --mic (0 = auto: try "
                         "16000 then the device default); resampled to 16 kHz")
    ap.add_argument("--mic-channels", type=int, default=0,
                    help="force the capture channel count for --mic (0 = auto: "
                         "try 1, 2, then the device maximum); downmixed to mono")
    ap.add_argument("--mic-probe", type=float, default=0.0,
                    help="record N seconds from --mic, print the level, and exit "
                         "(diagnose a silent/hollow capture)")
    ap.add_argument("--mic-list", action="store_true",
                    help="list audio devices and exit")
    ap.add_argument("--pcm-stdin", action="store_true",
                    help="read 16 kHz mono signed-16 PCM from stdin as a live stream")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="stop a live --mic/--pcm-stdin stream after N seconds "
                         "(default 0 = run until Ctrl+C)")
    ap.add_argument("--script", help="transcript JSON for ScriptedASR (required unless --synth)")
    ap.add_argument("--out", help="write events+metrics JSON here")
    ap.add_argument("--loss", help="comma-separated t0-t1 loss windows, e.g. 5.0-5.4")
    ap.add_argument("--realtime", action="store_true", help="release audio at wall-clock rate")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    ap.add_argument("--chunk-ms", type=float, default=160.0)
    ap.add_argument("--stability-ms", type=float, default=120.0)
    ap.add_argument("--max-seg-chars", type=int, default=0,
                    help="force a caption at this many draft characters "
                         "(default 0 = no character cap)")
    ap.add_argument("--max-seg-seconds", type=float, default=15.0,
                    help="force a caption once a draft has been growing this long "
                         "(default 15 s; pass 6 for tighter latency evidence)")
    ap.add_argument("--sentence-gap-ms", type=float, default=800.0,
                    help="with --split-on-pauses: word gap that counts as a "
                         "sentence boundary")
    ap.add_argument("--sentence-min-chars", type=int, default=40,
                    help="with --split-on-pauses: shortest piece worth cutting")
    ap.add_argument("--split-on-pauses", action="store_true",
                    help="split a committed draft at pauses inside it (off by "
                         "default: one draft run = one caption)")

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
    ap.add_argument("--asr-model-dir",
                    help="extracted streaming model; default: the Kroko 2025 "
                         "English model under models/ when downloaded, else the "
                         "legacy 2023 model")
    ap.add_argument("--asr-provider", default="cpu")
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--decoding-method", default="greedy_search",
                    choices=["greedy_search", "modified_beam_search"],
                    help="greedy (default) keeps up in real time; beam is more "
                         "accurate but slower (use when evaluating)")
    ap.add_argument("--no-endpoint", action="store_true",
                    help="disable sherpa's endpoint detector (on by default); "
                         "captions then come only from forced cuts")
    ap.add_argument("--rule1-ms", type=float, default=2400.0,
                    help="endpoint rule1: trailing silence even with no decoded "
                         "speech (default 2400)")
    ap.add_argument("--rule2-ms", type=float, default=2000.0,
                    help="endpoint rule2: trailing silence after decoded speech "
                         "(default 2000)")
    ap.add_argument("--rule3-sec", type=float, default=20.0,
                    help="endpoint rule3: hard cap on one utterance (default 20 s)")

    ap.add_argument("--mt", choices=["identity", "opus"], default="identity")
    ap.add_argument("--mt-model", default="Helsinki-NLP/opus-mt-en-zh")
    ap.add_argument("--mt-device", default=None)
    ap.add_argument("--mt-context", action="store_true")
    ap.add_argument("--punct-model", default="",
                    help="punctuation model: a directory or a .onnx file. A dir "
                         "containing bpe.vocab uses the online English CNN-BiLSTM; "
                         "anything else uses the offline zh+en CT-Transformer")
    ap.add_argument("--semantic-cuts", action="store_true",
                    help="commit a caption when a stable sentence end appears "
                         "inside the draft (needs --punct-model)")
    ap.add_argument("--semantic-lookahead", type=int, default=8,
                    help="words that must remain after a candidate cut (default 8)")
    ap.add_argument("--semantic-confirm", type=int, default=2,
                    help="consecutive runs that must agree on the cut (default 2)")
    ap.add_argument("--semantic-min-chars", type=int, default=24,
                    help="do not look for a cut below this draft size (default 24)")
    ap.add_argument("--force-pause-ms", type=float, default=600.0,
                    help="when a draft must be forced, back off to the latest "
                         "pause at least this long (default 600 ms)")
    ap.add_argument("--force-min-chars", type=int, default=20,
                    help="text that must precede a forced pause cut (default 20)")

    ap.add_argument("--video-in", help="input video to subtitle")
    ap.add_argument("--video-out", help="write a subtitled MP4 here")
    ap.add_argument("--video-seconds", type=float, default=0.0,
                    help="length of the generated test video (0 = stream duration)")
    ap.add_argument("--font", help="path to a CJK font for burning subtitles")
    args = ap.parse_args(argv)
    if args.semantic_cuts and not args.punct_model:
        raise SystemExit("--semantic-cuts needs --punct-model <dir or model.onnx>")

    if args.asr == "sherpa":
        if not args.asr_model_dir:
            args.asr_model_dir = default_model_dir()
        asr_snapshot = "sherpa:" + Path(args.asr_model_dir or "?").name
    else:
        asr_snapshot = "scripted"
    cfg = BLineConfig(
        chunk_ms=args.chunk_ms,
        stability_delay_ms=args.stability_ms,
        max_seg_chars=args.max_seg_chars,
        max_seg_seconds=args.max_seg_seconds,
        sentence_gap_ms=args.sentence_gap_ms,
        sentence_min_chars=args.sentence_min_chars,
        split_on_pauses=args.split_on_pauses,
        semantic_cuts=args.semantic_cuts,
        semantic_lookahead_words=args.semantic_lookahead,
        semantic_confirm=args.semantic_confirm,
        semantic_min_chars=args.semantic_min_chars,
        force_pause_ms=args.force_pause_ms,
        force_min_chars=args.force_min_chars,
        enable_endpoint_detection=not args.no_endpoint,
        rule1_min_trailing_silence=args.rule1_ms / 1000.0,
        rule2_min_trailing_silence=args.rule2_ms / 1000.0,
        rule3_min_utterance_length=args.rule3_sec,
        asr_model=asr_snapshot,
        mt_model=args.mt_model if args.mt == "opus" else "identity",
    )

    loss = parse_loss(args.loss)
    source, entries, refs = build_source(args, cfg, loss)

    if getattr(args, "mic_probe", 0.0) and isinstance(source, MicFrameSource):
        probe_mic(source, args.mic_probe)
        return

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
    if args.asr == "sherpa":
        print("endpoint:",
              "on" if cfg.enable_endpoint_detection else "off",
              f"rule1={cfg.rule1_min_trailing_silence * 1000:.0f}ms",
              f"rule2={cfg.rule2_min_trailing_silence * 1000:.0f}ms",
              f"rule3={cfg.rule3_min_utterance_length:.0f}s")
    mt = build_mt(args)
    punct = None
    if args.punct_model:
        from b_line.punct import PunctRestorer
        punct = PunctRestorer(args.punct_model)
        cfg.punct_model = punct.describe()
    if args.semantic_cuts:
        print(f"semantic cuts: on (lookahead {cfg.semantic_lookahead_words} words, "
              f"confirm {cfg.semantic_confirm}, min {cfg.semantic_min_chars} chars)")
    # No punctuation model: each committed caption only gets a full stop.
    # Commas guessed from pauses read as wrong, so none are invented.
    cfg.rule_punctuation = punct is None
    pipe = SubtitlePipeline(cfg, asr, mt, punct=punct)

    t0 = time.perf_counter()
    events = pipe.run(source)
    proc = time.perf_counter() - t0

    met = evaluate(events, references=refs, duration_sec=pipe.duration_sec,
                   processing_sec=proc,
                   realtime=args.realtime or args.mic or args.pcm_stdin)

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
    if args.asr == "sherpa":
        print("endpoint_stats:", json.dumps({
            "endpoints": getattr(asr, "endpoint_count", 0),
            "forced": getattr(asr, "force_count", 0),
            "streams": getattr(asr, "stream_count", 0),
            "semantic_cuts": getattr(pipe, "semantic_cut_count", 0),
            "pause_cuts": getattr(pipe, "pause_cut_count", 0),
        }))

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
