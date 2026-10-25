#!/usr/bin/env python3
"""End-to-end self-test for the B-line initial version.

Builds a synthetic 'speech' signal, runs the full pipeline offline with a
scripted recognizer, and checks the invariants that matter:
  1. subtitles align to the script within a frame,
  2. a loss window produces a gap event and no concatenation,
  3. the feeder never reads ahead of the released horizon.
"""
from __future__ import annotations

import sys

from b_line.audio import synthesize_test_audio, save_wav_mono16
from b_line.asr import ScriptedASR
from b_line.config import BLineConfig
from b_line.feeder import FrameFeeder
from b_line.metrics import chrf, evaluate, wer
from b_line.mt import IdentityMT
from b_line.pipeline import SubtitlePipeline

SCRIPT = [
    (0.5, 1.9, "Welcome to digital signal processing.", "欢迎来到数字信号处理。", 0.95),
    (2.9, 4.3, "Today we study streaming speech recognition.", "今天我们学习流式语音识别。", 0.95),
    (5.3, 6.7, "Please mind the packet loss gaps.", "请注意丢包造成的缺口。", 0.95),
]
TOTAL = 9.0


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


def main():
    cfg = BLineConfig(stability_delay_ms=200.0, chunk_ms=320.0, vad_hangover_ms=300.0, max_seg_chars=60)
    utterances = [(t0, t1, 190.0 + 40.0 * i) for i, (t0, t1, _, _, _) in enumerate(SCRIPT)]
    x = synthesize_test_audio(utterances, TOTAL, sr=cfg.sample_rate, seed=7)
    save_wav_mono16("_selftest.wav", x, cfg.sample_rate)

    entries = [(t0, t1, en, conf) for (t0, t1, en, zh, conf) in SCRIPT]
    refs = [(t0, t1, en, zh) for (t0, t1, en, zh, conf) in SCRIPT]
    asr = ScriptedASR(entries, cfg)
    mt = IdentityMT()
    pipe = SubtitlePipeline(cfg, asr, mt)

    # 1) offline run with one loss window inside a silence gap
    events = pipe.run(x, cfg.sample_rate, loss_windows=[(4.55, 4.95)])
    subs = [e for e in events if not e.gap_flag]
    gaps = [e for e in events if e.gap_flag]

    ok = True
    ok &= check("subtitles produced", len(subs) == len(SCRIPT), f"got {len(subs)}")
    ok &= check("gap event produced", len(gaps) == 1, f"got {len(gaps)}")

    ref_text = " ".join(r[2] for r in refs)
    hyp_text = " ".join(e.text_en for e in subs)
    ok &= check("transcript matches script", ref_text == hyp_text, hyp_text[:60])

    terr = max(abs(subs[i].t_start - refs[i][0]) for i in range(min(len(subs), len(refs))))
    ok &= check("onset within 60 ms", terr <= 0.06, f"{terr * 1000:.1f} ms")

    met = evaluate(events, references=refs, duration_sec=pipe.duration_sec,
                   processing_sec=pipe.processing_sec)
    ok &= check("WER == 0", met.get("wer") == 0.0, f"wer={met.get('wer')}")
    print("metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in met.items()})

    # 2) metric unit checks
    ok &= check("WER unit exact", wer("a b c", "a b c") == 0.0)
    ok &= check("chrF unit identity", abs(chrf("你好世界", "你好世界") - 1.0) < 1e-6)
    ok &= check("chrF unit mismatch", chrf("你好世界", "你好 世界") < 1.0)

    # 3) feeder never-read-future
    f = FrameFeeder(x, cfg.sample_rate, cfg)
    f.release(1.0)
    items = f.drain()
    ok &= check("feeder stays within horizon", all(i.end_sec <= 1.0 + 1e-9 for i in items))
    ok &= check("feeder advances", len(items) > 0)

    print()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
