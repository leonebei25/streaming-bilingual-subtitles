# B-line: streaming ASR → segmentation → MT → subtitles (initial version)

The recognition / translation half of the real-time bilingual subtitle project.
This is a working skeleton: the whole data flow runs end-to-end right now using a
deterministic "scripted" recognizer and an identity translator. The real Qwen3-ASR
and translation backends are documented adapters that fail loudly until the GPU
environment is configured.

## Layout

    b_line/
      config.py      frozen configuration + snapshot hash
      types.py       data contracts (frames, gaps, segments, subtitle events)
      audio.py       16 kHz mono I/O, resampling, STFT framing, test synthesis
      feeder.py      FrameFeeder: releases audio only up to the arrived horizon
      vad.py         energy + zero-crossing VAD with hangover endpointing
      asr.py         ASR protocol, ScriptedASR (mock), Qwen3ASR (stub)
      segmenter.py   stable-prefix segmentation (endpoint + forced split + gap flush)
      mt.py          MT protocol, IdentityMT (mock), PretrainedMT (stub)
      metrics.py     WER, chrF, timing error, RTF, latency percentiles
      manifest.py    60/20/20 data split by speaker/source
      pipeline.py    end-to-end orchestrator
    run_pipeline.py  CLI driver
    selftest.py      end-to-end smoke test with assertions
    tests/           unit tests

## Quick start

    # 1. run the self-test (also writes _selftest.wav)
    python selftest.py

    # 2. run the self-contained demo (synthesizes sample.wav)
    python run_pipeline.py --synth --loss 4.55-4.95

    # 3. run on your own audio with a transcript
    python run_pipeline.py --wav in.wav --script script.json --out events.json

    # 4. unit tests
    python -m unittest discover -s tests -v

The transcript JSON looks like:

    {
      "utterances": [
        {"t0": 0.5, "t1": 1.9, "en": "Hello world.", "zh": "你好，世界。", "conf": 0.95}
      ]
    }

## The invariants this version enforces

1. The real-time branch never reads audio that has not arrived: FrameFeeder
   only emits frames up to the released horizon.
2. Packet loss leaves a hole: a loss window becomes a gap event, and audio is
   never concatenated across it.
3. Subtitle timing comes from the pipeline, not the recognizer: frames carry a
   sequence number and sample offset; the segmenter attaches t_start / t_end.
4. Recognition and translation are separate modules (a course requirement).

## What is real vs. stubbed

Real (works now):
  - feeder / VAD / segmentation / orchestration / metrics / manifest / CLI
  - ScriptedASR models the streaming behaviour: partial text grows, a prefix
    stabilizes after a configurable delay

Stubbed (fail loudly until configured):
  - Qwen3ASR: needs the qwen-asr package + vLLM backend. Note that Qwen3-ASR
    streaming is vLLM-only and does not return timestamps, so the pipeline
    keeps its own chunk -> offset bookkeeping.
  - PretrainedMT: needs a pretrained NMT / LLM with frozen weights, context
    injection and terminology constraints.

## Config freeze

BLineConfig.freeze() returns a SHA-256 of the full configuration. Save it with
every experiment run so model versions and decoding settings stay reproducible.

## Next steps

  - wire Qwen3ASR.reset() / stream() against the vLLM streaming client,
  - implement PretrainedMT.translate() with context and terminology handling,
  - add word-level timing for the renderer (optionally Qwen3-ForcedAligner
    offline, labelled as such),
  - build the 60/20/20 manifest from the collected clips.
