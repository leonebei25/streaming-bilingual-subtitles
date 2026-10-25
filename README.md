# B-line: streaming ASR → segmentation → MT → subtitles

The recognition / translation half of the real-time bilingual subtitle project.

**The pipeline consumes a stream, never a buffer.** Its entry point takes a
FrameSource, and a source hands out audio only up to a "released" horizon — i.e.
only audio that has actually arrived. This is the manual's rule that the
real-time branch must not access audio that has not yet arrived.

## Layout

    b_line/
      config.py      frozen configuration + snapshot hash
      types.py       data contracts (frames, gaps, segments, subtitle events)
      sources.py     FrameSource: WavFrameSource / SyntheticFrameSource / MicFrameSource
      vad.py         energy + zero-crossing VAD with hangover endpointing
      asr.py         ASR protocol, ScriptedASR (mock), Qwen3ASR (stub)
      segmenter.py   stable-prefix segmentation (endpoint + forced split + gap flush)
      mt.py          MT protocol, IdentityMT (mock), PretrainedMT (stub)
      metrics.py     WER, chrF, timing error, RTF, latency percentiles
      manifest.py    60/20/20 data split by speaker/source
      pipeline.py    orchestration: source -> VAD -> ASR -> segmenter -> MT -> events
    run_pipeline.py  CLI driver
    selftest.py      end-to-end smoke test with assertions
    tests/           unit tests

## The stream contract

    class FrameSource:
        take(max_items=512) -> list[AudioFrame | Gap]
        ended() -> bool
        duration_sec: float
        released_sec: float        # how much of the stream has arrived

Three implementations follow the manual's two input kinds:

| source | manual wording | reads ahead? |
|---|---|---|
| WavFrameSource | "recordings replayed with their original timestamps" | no: one hop per read, zero samples on construction |
| MicFrameSource | "a camera and microphone feed" | stub: device integration point |
| SyntheticFrameSource | test / demo signal | no: generated hop by hop |

WavFrameSource requires 16 kHz mono/stereo 16-bit PCM. Resampling belongs to the
A-line, which owns capture and format normalisation.

## Quick start

    # 1. self-test: synthetic stream, assertions, no files written
    python selftest.py

    # 2. CLI demo over a synthetic stream
    python run_pipeline.py --synth --loss 4.55-4.95

    # 3. real file, replayed at wall-clock rate
    python run_pipeline.py --wav in.wav --script script.json --realtime

    # 4. unit tests
    python -m unittest discover -s tests -v

The transcript JSON used by the mock recognizer looks like:

    {
      "utterances": [
        {"t0": 0.5, "t1": 1.9, "en": "Hello world.", "zh": "你好，世界。", "conf": 0.95}
      ]
    }

## Invariants enforced by the code

1. **No reading ahead**: take() never returns an item past released_sec
   (tested, including a realtime source).
2. **Lazy reading**: constructing a WavFrameSource reads zero samples.
3. **One gap per loss window**: a dropped interval becomes exactly one Gap, even
   when the release horizon is still inside it — no fragmentation, and no
   concatenation across the hole.
4. **Timestamps come from the pipeline**: frames carry a sequence number and
   sample offset, so t_start / t_end are frame-accurate, not chunk-aligned.
5. **ASR and MT stay separate modules** (a course requirement), so the A/B/C
   enhancement comparison can hold both fixed.

## Real vs. stubbed

Real: sources, VAD, segmentation, orchestration, metrics, manifest, CLI.

Stubbed (fail loudly until configured):
  * Qwen3ASR — needs the qwen-asr package + a vLLM backend. Note that Qwen3-ASR
    streaming is vLLM-only and returns no timestamps, so the pipeline keeps its
    own chunk -> offset bookkeeping.
  * PretrainedMT — needs a frozen pretrained NMT / LLM, context injection and
    terminology constraints.
  * MicFrameSource — needs a 16 kHz mono capture device (sounddevice).

## Next steps

  * implement MicFrameSource.take() from a bounded capture queue,
  * wire Qwen3ASR.reset() / stream() to the vLLM streaming client,
  * implement PretrainedMT.translate() with context and terminology handling,
  * build the 60/20/20 manifest from the collected clips.
