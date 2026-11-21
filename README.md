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
      asr.py         ASR protocol, ScriptedASR (mock), SherpaASR (real), Qwen3ASR (stub)
      segmenter.py   ASR-finalized sentence segmentation (stable_chars + onset)
      sentence.py    sentence splitting (punctuation, discourse markers, cap)
      punct.py       optional punctuation restoration (sherpa-onnx CT-Transformer)
      mt.py          MT protocol, IdentityMT (mock), OpusMT (real), PretrainedMT (stub)
      metrics.py     WER, chrF, timing error, RTF, latency percentiles
      manifest.py    60/20/20 data split by speaker/source
      pipeline.py    orchestration: source -> VAD -> ASR -> segmenter -> MT -> events
      display.py     reading-time pacing and the caption queue (min hold, backlog)
    a_line/
      link.py        packetization, 40 ms base delay, jitter, loss, jitter buffer
      enhance/
        base.py      Enhancer protocol: reset / process / algorithmic_delay_ms
        bypass.py    A chain: no enhancement
        spectral.py  B chain: causal spectral subtraction / Wiener
        source.py    EnhancedFrameSource: blocks, enhances, re-emits original frames
      video.py       burn subtitle events into an MP4
    run_pipeline.py  CLI driver
    scripts/live_demo.py      video + scrolling transcript window
    scripts/desktop_smoke.py  A/B smoke run with events, metrics and video
    tools/convert_wav.py      convert any PCM WAV to 16 kHz mono 16-bit
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

    # 0. activate the virtual environment (PowerShell; the leading dot matters)
    . .\activate.ps1
    # cmd.exe:  activate.bat
    # Git Bash: source ./activate.sh

    # 0.5 convert a PCM WAV (44.1/48 kHz, stereo, ...) to 16 kHz mono
    python tools/convert_wav.py video_test/vedio_1.wav

    # 1. self-test: synthetic stream, assertions, no files written
    python selftest.py

    # 2. CLI demo over a synthetic stream
    python run_pipeline.py --synth --loss 4.55-4.95

    # 3. real file, replayed at wall-clock rate
    python run_pipeline.py --wav in.wav --script script.json --realtime

    # 4. unit tests
    python -m unittest discover -s tests -v

    # 5. desktop A/B smoke: link -> enhancement -> ASR -> MT -> events -> MP4
    python scripts/desktop_smoke.py --synth

    # 6. real ASR + real MT on a WAV
    python run_pipeline.py --wav in.wav --script script.json \
        --asr sherpa --asr-model-dir models/sherpa-en \
        --mt opus --enhance spectral --link --link-loss 0.05 \
        --video-out out.mp4 --out run.json

## Converting audio to 16 kHz

The pipeline accepts only 16 kHz, 16-bit PCM WAV (mono or stereo). Use the
converter for anything else:

    python tools/convert_wav.py in.wav                  # -> in_16k.wav
    python tools/convert_wav.py in.wav -o out.wav
    python tools/convert_wav.py lecture_dir --out-dir lecture_16k --recursive
    python tools/convert_wav.py in.wav --keep-stereo     # default downmixes to mono
    python tools/convert_wav.py in.wav --normalize       # scale peak to 0.99

It reads 8/16/24/32-bit PCM WAV, downmixes to mono by default, and resamples
with `scipy.signal.resample_poly` (polyphase anti-aliasing, not naive
decimation). The output is always 16-bit PCM. Float WAV is not supported by
Python's `wave`; convert it with ffmpeg first. `scipy` is in
`requirements-desktop.txt`.

Each run prints a JSON summary with source/output rates, duration, resample
ratio, peak before writing and the number of clipped samples; a non-zero exit
code means at least one file failed.

## A/B/C switching

The three course pipelines share one code path; only `--enhance` changes:

| chain | flag | what it does | training |
|---|---|---|---|
| A | `--enhance none` | pass received audio straight to ASR | none |
| B | `--enhance spectral` | causal spectral subtraction, 25 ms / 10 ms | none |
| B' | `--enhance wiener` | causal Wiener variant | none |
| C | `--enhance dnn` (planned) | causal LSTM/TF-mask network | once, then frozen |

`--link` applies the same packet loss / jitter trace to every chain when the
seed is identical. Use `--link-loss 0.05 --link-jitter-ms 100
--link-buffer-ms 120` for the course's stress condition, and
`--link-burst 8.0-8.2` for the 200 ms burst.

`a_line/enhance/source.py` buffers 320 ms blocks by default (`--enhance-block-ms`)
so the STFT and the network see enough context. That block delay, the
enhancer's own algorithmic delay (15 ms for the 25/10 spectral stage), the link
playout delay, and the ASR lookahead must all be reported separately.

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
6. **A short gap is a hole, not a sentence boundary**: gaps shorter than
   `gap_reset_ms` (default 500 ms) emit a gap event and flush the recogniser
   chunk, but keep the VAD and segmenter state, so a 20 ms lost packet does not
   shift every later subtitle.
7. **Sentences come from the recognizer, not the VAD**: `stable_chars` is the
   prefix the ASR has finalized, and each growth of it commits one whole
   sentence. Sherpa's endpoint (language-model based, ~1.2 s of trailing
   silence) decides boundaries, so a mid-sentence breath no longer chops a
   caption. The VAD only stamps `t_start` with the frame-accurate speech onset.
   If a sentence stays a draft past `max_seg_seconds` (10 s) / `max_seg_chars`
   (160) -- the recognizer never endpointed, e.g. music or continuous speech --
   the pipeline asks the ASR to close it (`force_finalize`), so captions cannot
   stop appearing.
8. **Long stretches are split at real pauses, and every caption ends with a
   full stop**: when the recognizer exposes word timestamps (`get_result_all`),
   a forced slice is cut at gaps of at least `sentence_gap_ms` (default 800 ms)
   once the piece is at least `sentence_min_chars` (60) long, so the cut lands
   where the speaker actually paused. The recognizer emits no marks, so a
   committed caption gets a terminal full stop (`rule_punctuation`). Marks
   *inside* a sentence are never guessed from pauses -- that produced wrong
   commas -- only `--punct-model` may place them.
9. **The transcript keeps everything and types itself out**: every finalized
   sentence is appended with its timestamp, its English text, and the Chinese
   translation underneath when MT is on. Text is revealed one character at a
   time (30 characters/s, draining any backlog over ~1.2 s) -- including a
   sentence that arrives in one piece from a backend that returns no partials,
   so a block of text is never pasted on screen. The sentence still being
   spoken is the green `▌` line at the bottom; when the recognizer ends it, the
   same words move into the transcript with the full stop added. The panel
   follows the newest line until you scroll up, and End (or scrolling back
   down) resumes it. `CaptionQueue` still supplies the hold/lag figures printed
   by `--debug`, but it no longer decides what is on screen.

## Real vs. stubbed

Real: sources, VAD, segmentation, orchestration, metrics, manifest, CLI, link
simulation, spectral/Wiener enhancement, subtitle video rendering, SherpaASR,
OpusMT.

Stubbed (fail loudly until configured), or not implemented yet:
  * Qwen3ASR — needs the qwen-asr package + a vLLM backend. Note that Qwen3-ASR
    streaming is vLLM-only and returns no timestamps, so the pipeline keeps its
    own chunk -> offset bookkeeping.
  * PretrainedMT — superseded by OpusMT for the desktop pipeline; kept as a
    placeholder for a larger frozen NMT / LLM backend.
  * The C-chain DNN enhancer (`--enhance dnn`) — the dataset, training script
    and ONNX wrapper are the next piece of work.
  * MicFrameSource — needs a 16 kHz mono capture device (sounddevice).

Desktop dependencies are in `requirements-desktop.txt`. The sherpa model is a
separate download (e.g. `sherpa-onnx-streaming-zipformer-en-2023-06-26`) and the
MarianMT weights are fetched by `transformers` or supplied as a local directory
for offline use.

## Next steps

  * C chain: `a_line/enhance/dataset.py`, `model.py`, `train.py`, `dnn.py`
    (causal LSTM/TF-mask network, ONNX export, `--enhance dnn`),
  * data: 60 clips / 30 min / 6 speakers, English transcripts + reviewed
    Chinese references, DNS noise/RIR, 60/20/20 manifest,
  * 10-minute continuous run with latency / buffer / loss display,
  * MicFrameSource from a bounded capture queue (demo only; experiments replay WAV),
  * optional: Qwen3ASR / a small LLM MT backend as an extension, not the main path,
  * Jetson Orin Nano packaging: offline wheels, ONNX/TensorRT, 10-minute on-device run.
