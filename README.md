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
      punct.py       optional punctuation restoration (sherpa-onnx online
                     CNN-BiLSTM, or offline zh+en CT-Transformer)
      mt.py          MT protocol, IdentityMT (mock), OpusMT (torch), CT2MT
                     (CTranslate2 int8, the Jetson path), PretrainedMT (stub)
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

    # 6. real ASR + real MT on a WAV. With --asr-model-dir omitted it uses the
    #    Kroko 2025 English model under models/ when present (recommended), else
    #    the legacy 2023 model.
    python run_pipeline.py --wav in.wav \
        --asr sherpa \
        --mt opus --enhance spectral --link --link-loss 0.05 \
        --video-out out.mp4 --out run.json

    # 7. live microphone input (native streaming: audio is consumed hop by hop as
    #    it arrives, never preloaded). Needs `pip install sounddevice`.
    python run_pipeline.py --mic-list                 # find a device index
    python run_pipeline.py --mic --asr sherpa --max-seconds 60 --out runs/mic.json

    # 7b. same live input, but with the on-screen caption window. No --video:
    #     play a video in any external player and let the mic/loopback hear it.
    python scripts/live_demo.py --mic --mt opus --mt-model models/opus-mt-en-zh

    # 7c. deployment backend: Opus MT converted to CTranslate2 int8. No torch
    #     at inference time; this is what runs on the Jetson. Convert once:
    python scripts/convert_mt_ct2.py --model models/opus-mt-en-zh \
        --out models/opus-mt-en-zh-ct2
    python scripts/live_demo.py --mic --mt ct2 --mt-model models/opus-mt-en-zh-ct2
    python scripts/check_mt.py --backend ct2 --threads 6   # load + ms/sentence

    # 7d. Linux / Jetson launcher: run_mic.sh wraps the venv, the display
    #     check and PulseAudio loopback. MT=auto uses ct2 when converted.
    ./run_mic.sh                        # microphone
    ./run_mic.sh monitor                # capture what the machine plays
    ./run_mic.sh devices                # PortAudio inputs + pactl sources
    MT=ct2 ./run_mic.sh monitor         # deployment backend, system audio

    # 8. live PCM over stdin (16 kHz mono signed-16 little-endian), e.g. a pipe
    #    or socket feed. Same streaming contract as --mic. run_mic.sh monitor
    #    is exactly this: parec --format=s16le --rate=16000 | live_demo.
    some-feed | python run_pipeline.py --pcm-stdin --asr sherpa --max-seconds 60

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
   sentence. Sherpa's endpoint decides boundaries from trailing silence: 1.2 s
   after decoded speech, 2.4 s even with nothing decoded, or a 20 s hard cap.
   Endpoint detection is enabled explicitly (`enable_endpoint_detection`,
   default on; sherpa's own default is off, which silently turns sentence
   segmentation into forced 10 s / 160-character cuts). Tune it with
   `--rule1-ms`, `--rule2-ms`, `--rule3-sec`, or turn it off with
   `--no-endpoint`. The VAD only stamps `t_start` with the frame-accurate
   speech onset.
   If a sentence stays a draft past `max_seg_seconds` (15 s in the live demo) --
   the recognizer never endpointed, e.g. music or continuous speech -- the
   pipeline asks the ASR to close it (`force_finalize`), so captions cannot stop
   appearing. Before giving up, a forced commit first backs off to the latest
   pause of at least `force_pause_ms` (400 ms) using the word timings, and only
   cuts at the draft's end when there is no such pause. A character cap is
   available (`--max-seg-chars`) but is off by default; the draft's age and
   sherpa's 20 s rule3 are the only length limits.
8. **One draft run is one caption, and every caption ends with a full stop**:
   whatever boundary ends the draft (the ASR endpoint, a semantic cut, or a
   forced commit) is the only splitter, so the line the viewer watched growing
   becomes exactly one caption. Word times are shifted to absolute audio
   seconds, so a caption's `t_end` is the end of its last word rather than the
   moment it was committed, and two captions never overlap. The recognizer
   emits no marks, so a committed caption gets a terminal full stop
   (`rule_punctuation`). Marks *inside* a sentence are never guessed from
   pauses -- that produced wrong commas -- only `--punct-model` may place them.
   `--split-on-pauses` re-enables cutting a committed slice at word gaps of at
   least `sentence_gap_ms` (800 ms) once the piece is at least
   `sentence_min_chars` (60) long, when smaller captions matter more than the
   draft/final correspondence.
9. **Semantic cuts (optional)**: with `--punct-model` plus `--semantic-cuts`,
   the live draft is punctuated on a ~320 ms cadence. A sentence-final mark
   that leaves at least `--semantic-lookahead` (8) words after it and appears at
   the same offset on `--semantic-confirm` (2) consecutive runs becomes a cut:
   everything before it is committed, the rest keeps growing as the draft. The
   recognizer is never reset, so only the committed/draft boundary moves; the
   endpoint still caps the utterance and `max_seg_seconds` is the stall guard.
   Leave it off for the A/B baseline.
10. **The transcript keeps everything and types itself out**: every finalized
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
OpusMT, CT2MT.

Stubbed (fail loudly until configured), or not implemented yet:
  * Qwen3ASR — needs the qwen-asr package + a vLLM backend. Note that Qwen3-ASR
    streaming is vLLM-only and returns no timestamps, so the pipeline keeps its
    own chunk -> offset bookkeeping.
  * PretrainedMT — superseded by OpusMT for the desktop pipeline; kept as a
    placeholder for a larger frozen NMT / LLM backend.
  * The C-chain DNN enhancer (`--enhance dnn`) — the dataset, training script
    and ONNX wrapper are the next piece of work.
  * MicFrameSource — needs a 16 kHz mono capture device (sounddevice).

Desktop dependencies are in `requirements-desktop.txt`. The ASR model is a
separate download; the recommended streaming English model is the Kroko 2025
package (much stronger on real broadcast/news audio than the LibriSpeech-trained
2023 one, and still real-time):

    HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
        csukuangfj/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06 \
        --local-dir models/sherpa-onnx-streaming-zipformer-en-kroko-2025-08-06

When `--asr-model-dir` is omitted, `run_pipeline.py` and `scripts/live_demo.py`
pick that directory automatically (falling back to
`sherpa-onnx-streaming-zipformer-en-2023-06-26` if only that is present).
`--decoding-method modified_beam_search` trades latency for accuracy when
evaluating. The MarianMT weights are fetched by `transformers` or supplied as a
local directory for offline use.

There are two interchangeable backends for the same Marian checkpoint.
`--mt opus` runs it through torch (the desktop baseline). `--mt ct2` runs a
CTranslate2 conversion: `scripts/convert_mt_ct2.py` writes a portable directory
(int8 by default, roughly a quarter of the size) and inference then needs only
`ctranslate2` plus the Marian tokenizer -- no torch and no CUDA build. That is
the Jetson deployment path. `--mt auto` prefers `models/opus-mt-en-zh-ct2` when
it exists, then `models/opus-mt-en-zh`. Latency and quality differences between
the two are measured with `scripts/check_mt.py --backend ct2|opus`.

A punctuation model is optional. `--punct-model` accepts an extracted directory
or a single `.onnx` file: a directory containing `bpe.vocab` is the online
English CNN-BiLSTM (`sherpa-onnx-online-punct-en-2024-08-06`, ~30 MB, also
restores casing), anything else is the offline zh+en CT-Transformer (~279 MB).
Both come from the `punctuation-models` release; `scripts/check_punct.py`
loads one and prints its per-call latency. Punctuation adds marks to a caption;
it does not change where the caption starts or ends.

## Next steps

  * C chain: `a_line/enhance/dataset.py`, `model.py`, `train.py`, `dnn.py`
    (causal LSTM/TF-mask network, ONNX export, `--enhance dnn`),
  * data: 60 clips / 30 min / 6 speakers, English transcripts + reviewed
    Chinese references, DNS noise/RIR, 60/20/20 manifest,
  * 10-minute continuous run with latency / buffer / loss display,
  * MicFrameSource from a bounded capture queue (demo only; experiments replay WAV),
  * optional: Qwen3ASR / a small LLM MT backend as an extension, not the main path,
  * Jetson Orin Nano packaging: offline wheels, ONNX/TensorRT, 10-minute on-device run.
