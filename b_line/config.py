"""Frozen configuration for the B-line pipeline.

Every value that affects reproducibility or a reported metric belongs here.
freeze() returns a content hash used as the config snapshot the project
requires (models, weights and decoding settings must be fixed and recorded).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass
class BLineConfig:
    # Audio / DSP (the interface with A-line)
    sample_rate: int = 16000
    window_ms: float = 25.0
    hop_ms: float = 10.0

    # Streaming ASR
    chunk_ms: float = 160.0          # audio fed to the recognizer per step; half the
                                     # old 320 ms halves the wait for a new partial
    stability_delay_ms: float = 120.0  # after an utterance ends, how long until its
                                      # text stops being rewritten (becomes "stable")
    asr_model: str = "scripted"      # "scripted" | "qwen3-0.6b" | "qwen3-1.7b"

    # Endpoint detection (the sentence-boundary source for sherpa)
    # sherpa-onnx's factory default is False. With it off, is_endpoint() always
    # returns False, stable_chars never grows, and every caption is a forced
    # 10 s / 160-character cut instead of a sentence. Keep it on for sherpa.
    enable_endpoint_detection: bool = True
    # rule1: trailing silence that ends an utterance even when nothing was
    # decoded (a quiet room). rule2: trailing silence after decoded speech --
    # the normal trigger. rule3: hard cap on the length of one utterance.
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 1.2
    rule3_min_utterance_length: float = 20.0

    # VAD / endpointing
    # The VAD no longer decides sentence boundaries -- the ASR endpoint does
    # (sherpa's is language-model based and waits ~1.2 s of silence). The VAD
    # only marks where speech begins (for t_start) and estimates the noise
    # floor, so its hangover can be long without chopping sentences.
    vad_energy_ratio: float = 40.0   # speech requires energy >= ratio * noise_floor
    vad_zcr_max: float = 0.4         # zero-crossing veto (rejects wideband noise bursts)
    vad_hangover_ms: float = 500.0   # trailing silence before the VAD resets its
                                     # speech run (not a sentence split; keeps
                                     # t_start aligned to each speech run)
    vad_init_frames: int = 50        # frames used to estimate the noise floor

    # Packet-loss handling
    # A short gap is a hole in the timeline, not the end of a sentence. Only a
    # gap at least this long flushes the segment and resets the VAD.
    gap_reset_ms: float = 500.0

    # Segmentation
    # Sentence boundaries come from the ASR's own endpoint (sherpa rule2 fires
    # after ~1.2 s of trailing silence, rule3 caps a sentence at ~20 s). The
    # segmenter does not cut finalized sentences itself; these two values are
    # documented targets for how long a single caption should be. To tighten a
    # long sentence, tune sherpa's endpoint rules, not a post-hoc character cap.
    max_seg_chars: int = 0           # 0 = no character cap; a draft is bounded by
                                     # max_seg_seconds and sherpa's rule3 instead
    max_seg_seconds: float = 15.0    # a draft older than this is force-finalized;
                                     # long enough to keep a whole sentence together
                                     # (still below sherpa's 20 s rule3)
    sentence_gap_ms: float = 800.0   # word gap that counts as a sentence boundary;
                                     # short pauses stay inside one caption
    sentence_min_chars: int = 60     # a caption shorter than this is not worth a cut
    # One draft run on screen is one caption: whichever boundary ends the draft
    # (the ASR endpoint or a forced commit) is the only splitter. Set this to
    # True to re-enable cutting a committed slice at the pauses inside it.
    split_on_pauses: bool = False

    # Semantic cuts: ask a punctuation model where sentences end inside the
    # draft and commit a confirmed boundary early. Needs --punct-model.
    semantic_cuts: bool = False
    semantic_lookahead_words: int = 8   # never cut this close to the live edge
    semantic_confirm: int = 2           # same offset on this many runs
    semantic_step_chars: int = 8        # re-run after the draft grows this much
    semantic_step_sec: float = 0.32     # ... or after this long
    semantic_min_chars: int = 24        # do not bother below this size

    # Forced commits back off to a real pause instead of cutting wherever the
    # draft happens to end. A pause shorter than this is not worth a cut.
    force_pause_ms: float = 400.0
    force_min_chars: int = 20           # text that must precede the pause

    # Rule-based punctuation, for when no punctuation model is configured: a
    # full stop at the end of every committed sentence, nothing else. Commas
    # guessed from pauses read as wrong; only the model (--punct-model) may
    # place marks inside a sentence.
    rule_punctuation: bool = False

    # Translation
    mt_model: str = "identity"       # "identity" | a pretrained MT model name
    mt_context_sentences: int = 1    # previous sentences injected as context

    # Punctuation restoration (empty = none; the ASR text has no sentence marks)
    punct_model: str = ""

    # Reproducibility
    seed: int = 0

    @property
    def hop_len(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def window_len(self) -> int:
        return int(round(self.sample_rate * self.window_ms / 1000.0))

    @property
    def chunk_len(self) -> int:
        return int(round(self.sample_rate * self.chunk_ms / 1000.0))

    @property
    def chunk_frames(self) -> int:
        return max(1, self.chunk_len // self.hop_len)

    def to_dict(self) -> dict:
        return asdict(self)

    def freeze(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "BLineConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
