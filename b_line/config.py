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
    chunk_ms: float = 320.0          # audio fed to the recognizer per step
    stability_delay_ms: float = 240.0  # after an utterance ends, how long until its
                                      # text stops being rewritten (becomes "stable")
    asr_model: str = "scripted"      # "scripted" | "qwen3-0.6b" | "qwen3-1.7b"

    # VAD / endpointing
    vad_energy_ratio: float = 40.0   # speech requires energy >= ratio * noise_floor
    vad_zcr_max: float = 0.4         # zero-crossing veto (rejects wideband noise bursts)
    vad_hangover_ms: float = 300.0   # trailing silence before an endpoint fires
    vad_init_frames: int = 50        # frames used to estimate the noise floor

    # Segmentation
    max_seg_chars: int = 40          # force a split when a segment grows past this
    max_seg_seconds: float = 6.0     # reserved; not enforced in the initial version

    # Translation
    mt_model: str = "identity"       # "identity" | a pretrained MT model name
    mt_context_sentences: int = 1    # previous sentences injected as context

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
