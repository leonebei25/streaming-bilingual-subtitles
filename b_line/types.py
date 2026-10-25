"""Data contracts shared across the B-line modules.

Keeping these small and explicit is the point of the W0 interface agreement:
A-line hands us timestamped frames; we emit timestamped subtitle events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AudioFrame:
    """One 10 ms hop of mono audio that has actually 'arrived'."""
    start_sec: float
    end_sec: float
    data: np.ndarray  # float32, shape (hop_len,)
    seq: int          # monotonically increasing sequence number

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class Gap:
    """A hole in the timeline caused by packet loss. Never filled by concatenation."""
    start_sec: float
    end_sec: float
    reason: str = "loss"


@dataclass
class ASRUpdate:
    """Incremental recognizer output.

    'text' is the full partial transcript so far and may be rewritten on later
    updates. 'stable_chars' is the length of the prefix that will not change.
    """
    text: str
    stable_chars: int
    confidence: float
    finalized: bool = False


@dataclass
class Segment:
    """A finalized English utterance ready for translation."""
    seg_id: int
    t_start: float
    t_end: float
    text_en: str
    conf_asr: float


@dataclass
class SubtitleEvent:
    """The unit the renderer displays."""
    seg_id: int
    t_start: float
    t_end: float
    text_en: str
    text_zh: str
    conf_asr: float
    conf_mt: float
    revision: int
    gap_flag: bool
    emitted_at: Optional[float] = None  # wall-clock offset when produced (latency)
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seg_id": self.seg_id,
            "t_start": round(self.t_start, 4),
            "t_end": round(self.t_end, 4),
            "text_en": self.text_en,
            "text_zh": self.text_zh,
            "conf_asr": round(self.conf_asr, 4),
            "conf_mt": round(self.conf_mt, 4),
            "revision": self.revision,
            "gap_flag": self.gap_flag,
            "emitted_at": None if self.emitted_at is None else round(self.emitted_at, 4),
        }
