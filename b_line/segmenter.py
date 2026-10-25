"""Stable-prefix segmentation.

The recognizer rewrites its partial transcript; the segmenter decides which part
of that text is safe to commit. A finalized segment is emitted when its utterance
has ended (VAD endpoint) and the recognizer has had enough time to stabilize, or
earlier if the segment exceeds the configured maximum length.
"""
from __future__ import annotations

from typing import List, Optional

from .config import BLineConfig
from .types import ASRUpdate, Segment


class StablePrefixSegmenter:
    def __init__(self, cfg: BLineConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.session_stable = ""
        self.emitted = 0
        self.cur_start: Optional[float] = None
        self.cur_end: Optional[float] = None
        self.last_conf = 1.0
        self.pending_endpoint_te: Optional[float] = None
        self.seg_id = 0
        self.pending: List[Segment] = []

    def note_speech(self, is_speech: bool, now_sec: float, onset_sec: Optional[float] = None) -> None:
        """Track the current speech run at frame granularity.

        Called for every hop so that t_start/t_end are frame-accurate rather than
        chunk-aligned. 'onset_sec' is the VAD's frame-exact speech onset.
        """
        if is_speech:
            if self.cur_start is None:
                self.cur_start = onset_sec if onset_sec is not None else now_sec
            self.cur_end = now_sec

    def update(self, update: ASRUpdate, now_sec: float) -> None:
        self.last_conf = update.confidence
        self.session_stable = update.text[:update.stable_chars]
        if self.pending_endpoint_te is not None:
            delay = self.cfg.stability_delay_ms / 1000.0
            if now_sec >= self.pending_endpoint_te + delay:
                self._finalize(self.pending_endpoint_te)
                self.pending_endpoint_te = None
        self._force_split()

    def on_endpoint(self, te: float) -> None:
        """Mark that speech ended at 'te'; finalize once the ASR stabilizes."""
        self.pending_endpoint_te = te

    def flush(self, te: Optional[float] = None) -> None:
        """Finalize immediately (used at gaps / end of stream)."""
        t = te if te is not None else (self.cur_end if self.cur_end is not None else 0.0)
        self._finalize(t)
        self.pending_endpoint_te = None

    def pop(self) -> List[Segment]:
        out = self.pending
        self.pending = []
        return out

    def _force_split(self) -> None:
        while len(self.session_stable) - self.emitted >= self.cfg.max_seg_chars:
            start = self.emitted
            end = start + self.cfg.max_seg_chars
            chunk = self.session_stable[start:end]
            sp = chunk.rfind(" ")
            if sp > self.cfg.max_seg_chars // 2:
                end = start + sp
                chunk = chunk[:sp]
            if not chunk.strip():
                self.emitted = end
                continue
            s = self.cur_start if self.cur_start is not None else 0.0
            e = self.cur_end if self.cur_end is not None else s
            self.pending.append(Segment(self.seg_id, s, e, chunk.strip(), self.last_conf))
            self.seg_id += 1
            self.emitted = end

    def _finalize(self, te: float) -> None:
        text = self.session_stable[self.emitted:]
        self.emitted = len(self.session_stable)
        if text.strip():
            s = self.cur_start if self.cur_start is not None else te
            self.pending.append(Segment(self.seg_id, s, te, text.strip(), self.last_conf))
            self.seg_id += 1
        self.cur_start = None
        self.cur_end = None
