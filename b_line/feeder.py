"""FrameFeeder: the only object allowed to hand audio frames to the pipeline.

It releases audio only up to an explicit cursor (the 'arrived' horizon). Calling
drain() therefore can never return a frame that has not arrived yet -- this is
the concrete enforcement of the project rule that the real-time branch must not
read audio that has not arrived.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple, Union

import numpy as np

from .config import BLineConfig
from .types import AudioFrame, Gap

Item = Union[AudioFrame, Gap]


class FrameFeeder:
    def __init__(
        self,
        samples: np.ndarray,
        sr: int,
        cfg: BLineConfig,
        loss_windows: Optional[List[Tuple[float, float]]] = None,
    ):
        self.samples = np.asarray(samples, dtype=np.float32)
        self.sr = sr
        self.cfg = cfg
        self.hop = cfg.hop_len
        self.loss = sorted(loss_windows or [])
        self.li = 0
        self.pos = 0            # next sample to emit
        self.released_sec = 0.0
        self.seq = 0

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sr

    def release(self, upto_sec: float) -> None:
        self.released_sec = max(self.released_sec, min(upto_sec, self.duration_sec))

    def release_all(self) -> None:
        self.release(self.duration_sec)

    def exhausted(self) -> bool:
        return self.pos >= len(self.samples)

    def drain(self) -> List[Item]:
        """Return every fully-arrived hop (or gap) up to the released horizon."""
        limit = min(int(self.released_sec * self.sr), len(self.samples))
        out: List[Item] = []
        while self.pos < limit:
            # skip past any loss window that has already ended
            while self.li < len(self.loss) and self.pos >= int(self.loss[self.li][1] * self.sr):
                self.li += 1
            if self.li < len(self.loss):
                g0, g1 = self.loss[self.li]
                g0s, g1s = int(g0 * self.sr), int(g1 * self.sr)
                if g0s <= self.pos < g1s:
                    gap_end = min(g1, limit / self.sr)
                    out.append(Gap(max(self.pos / self.sr, g0), gap_end))
                    self.pos = int(gap_end * self.sr)
                    if self.pos >= g1s:
                        self.li += 1
                    continue
            # emit a full hop if it is completely within the horizon
            if self.pos + self.hop <= limit:
                fs = self.pos / self.sr
                fe = (self.pos + self.hop) / self.sr
                out.append(AudioFrame(fs, fe, self.samples[self.pos:self.pos + self.hop].copy(), self.seq))
                self.seq += 1
                self.pos += self.hop
            else:
                break
        # trailing partial frame, only once the whole file has been released
        if limit >= len(self.samples) and self.pos < len(self.samples):
            fs = self.pos / self.sr
            fe = len(self.samples) / self.sr
            out.append(AudioFrame(fs, fe, self.samples[self.pos:].copy(), self.seq))
            self.seq += 1
            self.pos = len(self.samples)
        return out

    def replay(self, speed: float = 1.0):
        """Yield items as they arrive in (simulated) real time."""
        wall0 = time.monotonic()
        while True:
            self.release((time.monotonic() - wall0) * speed)
            items = self.drain()
            for it in items:
                yield it
            if self.exhausted():
                break
            time.sleep(0.004)
