"""Wrap a FrameSource with an enhancement stage.

The pipeline must keep seeing a normal stream: frames with their original
timestamps and sequence numbers, and Gap items where the link dropped audio.
This wrapper buffers complete blocks (default 320 ms), runs the enhancer once
per block, then re-slices the output back into the original frames. Timestamps
are never rewritten, so the segmenter's frame-accurate timing is preserved.

`released_sec` deliberately reports the inner source's arrival horizon, not
the horizon of the enhanced audio. The difference is the block buffering the
enhancer adds, and that delay must show up in end-to-end latency rather than
being hidden.
"""
from __future__ import annotations

from typing import List

import numpy as np

from b_line.sources import FrameSource, Item
from b_line.types import AudioFrame, Gap

from .base import Enhancer


class EnhancedFrameSource(FrameSource):
    def __init__(
        self,
        inner: FrameSource,
        enhancer: Enhancer,
        sample_rate: int = 16000,
        block_ms: float = 320.0,
    ):
        self.inner = inner
        self.enhancer = enhancer
        self.sample_rate = int(sample_rate)
        self.block_len = max(1, int(round(self.sample_rate * block_ms / 1000.0)))

        self._pending: List[Item] = []
        self._frames: List[AudioFrame] = []
        self._samples = 0
        self._inner_done = False

    @property
    def realtime(self) -> bool:
        return self.inner.realtime

    @property
    def duration_sec(self) -> float:
        return self.inner.duration_sec

    @property
    def released_sec(self) -> float:
        # Arrival horizon, including the block delay the enhancer introduces.
        return self.inner.released_sec

    def ended(self) -> bool:
        return self._inner_done and not self._pending and not self._frames

    def take(self, max_items: int = 512) -> List[Item]:
        out: List[Item] = []
        while len(out) < max_items:
            if self._pending:
                out.append(self._pending.pop(0))
                continue
            if self._inner_done:
                break
            items = self.inner.take(max_items=max(1, max_items - len(out)))
            if not items:
                if self.inner.ended():
                    self._inner_done = True
                    self._flush(final=True)
                    continue
                break
            for it in items:
                if isinstance(it, Gap):
                    self._flush(final=False)
                    self._pending.append(it)
                    self.enhancer.reset()
                else:
                    self._frames.append(it)
                    self._samples += len(it.data)
                    if self._samples >= self.block_len:
                        self._flush(final=False)
        return out

    def _flush(self, final: bool) -> None:
        if not self._frames:
            return
        if not final and self._samples < self.block_len:
            return

        lengths = [len(f.data) for f in self._frames]
        x = np.concatenate([f.data for f in self._frames]).astype(np.float32)
        y = self.enhancer.process(x)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(y) < len(x):
            y = np.pad(y, (0, len(x) - len(y)))
        elif len(y) > len(x):
            y = y[: len(x)]

        pos = 0
        for frame, n in zip(self._frames, lengths):
            piece = y[pos : pos + n]
            self._pending.append(
                AudioFrame(
                    start_sec=frame.start_sec,
                    end_sec=frame.end_sec,
                    data=piece,
                    seq=frame.seq,
                )
            )
            pos += n
        self._frames = []
        self._samples = 0

    def close(self) -> None:
        self.enhancer.close()
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()
