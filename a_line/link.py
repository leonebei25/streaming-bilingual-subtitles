"""Constrained-link simulation: packetization, delay, jitter, loss, playout.

The course asks for 2 Mbps / 512 kbps links, 40 ms base delay plus 0-100 ms
jitter, 0/1/5% packet loss and one 200 ms burst. This module models the part
that changes the audio timeline, not the bitrate itself: it turns 10 ms frames
into 20 ms packets, drops or delays packets, and releases frames at a fixed
playout deadline into a jitter buffer.

A loss (or a packet that arrives after its playout deadline) becomes one Gap
covering the packet; consecutive drops are merged when they leave the buffer.
Everything is driven by the inner source's `released_sec`, so the simulator
never sees audio that has not arrived -- the same rule as the rest of the
pipeline.

The random stream is seeded, so A, B and C can each build their own LinkSource
with the same seed and see exactly the same loss/jitter trace. Do not share one
LinkSource instance between chains: each chain must consume its own copy.
"""
from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from b_line.sources import FrameSource, Item
from b_line.types import AudioFrame, Gap


@dataclass
class LinkConfig:
    packet_ms: float = 20.0
    base_delay_ms: float = 40.0
    jitter_ms: float = 0.0
    loss_rate: float = 0.0
    burst_windows: Sequence[Tuple[float, float]] = field(default_factory=tuple)
    jitter_buffer_ms: float = 120.0
    seed: int = 0

    @property
    def playout_delay_ms(self) -> float:
        return self.base_delay_ms + self.jitter_buffer_ms


class LinkSource(FrameSource):
    """Wrap a FrameSource with a packet link and a playout buffer."""

    def __init__(self, inner: FrameSource, cfg: LinkConfig, sample_rate: int = 16000):
        self.inner = inner
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.packet_len = max(1, int(round(self.sample_rate * cfg.packet_ms / 1000.0)))
        self.rng = random.Random(cfg.seed)

        self._frames: List[AudioFrame] = []
        self._samples = 0
        self._scheduled: List[Tuple[float, int, str, object]] = []
        self._order = 0
        self._inner_done = False

    @property
    def realtime(self) -> bool:
        return self.inner.realtime

    @property
    def duration_sec(self) -> float:
        return self.inner.duration_sec

    @property
    def released_sec(self) -> float:
        if self._inner_done:
            # The tail is drained after the source ends; expose the playout
            # horizon so tail latency is not silently discounted.
            return self.inner.duration_sec + self.cfg.playout_delay_ms / 1000.0
        return self.inner.released_sec

    def ended(self) -> bool:
        return self._inner_done and not self._scheduled and not self._frames

    def take(self, max_items: int = 512) -> List[Item]:
        out: List[Item] = []
        while len(out) < max_items:
            now = float("inf") if self._inner_done else self.inner.released_sec
            if self._scheduled and self._scheduled[0][0] <= now + 1e-9:
                _, _, kind, payload = heapq.heappop(self._scheduled)
                if kind == "packet":
                    out.extend(payload)  # type: ignore[arg-type]
                else:
                    self._emit_gap(out, payload)  # type: ignore[arg-type]
                continue
            if self._inner_done:
                break

            items = self.inner.take(max_items=max(1, max_items - len(out)))
            if not items:
                if self.inner.ended():
                    self._inner_done = True
                    self._flush_packet()
                    continue
                break
            for it in items:
                if isinstance(it, Gap):
                    self._flush_packet()
                    self._schedule_gap(it.start_sec, it.end_sec)
                else:
                    self._frames.append(it)
                    self._samples += len(it.data)
                    if self._samples >= self.packet_len:
                        self._flush_packet()
        return out

    # -- packet scheduling -------------------------------------------------

    def _flush_packet(self) -> None:
        if not self._frames:
            return
        frames = self._frames
        self._frames = []
        self._samples = 0
        start = frames[0].start_sec
        end = frames[-1].end_sec

        # Always draw both the loss and jitter values, even for a lost packet,
        # so the trace is identical across chains that use the same seed.
        lost = self.rng.random() < self.cfg.loss_rate
        if any(start < b1 and end > b0 for (b0, b1) in self.cfg.burst_windows):
            lost = True
        jitter = self.rng.uniform(0.0, self.cfg.jitter_ms) / 1000.0

        arrival = end + self.cfg.base_delay_ms / 1000.0 + jitter
        playout = end + self.cfg.playout_delay_ms / 1000.0
        if lost or arrival > playout + 1e-9:
            self._schedule_gap(start, end, playout)
        else:
            self._push(playout, "packet", frames)

    def _schedule_gap(self, start: float, end: float, playout: Optional[float] = None) -> None:
        if playout is None:
            playout = end + self.cfg.playout_delay_ms / 1000.0
        self._push(playout, "gap", (start, end))

    def _push(self, playout: float, kind: str, payload: object) -> None:
        heapq.heappush(self._scheduled, (playout, self._order, kind, payload))
        self._order += 1

    @staticmethod
    def _emit_gap(out: List[Item], payload: Tuple[float, float]) -> None:
        start, end = payload
        if out and isinstance(out[-1], Gap):
            last = out[-1]
            if abs(last.end_sec - start) < 1e-9:
                out[-1] = Gap(last.start_sec, end, last.reason)
                return
            if start <= last.end_sec + 1e-9:
                out[-1] = Gap(last.start_sec, max(last.end_sec, end), last.reason)
                return
        out.append(Gap(start, end))

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()
