"""Audio sources for the B-line.

The pipeline never receives a whole decoded buffer. It pulls items from a
FrameSource, and a source hands out audio only up to a "released" horizon -- i.e.
only audio that has actually arrived. This enforces the manual's rule that the
real-time branch must not access audio that has not yet arrived.

Mirroring the manual, a source is one of:
  * a capture feed          -> MicFrameSource (stub: device integration point)
  * a replay of a recording -> WavFrameSource (keeps the original timestamps)
  * a deterministic signal  -> SyntheticFrameSource (tests / demo, no file needed)

Every source yields AudioFrame hops, and emits Gap items where the link dropped
audio, so the receiver only ever operates on data it actually received.
"""
from __future__ import annotations

import time
import wave
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import BLineConfig
from .types import AudioFrame, Gap

Item = Union[AudioFrame, Gap]


class FrameSource(ABC):
    """A pull-based frame stream.

    Contract: take() never returns an item whose end lies beyond released_sec.
    """

    realtime: bool = False

    @abstractmethod
    def take(self, max_items: int = 512) -> List[Item]:
        """Return up to max_items items that have arrived (AudioFrame or Gap)."""

    @abstractmethod
    def ended(self) -> bool:
        """True once every sample of the stream has been handed out."""

    @property
    @abstractmethod
    def duration_sec(self) -> float:
        ...

    @property
    @abstractmethod
    def released_sec(self) -> float:
        """How much of the stream has 'arrived' so far, in seconds."""


class PacedFrameSource(FrameSource):
    """Hop slicing, loss windows and arrival pacing shared by all concrete sources."""

    def __init__(
        self,
        cfg: BLineConfig,
        total_samples: int,
        loss_windows: Optional[Sequence[Tuple[float, float]]] = None,
        realtime: bool = False,
        speed: float = 1.0,
    ):
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.hop = cfg.hop_len
        self.total_samples = int(total_samples)
        self.loss = sorted(tuple(w) for w in (loss_windows or []))
        self.li = 0
        self.pos = 0            # next sample to hand out
        self.seq = 0
        self.realtime = bool(realtime)
        self.speed = max(1e-6, float(speed))
        self._t0 = time.monotonic()
        self._released = 0.0

    @property
    def duration_sec(self) -> float:
        return self.total_samples / self.sr

    @property
    def released_sec(self) -> float:
        return self._released

    @abstractmethod
    def _read(self, start: int, n: int) -> np.ndarray:
        """Return n float32 samples starting at absolute sample index 'start'."""

    def _horizon(self) -> float:
        if not self.realtime:
            return self.duration_sec
        return min(self.duration_sec, (time.monotonic() - self._t0) * self.speed)

    def ended(self) -> bool:
        return self.pos >= self.total_samples

    def take(self, max_items: int = 512) -> List[Item]:
        self._released = self._horizon()
        limit = min(int(self._released * self.sr), self.total_samples)
        out: List[Item] = []

        while self.pos < limit and len(out) < max_items:
            # step over loss windows that have already finished
            while self.li < len(self.loss) and self.pos >= int(self.loss[self.li][1] * self.sr):
                self.li += 1
            if self.li < len(self.loss):
                g0, g1 = self.loss[self.li]
                g0s, g1s = int(g0 * self.sr), int(g1 * self.sr)
                if g0s <= self.pos < g1s:
                    # A gap is only reported once the whole dropped interval has
                    # "arrived", so one loss window maps to exactly one Gap even
                    # when the release horizon is still inside it.
                    if limit < g1s and limit < self.total_samples:
                        break
                    gap_end = min(g1, self.duration_sec)
                    out.append(Gap(max(self.pos / self.sr, g0), gap_end))
                    self.pos = max(int(gap_end * self.sr), self.pos)
                    if self.pos >= g1s:
                        self.li += 1
                    continue

            if self.pos + self.hop <= limit:
                fs = self.pos / self.sr
                fe = (self.pos + self.hop) / self.sr
                out.append(AudioFrame(fs, fe, self._read(self.pos, self.hop), self.seq))
                self.seq += 1
                self.pos += self.hop
            else:
                break

        # trailing partial hop, only once the whole stream has been released
        if (
            len(out) < max_items
            and limit >= self.total_samples
            and self.pos < self.total_samples
            and self.pos + self.hop > self.total_samples
        ):
            fs = self.pos / self.sr
            fe = self.total_samples / self.sr
            out.append(AudioFrame(fs, fe, self._read(self.pos, self.total_samples - self.pos), self.seq))
            self.seq += 1
            self.pos = self.total_samples
        return out


class WavFrameSource(PacedFrameSource):
    """Replay a 16-bit PCM WAV, reading it one hop at a time.

    Nothing is preloaded: constructor reads zero samples, and each hop seeks and
    reads exactly hop_len samples. Only 16 kHz mono/stereo is accepted --- format
    conversion belongs to the A-line, which owns capture and normalisation.
    """

    def __init__(
        self,
        path: str,
        cfg: BLineConfig,
        loss_windows: Optional[Sequence[Tuple[float, float]]] = None,
        realtime: bool = False,
        speed: float = 1.0,
    ):
        self.path = path
        self._wav = wave.open(path, "rb")
        nch = self._wav.getnchannels()
        sw = self._wav.getsampwidth()
        sr = self._wav.getframerate()
        nframes = self._wav.getnframes()
        if sw != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported (got {sw * 8}-bit)")
        if sr != cfg.sample_rate:
            raise ValueError(
                f"{path}: expected {cfg.sample_rate} Hz, got {sr} Hz "
                "(resample upstream; the B-line takes a 16 kHz stream)"
            )
        if nch not in (1, 2):
            raise ValueError(f"{path}: expected mono or stereo, got {nch} channels")
        self._nch = nch
        self.samples_read = 0
        super().__init__(cfg, nframes, loss_windows, realtime, speed)

    def _read(self, start: int, n: int) -> np.ndarray:
        self._wav.setpos(int(start))
        raw = self._wav.readframes(int(n))
        got = len(raw) // (2 * self._nch)
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self._nch == 2:
            arr = arr.reshape(-1, 2).mean(axis=1)
        self.samples_read += got
        if got < n:
            arr = np.pad(arr, (0, int(n) - got))
        return arr.astype(np.float32, copy=False)

    def close(self) -> None:
        try:
            self._wav.close()
        except Exception:
            pass


class SyntheticFrameSource(PacedFrameSource):
    """Deterministic speech-like signal generated hop by hop, never materialised.

    Each utterance is an amplitude-modulated two-harmonic burst over silence, so
    an energy VAD can separate speech from silence. Noise is seeded by absolute
    sample position, so the signal does not depend on read order.
    """

    def __init__(
        self,
        cfg: BLineConfig,
        utterances: Sequence[Tuple[float, float, float]],
        total_duration: float,
        loss_windows: Optional[Sequence[Tuple[float, float]]] = None,
        realtime: bool = False,
        speed: float = 1.0,
        seed: int = 0,
    ):
        self.utterances = [(float(a), float(b), float(f)) for (a, b, f) in utterances]
        self.seed = int(seed)
        super().__init__(cfg, int(round(total_duration * cfg.sample_rate)), loss_windows, realtime, speed)

    def _read(self, start: int, n: int) -> np.ndarray:
        idx = np.arange(start, start + n, dtype=np.float64)
        t = idx / self.sr
        y = np.zeros(n, dtype=np.float64)
        for (t0, t1, f0) in self.utterances:
            m = (t >= t0) & (t < t1)
            if not np.any(m):
                continue
            tt = t[m]
            # envelope is relative to the utterance onset so every burst starts
            # at a non-zero amplitude and the VAD sees a sharp onset
            rel = tt - t0
            env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * rel)
            y[m] += (0.55 * np.sin(2 * np.pi * f0 * tt) + 0.30 * np.sin(2 * np.pi * 2 * f0 * tt)) * env
        rng = np.random.default_rng((self.seed, int(start)))
        y += 0.02 * rng.standard_normal(n)
        return y.astype(np.float32)


class MicFrameSource(FrameSource):
    """Live capture source (the manual's 'camera and microphone feed').

    Integration point, deliberately left unimplemented so no hidden
    whole-buffer path sneaks into the pipeline. To implement:

      * open the device at 16 kHz mono (sounddevice / soundcard),
      * in the audio callback, append samples to a bounded queue with the
        capture clock as timestamp,
      * take() drains complete hops from that queue as AudioFrame items,
      * ended() returns True when the caller signals stop.
    """

    def __init__(self, cfg: Optional[BLineConfig] = None, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device
        raise RuntimeError(
            "MicFrameSource is not implemented yet. Wire a 16 kHz mono capture "
            "device here; the pipeline needs nothing else (take() + ended())."
        )
