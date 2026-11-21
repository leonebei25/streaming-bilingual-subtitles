"""Audio sources for the B-line.

The pipeline never receives a whole decoded buffer. It pulls items from a
FrameSource, and a source hands out audio only up to a "released" horizon -- i.e.
only audio that has actually arrived. This enforces the manual's rule that the
real-time branch must not access audio that has not yet arrived.

Mirroring the manual, a source is one of:
  * a live push-fed feed   -> MicFrameSource / StreamingFrameSource (never preloaded)
  * a replay of a recording -> WavFrameSource (keeps the original timestamps)
  * a deterministic signal  -> SyntheticFrameSource (tests / demo, no file needed)

Every source yields AudioFrame hops, and emits Gap items where the link dropped
audio, so the receiver only ever operates on data it actually received.
"""
from __future__ import annotations

import collections
import sys
import threading
import time
import wave
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .config import BLineConfig
from .types import AudioFrame, Gap

Item = Union[AudioFrame, Gap]


def _normalize_device(device):
    """sounddevice selects by int index or by name substring; argparse gives str.

    A bare number (``"10"``) must become the index ``10`` or sounddevice looks
    for a device whose *name* contains "10". Names and None pass through.
    """
    if isinstance(device, str):
        d = device.strip()
        if d == "":
            return None
        return int(d) if d.lstrip("+-").isdigit() else d
    return device


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


class StreamingFrameSource(FrameSource):
    """Push-fed, real-time frame source (microphone / pipe / network).

    A producer -- a sounddevice callback, a socket reader thread, an ffmpeg
    pipe -- hands raw mono audio to :meth:`push`; the pipeline pulls completed
    hops from :meth:`take`. Only audio that has actually arrived is ever handed
    out, and only a bounded window of it is buffered, so the whole stream is
    never held in memory: this is the manual's native streaming input, not a
    preloaded file.

    If the consumer falls behind and the buffer would exceed ``max_buffer_sec``,
    the oldest samples are dropped and reported as a :class:`Gap`, so the
    receiver knows the timeline has a hole rather than silently stitching audio.

    ``push`` is safe to call from any thread (the audio callback, a reader
    thread). ``finish`` marks the stream ended; ``limit_sec > 0`` ends it
    automatically once that much audio has arrived.
    """

    realtime = True

    def __init__(self, cfg: BLineConfig, max_buffer_sec: float = 20.0,
                 limit_sec: float = 0.0):
        self.cfg = cfg
        self.sr = cfg.sample_rate
        self.hop = cfg.hop_len
        self._max_buffered = max(self.hop, int(max_buffer_sec * self.sr))
        self._limit_sec = float(limit_sec)
        self._lock = threading.Lock()
        self._chunks: "collections.deque" = collections.deque()  # float32, oldest first
        self._buffered = 0        # samples currently buffered
        self._received = 0        # samples ever pushed
        self._next = 0            # absolute index of the next sample to hand out
        self._gaps: "collections.deque" = collections.deque()    # pending (start, end)
        self._finished = False
        self._seq = 0
        self._dropped = 0

    # -- producer side --------------------------------------------------
    def push(self, data, sample_rate: Optional[int] = None,
             channels: int = 1) -> None:
        """Append mono audio at ``cfg.sample_rate``; int16 or float32 accepted."""
        arr = np.asarray(data)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if channels and int(channels) > 1:
            arr = arr.reshape(-1, int(channels)).mean(axis=1)
        arr = np.ascontiguousarray(arr.reshape(-1))
        if sample_rate is not None and int(sample_rate) != self.sr:
            raise ValueError(
                f"stream sample rate {sample_rate} != {self.sr}; resample to "
                f"{self.sr} Hz upstream -- the B-line takes a 16 kHz stream")
        if arr.size == 0:
            return
        with self._lock:
            self._chunks.append(arr)
            self._buffered += arr.size
            self._received += arr.size
            # Bound memory. Dropping the oldest audio is reported as a gap so a
            # slow consumer degrades to "lost audio", never "unbounded buffer".
            while self._buffered > self._max_buffered and self._chunks:
                need = self._buffered - self._max_buffered
                oldest = self._chunks[0]
                d = min(oldest.size, need)
                if d >= oldest.size:
                    self._chunks.popleft()
                else:
                    self._chunks[0] = oldest[d:]
                self._buffered -= d
                self._gaps.append((self._next / self.sr,
                                   (self._next + d) / self.sr))
                self._next += d
                self._dropped += d
            if self._limit_sec > 0 and self._received / self.sr >= self._limit_sec:
                self._finished = True

    def finish(self) -> None:
        """Signal end of stream; buffered hops are still handed out first."""
        with self._lock:
            self._finished = True

    @property
    def dropped_samples(self) -> int:
        return self._dropped

    # -- consumer side --------------------------------------------------
    @property
    def duration_sec(self) -> float:
        with self._lock:
            return self._received / self.sr

    @property
    def released_sec(self) -> float:
        with self._lock:
            return self._received / self.sr

    def ended(self) -> bool:
        with self._lock:
            return self._finished and self._buffered < self.hop

    def _pop_front(self, n: int) -> np.ndarray:
        """Remove exactly n leading samples (lock held by the caller)."""
        parts, got = [], 0
        while got < n and self._chunks:
            c = self._chunks[0]
            take = min(c.size, n - got)
            parts.append(c[:take])
            got += take
            if take >= c.size:
                self._chunks.popleft()
            else:
                self._chunks[0] = c[take:]
        self._buffered -= got
        if len(parts) == 1:
            return parts[0]
        return np.concatenate(parts)

    def take(self, max_items: int = 512) -> List[Item]:
        out: List[Item] = []
        with self._lock:
            while self._gaps and len(out) < max_items:
                s, e = self._gaps.popleft()
                out.append(Gap(s, e))
            n_hops = min(self._buffered // self.hop, max_items - len(out))
            for _ in range(n_hops):
                block = self._pop_front(self.hop)
                fs = self._next / self.sr
                fe = (self._next + self.hop) / self.sr
                out.append(AudioFrame(fs, fe, block, self._seq))
                self._seq += 1
                self._next += self.hop
            # Trailing partial hop, only once the producer says it is done.
            if self._finished and self._buffered > 0 and len(out) < max_items:
                rem = self._buffered
                block = self._pop_front(rem)
                fs = self._next / self.sr
                fe = (self._next + rem) / self.sr
                out.append(AudioFrame(fs, fe, block, self._seq))
                self._seq += 1
                self._next += rem
        return out


class MicFrameSource(StreamingFrameSource):
    """Live microphone capture through ``sounddevice``.

    Opens the default (or a named/indexed) input device at 16 kHz mono and
    pushes every callback block into the streaming buffer, so the pipeline
    consumes the microphone in real time -- nothing is preloaded. If a device
    (e.g. Stereo Mix, a mic array) only exposes its native rate, it is opened at
    that rate and each block is resampled to 16 kHz automatically.

    Stop with Ctrl+C or ``--max-seconds``. ``list_devices()`` prints choices.
    """

    def __init__(self, cfg: BLineConfig, device=None, max_buffer_sec: float = 20.0,
                 limit_sec: float = 0.0, blocksize: int = 0,
                 force_rate: Optional[int] = None,
                 force_channels: Optional[int] = None):
        super().__init__(cfg, max_buffer_sec=max_buffer_sec, limit_sec=limit_sec)
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "MicFrameSource needs sounddevice: `pip install sounddevice`"
            ) from exc
        # argparse hands the device in as a string, but sounddevice treats a
        # string as a device-*name* substring. Turn a bare number into an index.
        device = _normalize_device(device)
        self.device = device
        self.capture_rate = cfg.sample_rate
        self._channels = 1
        self._resample = None

        # Some inputs (Stereo Mix, mic arrays) reject mono or 16 kHz: a stereo
        # device answers "Invalid number of channels" for 1 channel, and some
        # only offer 44.1/48 kHz. Try the supported shapes in order and downmix /
        # resample to 16 kHz mono internally.
        try:
            info = sd.query_devices(device, "input")
            max_in = int(info["max_input_channels"])
            native = int(round(float(info["default_samplerate"])))
        except Exception:
            max_in, native = 0, cfg.sample_rate
        channel_opts = []
        for c in ([force_channels] if force_channels else [1, 2, max_in]):
            if c and c > 0 and c not in channel_opts:
                channel_opts.append(c)
        rate_opts = []
        for r in ([force_rate] if force_rate else [cfg.sample_rate, native]):
            if r and r > 0 and r not in rate_opts:
                rate_opts.append(r)

        last_exc = None
        opened = False
        attempts = []
        for rate in rate_opts:
            for ch in channel_opts:
                attempts.append(f"{ch}ch@{rate}Hz")
                try:
                    raw = sd.RawInputStream(
                        samplerate=rate, channels=ch, dtype="int16",
                        blocksize=blocksize, device=device, callback=self._callback)
                except Exception as exc:  # pragma: no cover - depends on the machine
                    last_exc = exc
                    continue
                # Set the shape before start() so the first callback is correct.
                self._channels = ch
                self.capture_rate = rate
                if rate != cfg.sample_rate:
                    try:
                        from math import gcd
                        from scipy.signal import resample_poly
                    except ImportError as exc:
                        raw.close()
                        raise RuntimeError(
                            "this input only runs at a rate other than 16 kHz and "
                            "resampling needs scipy: `pip install scipy`") from exc
                    g = gcd(rate, cfg.sample_rate)
                    self._resample = (resample_poly, cfg.sample_rate // g, rate // g)
                try:
                    raw.start()
                except Exception as exc:  # pragma: no cover - depends on the machine
                    last_exc = exc
                    self._resample = None
                    try:
                        raw.close()
                    except Exception:
                        pass
                    continue
                self._raw = raw
                opened = True
                break
            if opened:
                break
        if not opened:
            raise RuntimeError(
                f"could not open audio input {device!r}; tried "
                f"{', '.join(attempts)}. Last error: {last_exc}. "
                "'Invalid device' = disabled/busy/wrong host API; "
                "'Invalid number of channels' = it needs another channel count. "
                "Force a shape with --mic-channels / --mic-rate, or pick another "
                "index from --mic-list.") from last_exc

    def _callback(self, indata, frames, time_info, status):  # audio thread
        x = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32) / 32768.0
        if self._channels > 1:
            x = x.reshape(-1, self._channels).mean(axis=1)
        if self._resample is not None:
            fn, up, down = self._resample
            x = fn(x, up, down).astype(np.float32, copy=False)
        self.push(x)

    def stop(self) -> None:
        try:
            self._raw.stop()
            self._raw.close()
        finally:
            self.finish()

    @staticmethod
    def list_devices():
        import sounddevice as sd
        return sd.query_devices()


class PcmStreamSource(StreamingFrameSource):
    """Read 16 kHz mono signed-16 PCM from a binary stream (stdin / socket / pipe).

    A daemon thread pulls raw little-endian int16 blocks and pushes them, so the
    pipeline processes audio as it arrives. This is the same streaming contract
    the microphone uses and lets you test the live path without a device, e.g.
    feeding a socket, a named pipe, or ``ffmpeg ... -f s16le -`` on a platform
    with true binary pipes.
    """

    def __init__(self, cfg: BLineConfig, stream=None, max_buffer_sec: float = 20.0,
                 limit_sec: float = 0.0, read_bytes: int = 4096):
        super().__init__(cfg, max_buffer_sec=max_buffer_sec, limit_sec=limit_sec)
        self._stream = stream if stream is not None else sys.stdin.buffer
        self._read_bytes = max(2, int(read_bytes))
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        try:
            while True:
                raw = self._stream.read(self._read_bytes)
                if not raw:
                    break
                if len(raw) % 2:            # keep sample alignment across reads
                    raw = raw[: len(raw) - 1]
                self.push(np.frombuffer(raw, dtype=np.int16))
        finally:
            self.finish()

    def stop(self) -> None:
        self.finish()
