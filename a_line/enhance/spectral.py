"""B-chain: causal spectral subtraction / Wiener filtering.

This is the traditional enhancement baseline. It uses the course framing
(25 ms window, 10 ms hop) and a causal analysis window: the frame ending at the
current hop is built only from samples that have already arrived. A symmetric
window used this way needs the oldest hop of the overlap-add buffer, which
delays the output by `window - hop` samples (15 ms at 25/10). That delay is
declared through `algorithmic_delay_ms` so it can be added to the latency
budget instead of being hidden.

No training and no GPU are involved. The noise floor is estimated from a
rolling low percentile of past magnitude spectra, so the first frames adapt
quickly and the estimate follows slow changes in fan/room noise.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .base import Enhancer


class SpectralSubtractionEnhancer(Enhancer):
    name = "spectral"

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: float = 25.0,
        hop_ms: float = 10.0,
        n_fft: int = 512,
        method: str = "spectral",       # "spectral" | "wiener"
        over_subtraction: float = 2.0,
        floor: float = 0.05,
        noise_window_ms: float = 1000.0,
        noise_percentile: float = 10.0,
    ):
        if method not in ("spectral", "wiener"):
            raise ValueError("method must be 'spectral' or 'wiener'")
        self.sample_rate = int(sample_rate)
        self.window_ms = float(window_ms)
        self.hop_ms = float(hop_ms)
        self.win_len = int(round(self.sample_rate * self.window_ms / 1000.0))
        self.hop_len = int(round(self.sample_rate * self.hop_ms / 1000.0))
        if self.win_len <= 0 or self.hop_len <= 0:
            raise ValueError("window and hop must be positive")
        if n_fft < self.win_len:
            raise ValueError("n_fft must be at least the window length")
        self.n_fft = int(n_fft)
        self.method = method
        self.over_subtraction = float(over_subtraction)
        self.floor = float(floor)
        self.noise_window_ms = float(noise_window_ms)
        self.noise_percentile = float(noise_percentile)

        # Hann analysis/synthesis window; overlap-add is normalised by the sum
        # of squared windows, so no extra gain calibration is needed.
        self.window = np.hanning(self.win_len).astype(np.float32)
        self.algorithmic_delay_ms = (
            (self.win_len - self.hop_len) / self.sample_rate * 1000.0
        )
        self.reset()

    def reset(self) -> None:
        self._frame = np.zeros(self.win_len, dtype=np.float32)
        self._ola = np.zeros(self.win_len, dtype=np.float32)
        self._norm = np.zeros(self.win_len, dtype=np.float32)
        maxlen = max(5, int(round(self.noise_window_ms / self.hop_ms)))
        self._noise_hist: deque = deque(maxlen=maxlen)
        self._noise = None
        self._frames_seen = 0

    def _estimate_noise(self, mag: np.ndarray) -> np.ndarray:
        self._noise_hist.append(mag.astype(np.float32, copy=True))
        if len(self._noise_hist) < 5:
            self._noise = mag.copy()
        else:
            self._noise = np.percentile(
                np.stack(self._noise_hist), self.noise_percentile, axis=0
            ).astype(np.float32)
        return np.maximum(self._noise, 1e-8)

    def _mask(self, mag: np.ndarray, noise: np.ndarray) -> np.ndarray:
        if self.method == "wiener":
            noise_pow = noise * noise
            sig_pow = np.maximum(mag * mag - noise_pow, 0.0)
            mask = sig_pow / (sig_pow + noise_pow + 1e-12)
        else:
            clean = np.maximum(
                mag - self.over_subtraction * noise, self.floor * mag
            )
            mask = clean / (mag + 1e-12)
        return np.clip(mask, 0.0, 1.0).astype(np.float32)

    def _process_hop(self, hop: np.ndarray) -> np.ndarray:
        # Slide the causal analysis window: the frame ends at this hop.
        self._frame = np.concatenate(
            [self._frame[self.hop_len:], hop]
        ).astype(np.float32)
        spec = np.fft.rfft(self._frame * self.window, n=self.n_fft)
        mag = np.abs(spec).astype(np.float32)
        noise = self._estimate_noise(mag)
        mask = self._mask(mag, noise)
        y = np.fft.irfft(spec * mask, n=self.n_fft)[: self.win_len].astype(np.float32)

        self._ola += y * self.window
        self._norm += self.window * self.window
        out = self._ola[: self.hop_len] / np.maximum(
            self._norm[: self.hop_len], 1e-8
        )

        # Shift the overlap-add buffers one hop forward.
        self._ola = np.concatenate(
            [self._ola[self.hop_len:], np.zeros(self.hop_len, dtype=np.float32)]
        )
        self._norm = np.concatenate(
            [self._norm[self.hop_len:], np.zeros(self.hop_len, dtype=np.float32)]
        )
        self._frames_seen += 1
        # The first full window is incomplete; do not amplify its edges.
        if self._frames_seen <= self.win_len // self.hop_len:
            out = np.zeros_like(out)
        return out.astype(np.float32)

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        n = x.size
        if n == 0:
            return x
        pad = (-n) % self.hop_len
        if pad:
            x = np.pad(x, (0, pad))
        out = np.empty_like(x)
        for i in range(0, x.size, self.hop_len):
            out[i : i + self.hop_len] = self._process_hop(
                x[i : i + self.hop_len]
            )
        return out[:n]
