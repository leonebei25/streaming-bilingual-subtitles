"""Audio I/O and DSP primitives for the B-line.

The pipeline works on 16 kHz mono float32 audio. This module handles loading,
resampling, windowed framing (25 ms / 10 ms), and synthesizing test signals.
"""
from __future__ import annotations

import wave
from typing import Iterable, Tuple

import numpy as np

from .config import BLineConfig


def load_wav_mono16(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load a 16-bit PCM WAV and return (float32 mono samples, sample rate)."""
    with wave.open(path, "rb") as w:
        nch = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        nframes = w.getnframes()
        raw = w.readframes(nframes)
    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported (got {sampwidth * 8}-bit).")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch == 2:
        x = x.reshape(-1, 2).mean(axis=1)
    elif nch != 1:
        raise ValueError(f"Expected mono or stereo WAV (got {nch} channels).")
    if sr != target_sr:
        x = resample_linear(x, sr, target_sr)
    return x.astype(np.float32), target_sr


def save_wav_mono16(path: str, x: np.ndarray, sr: int = 16000) -> None:
    y = np.clip(x, -1.0, 1.0)
    pcm = (y * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def resample_linear(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    n_out = int(round(len(x) * sr_to / sr_from))
    t = np.arange(n_out) * (sr_from / sr_to)
    i0 = np.floor(t).astype(np.int64)
    frac = (t - i0).astype(np.float32)
    i0 = np.minimum(i0, len(x) - 1)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    return x[i0] * (1.0 - frac) + x[i1] * frac


def stft_frames(x: np.ndarray, cfg: BLineConfig) -> np.ndarray:
    """25 ms Hann window, 10 ms hop. Returns (n_frames, window_len) float32."""
    hop = cfg.hop_len
    win = cfg.window_len
    if len(x) < win:
        return np.zeros((0, win), dtype=np.float32)
    w = np.hanning(win).astype(np.float32)
    n_frames = 1 + (len(x) - win) // hop
    frames = np.empty((n_frames, win), dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        frames[i] = x[s:s + win] * w
    return frames


def synthesize_test_audio(
    utterances: Iterable[Tuple[float, float, float]],
    total_duration: float,
    sr: int = 16000,
    seed: int = 0,
) -> np.ndarray:
    """Build a deterministic 'speech-like' signal.

    Each utterance is a short amplitude-modulated tone burst (two harmonics plus
    light noise) placed on top of digital silence, so an energy VAD can separate
    speech from silence.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(int(round(total_duration * sr)), dtype=np.float32)
    for (t0, t1, f0) in utterances:
        n0 = int(round(t0 * sr))
        n1 = int(round(t1 * sr))
        if n1 > len(x):
            n1 = len(x)
        if n1 <= n0:
            continue
        t = np.arange(n1 - n0, dtype=np.float32) / sr
        env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
        y = (0.55 * np.sin(2 * np.pi * f0 * t) + 0.30 * np.sin(2 * np.pi * 2 * f0 * t)) * env
        y += 0.02 * rng.standard_normal(n1 - n0).astype(np.float32)
        x[n0:n1] += y.astype(np.float32)
    return x
