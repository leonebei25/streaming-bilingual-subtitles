"""Energy + zero-crossing VAD with hangover-based endpointing."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import BLineConfig
from .types import AudioFrame


class EnergyVAD:
    def __init__(self, cfg: BLineConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.speech = False
        self.seg_start = None       # onset of the current speech run (sec)
        self.last_voiced_end = None  # end of the last voiced frame (sec)
        self.hang = 0               # consecutive silent frames while in speech
        self._e: List[float] = []
        self._noise_floor = None

    def _features(self, x: np.ndarray) -> Tuple[float, float]:
        x = x.astype(np.float64)
        energy = float(np.mean(x * x))
        if len(x) < 2:
            return energy, 0.0
        signs = np.signbit(x).astype(np.int8)
        zcr = float(np.mean(signs[1:] != signs[:-1]))
        return energy, zcr

    def _threshold(self) -> float:
        if self._noise_floor is None:
            if len(self._e) >= self.cfg.vad_init_frames:
                self._noise_floor = float(np.quantile(self._e, 0.10))
            else:
                return 0.0 if not self._e else self._e[0] * self.cfg.vad_energy_ratio
        return max(self._noise_floor * self.cfg.vad_energy_ratio, 1e-9)

    def update(self, frame: AudioFrame) -> Tuple[bool, List[Tuple[float, float]]]:
        """Feed one hop. Returns (is_speech_now, list_of_endpoint_(t_start,t_end))."""
        energy, zcr = self._features(frame.data)
        self._e.append(energy)
        thr = self._threshold()
        voiced = energy > thr and zcr < self.cfg.vad_zcr_max

        endpoints: List[Tuple[float, float]] = []
        if voiced:
            if not self.speech:
                self.speech = True
                self.seg_start = frame.start_sec
            self.last_voiced_end = frame.end_sec
            self.hang = 0
        elif self.speech:
            self.hang += 1
            if self.hang * (self.cfg.hop_ms / 1000.0) >= self.cfg.vad_hangover_ms / 1000.0:
                endpoints.append((self.seg_start, self.last_voiced_end))
                self.speech = False
                self.seg_start = None
                self.last_voiced_end = None
                self.hang = 0
        return self.speech, endpoints
