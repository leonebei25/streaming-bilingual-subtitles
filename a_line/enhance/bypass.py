"""A-chain: no enhancement, audio passes through untouched."""
from __future__ import annotations

import numpy as np

from .base import Enhancer


class BypassEnhancer(Enhancer):
    name = "none"
    algorithmic_delay_ms = 0.0

    def reset(self) -> None:
        pass

    def process(self, samples: np.ndarray) -> np.ndarray:
        return np.asarray(samples, dtype=np.float32)
