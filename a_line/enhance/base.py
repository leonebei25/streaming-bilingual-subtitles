"""Common interface for the A/B/C enhancement stages.

An enhancer consumes float32 mono samples at the pipeline sample rate and
returns the same number of samples. It may keep internal state, but it must be
causal: the output at time t may depend only on samples up to t (plus whatever
algorithmic delay it declares). `algorithmic_delay_ms` is the delay the stage
adds; the report must record it separately from network and buffer delays.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Enhancer(ABC):
    name: str = "base"

    #: Delay between the input sample and the sample the stage outputs for it.
    algorithmic_delay_ms: float = 0.0

    def reset(self) -> None:
        """Forget all state. Called at construction and after a packet gap."""

    @abstractmethod
    def process(self, samples: np.ndarray) -> np.ndarray:
        """Enhance a block of mono float32 samples; return the same length."""

    def close(self) -> None:
        """Release resources (files, sessions). Safe to call twice."""
