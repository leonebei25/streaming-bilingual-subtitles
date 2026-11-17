"""Enhancement stages for the A/B/C comparison.

A: BypassEnhancer (no processing)
B: SpectralSubtractionEnhancer (traditional, no training)
C: a trained causal mask network, added later behind the same interface.
"""

from .base import Enhancer
from .bypass import BypassEnhancer
from .spectral import SpectralSubtractionEnhancer

__all__ = ["Enhancer", "BypassEnhancer", "SpectralSubtractionEnhancer"]
