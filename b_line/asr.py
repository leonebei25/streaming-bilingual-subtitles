"""Recognizer backends.

All backends implement the same streaming contract: feed chunks of audio, get an
incremental ASRUpdate. ScriptedASR is a deterministic stand-in used to develop
and test the pipeline without a GPU; Qwen3ASR documents the intended real
integration and fails loudly until its environment is configured.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from .config import BLineConfig
from .types import ASRUpdate

# A script entry: (t0, t1, text, confidence)
ScriptEntry = Tuple[float, float, str, float]


class ASRBackend(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def stream(self, samples: np.ndarray, start_sec: float, end_sec: float) -> ASRUpdate:
        ...

    def finalize(self) -> ASRUpdate:
        return ASRUpdate(text="", stable_chars=0, confidence=0.0, finalized=True)


class ScriptedASR(ASRBackend):
    """Reveal a known transcript as the audio timeline advances.

    Models the two streaming behaviours that matter for the pipeline: the partial
    text grows as more audio arrives, and a prefix becomes stable only after a
    configurable delay following the end of its utterance.
    """

    def __init__(self, script: List[ScriptEntry], cfg: BLineConfig):
        self.script = sorted(script, key=lambda u: u[0])
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.covered_sec = 0.0

    def _reveal(self):
        partial = ""
        stable = 0
        conf = 1.0
        delay = self.cfg.stability_delay_ms / 1000.0
        for (t0, t1, text, c) in self.script:
            if self.covered_sec < t0:
                break
            if self.covered_sec >= t1:
                partial += text
                if self.covered_sec >= t1 + delay:
                    stable += len(text)
                else:
                    conf = min(conf, 0.8)
            else:
                k = max(1, int(len(text) * (self.covered_sec - t0) / max(1e-6, t1 - t0)))
                partial += text[:k]
                conf = min(conf, 0.7)
                break  # the current utterance is still being spoken
        return partial, stable, conf

    def stream(self, samples, start_sec, end_sec) -> ASRUpdate:
        self.covered_sec = max(self.covered_sec, end_sec)
        text, stable, conf = self._reveal()
        return ASRUpdate(text=text, stable_chars=stable, confidence=conf)

    def finalize(self) -> ASRUpdate:
        self.covered_sec = float("inf")
        text, stable, conf = self._reveal()
        return ASRUpdate(text=text, stable_chars=stable, confidence=conf, finalized=True)


class Qwen3ASR(ASRBackend):
    """Adapter for Qwen3-ASR streaming inference.

    Intended integration points (to be completed once the GPU environment exists):

    * Streaming inference is only available through the vLLM backend.
    * Streaming inference does not return timestamps -- keep the
      chunk -> offset bookkeeping already done in FrameFeeder.
    * Keep the decoding parameters (chunk size, lookahead, max tokens) frozen
      and record them in the config snapshot alongside the model checkpoint.
    """

    def __init__(self, model: str = "Qwen/Qwen3-ASR-0.6B", cfg: Optional[BLineConfig] = None, **backend_kwargs):
        self.model = model
        self.cfg = cfg
        self._backend_kwargs = backend_kwargs
        raise RuntimeError(
            "Qwen3-ASR is not wired in this initial version. "
            "Install the qwen-asr package and a vLLM backend, then implement "
            "reset() and stream() against the streaming client."
        )
