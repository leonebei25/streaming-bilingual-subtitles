"""Translation backends.

Translation is kept as a separate module from recognition (a course
requirement) so the A/B/C comparison can hold ASR and MT fixed and change only
the audio enhancement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple


class MTBackend(ABC):
    @abstractmethod
    def translate(self, text_en: str, context: Sequence[str] = ()) -> Tuple[str, float]:
        """Return (translation, confidence)."""


class IdentityMT(MTBackend):
    """Deterministic stand-in that wraps the source text; lets the pipeline run
    end-to-end before a real translation model is configured."""

    def translate(self, text_en, context=()) -> Tuple[str, float]:
        return f"[zh]{text_en}[/zh]", 1.0


class PretrainedMT(MTBackend):
    """Placeholder for a frozen pretrained NMT / LLM translation backend.

    To implement:
    * load the checkpoint once and cache it (weights must be fixed),
    * inject up to mt_context_sentences previous sentences as context,
    * apply the terminology constraint (numbers / negation / proper nouns),
    * return a confidence score together with the translation.
    """

    def __init__(self, model: Optional[str] = None, **kwargs):
        raise RuntimeError(
            "PretrainedMT is not wired in this initial version. "
            "Configure a pretrained translation model and implement translate()."
        )
