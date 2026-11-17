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


class OpusMT(MTBackend):
    """Frozen MarianMT / OPUS model (default: opus-mt-en-zh).

    This is the real translation backend for the desktop pipeline. It is
    deliberately a small pretrained NMT model rather than an LLM: the course
    only requires a pretrained translation model with frozen weights and
    decoding settings, and MarianMT fits on CPU and on the Orin Nano.

    `use_context` prepends the previous sentence to the source. It is off by
    default because MarianMT was not trained on document context and can
    degrade; turn it on only if your evaluation shows a gain, and record it.
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-zh",
        device: Optional[str] = None,
        max_length: int = 256,
        num_beams: int = 4,
        use_context: bool = False,
    ):
        try:
            import torch
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "OpusMT needs torch, transformers and sentencepiece. Install "
                "with `pip install torch transformers sentencepiece` and make "
                "sure the model directory is available offline."
            ) from exc

        self._torch = torch
        self._tokenizer = MarianTokenizer.from_pretrained(model_name)
        self._model = MarianMTModel.from_pretrained(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        self.max_length = int(max_length)
        self.num_beams = int(num_beams)
        self.use_context = bool(use_context)

    def translate(self, text_en: str, context: Sequence[str] = ()) -> Tuple[str, float]:
        text = (text_en or "").strip()
        if not text:
            return "", 1.0
        if self.use_context and context:
            previous = (context[-1] or "").strip()
            if previous:
                text = f"{previous} {text}"

        batch = self._tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch = {k: v.to(self._device) for k, v in batch.items()}
        with self._torch.no_grad():
            generated = self._model.generate(
                **batch,
                max_length=self.max_length,
                num_beams=self.num_beams,
                output_scores=True,
                return_dict_in_generate=True,
            )
        zh = self._tokenizer.batch_decode(
            generated.sequences, skip_special_tokens=True
        )[0].strip()

        confidence = 1.0
        scores = getattr(generated, "sequences_scores", None)
        if scores is not None and len(scores) > 0:
            try:
                confidence = float(
                    self._torch.exp(scores[0]).clamp(0.0, 1.0).item()
                )
            except Exception:
                confidence = 1.0
        return zh, confidence


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
