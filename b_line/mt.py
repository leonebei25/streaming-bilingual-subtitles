"""Translation backends.

Translation is kept as a separate module from recognition (a course
requirement) so the A/B/C comparison can hold ASR and MT fixed and change only
the audio enhancement.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Tuple


class MTBackend(ABC):
    @abstractmethod
    def translate(self, text_en: str, context: Sequence[str] = ()) -> Tuple[str, float]:
        """Return (translation, confidence)."""


class IdentityMT(MTBackend):
    """Deterministic stand-in: performs no translation.

    It returns an empty string rather than tagging the source text. The tags
    used to end up on screen ([zh]...[/zh]) as soon as the mock was selected in
    the demo, which is never something a viewer should see. An empty string
    makes the display fall back to the English caption.
    """

    def translate(self, text_en, context=()) -> Tuple[str, float]:
        return "", 1.0


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
        local_files_only: Optional[bool] = None,
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

        # A local directory never needs the hub. A bare repo id does, unless
        # HF_HUB_OFFLINE is set -- otherwise every run waits on DNS before it
        # can say the model was already cached.
        if local_files_only is None:
            offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower()
            local_files_only = bool(offline and offline not in ("0", "false")) \
                or Path(model_name).is_dir()
        load_kwargs = {"local_files_only": True} if local_files_only else {}

        self._torch = torch
        try:
            self._tokenizer = MarianTokenizer.from_pretrained(model_name, **load_kwargs)
            self._model = MarianMTModel.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:
            source = ("the local directory" if Path(model_name).is_dir()
                      else "the Hugging Face cache")
            raise RuntimeError(
                f"cannot load the MT model {model_name!r} from {source}: {exc}\n"
                "Download it once with network access, then point at the folder:\n"
                "  pip install -U huggingface_hub\n"
                "  $env:HF_ENDPOINT = 'https://hf-mirror.com'   # if huggingface.co "
                "is unreachable\n"
                "  huggingface-cli download Helsinki-NLP/opus-mt-en-zh "
                "--local-dir models/opus-mt-en-zh\n"
                "  python scripts/live_demo.py ... --mt opus "
                "--mt-model models/opus-mt-en-zh\n"
                "To run without translation use --mt none (English only)."
            ) from exc
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
