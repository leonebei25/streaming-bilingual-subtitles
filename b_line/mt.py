"""Translation backends.

Translation is kept as a separate module from recognition (a course
requirement) so the A/B/C comparison can hold ASR and MT fixed and change only
the audio enhancement.
"""
from __future__ import annotations

import math
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


class CT2MT(MTBackend):
    """Opus/Marian MT served by CTranslate2 (CPU int8 by default).

    The Marian weights are converted once to the CTranslate2 format with
    ``scripts/convert_mt_ct2.py``. That directory is portable, so the model is
    quantised on the desktop and copied to the Jetson unchanged. At inference
    time only ``ctranslate2`` and the tokenizer half of ``transformers`` are
    imported -- torch stays out of the hot path, which is the whole point of
    the Jetson deployment.

    ``beam_size`` 4 matches OpusMT's default. ``intra_threads=0`` lets
    CTranslate2 pick a thread count; on the Orin Nano set it to the number of
    physical cores (6) if the pipeline competes with ASR.
    """

    def __init__(
        self,
        model_dir: str = "models/opus-mt-en-zh-ct2",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 4,
        max_decoding_length: int = 256,
        use_context: bool = False,
        intra_threads: int = 0,
        inter_threads: int = 1,
        tokenizer_dir: Optional[str] = None,
    ):
        try:
            import ctranslate2
            from transformers import MarianTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "CT2MT needs ctranslate2 and transformers (tokenizer only). "
                "Install with `pip install ctranslate2 transformers sentencepiece`; "
                "torch is not required at inference time."
            ) from exc

        path = Path(model_dir)
        if not path.is_dir():
            raise RuntimeError(
                f"no CTranslate2 model directory at {model_dir!r}.\n"
                "Convert the Marian/Opus checkpoint once (on the desktop; the "
                "result is portable to the Jetson):\n"
                "  pip install ctranslate2 transformers sentencepiece torch\n"
                "  python scripts/convert_mt_ct2.py --model models/opus-mt-en-zh "
                "--out models/opus-mt-en-zh-ct2\n"
                "Then point the pipeline at it with --mt ct2 --mt-model <dir>."
            )

        try:
            self._tokenizer = MarianTokenizer.from_pretrained(
                str(tokenizer_dir or model_dir), local_files_only=True
            )
        except Exception as exc:  # pragma: no cover - depends on the model dir
            raise RuntimeError(
                f"the CTranslate2 model at {model_dir!r} has no usable tokenizer "
                f"({exc}). Re-run the conversion with the tokenizer files copied "
                "(--copy-files tokenizer_config.json,vocab.json,source.spm,"
                "target.spm), or pass --mt-tokenizer-dir pointing at the original "
                "Hugging Face directory."
            ) from exc

        self.model_dir = str(model_dir)
        self.device = device or "cpu"
        self.compute_type = compute_type or "int8"
        self.beam_size = max(1, int(beam_size))
        self.max_decoding_length = int(max_decoding_length)
        self.use_context = bool(use_context)
        self.intra_threads = max(0, int(intra_threads))
        self.inter_threads = max(1, int(inter_threads))
        self._translator = ctranslate2.Translator(
            self.model_dir,
            device=self.device,
            compute_type=self.compute_type,
            inter_threads=self.inter_threads,
            intra_threads=self.intra_threads,
        )

    def translate(self, text_en: str, context: Sequence[str] = ()) -> Tuple[str, float]:
        text = (text_en or "").strip()
        if not text:
            return "", 1.0
        if self.use_context and context:
            previous = (context[-1] or "").strip()
            if previous:
                text = f"{previous} {text}"

        token_ids = self._tokenizer.encode(text)
        source_tokens = self._tokenizer.convert_ids_to_tokens(token_ids)
        results = self._translator.translate_batch(
            [source_tokens],
            beam_size=self.beam_size,
            max_decoding_length=self.max_decoding_length,
        )
        if not results or not results[0].hypotheses:
            return "", 0.0

        target_tokens = results[0].hypotheses[0]
        zh = self._tokenizer.decode(
            self._tokenizer.convert_tokens_to_ids(target_tokens),
            skip_special_tokens=True,
        ).strip()

        confidence = 1.0
        scores = getattr(results[0], "scores", None)
        if scores:
            try:
                # scores are cumulative log probabilities of the hypotheses.
                confidence = math.exp(min(0.0, float(scores[0])))
            except (TypeError, ValueError, OverflowError):
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
