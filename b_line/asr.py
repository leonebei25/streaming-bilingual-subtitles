"""Recognizer backends.

All backends implement the same streaming contract: feed chunks of audio, get an
incremental ASRUpdate. ScriptedASR is a deterministic stand-in used to develop
and test the pipeline without a GPU; Qwen3ASR documents the intended real
integration and fails loudly until its environment is configured.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
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

    def force_finalize(self) -> Optional[ASRUpdate]:
        """Fold the current partial into the finalized text.

        The pipeline calls this when one utterance has stayed a draft far
        longer than a caption should, and the recognizer has not ended it on
        its own (continuous speech, background noise, a model without
        endpointing). Returning None means this backend cannot do it and the
        draft is left alone.
        """
        return None


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

    def force_finalize(self) -> ASRUpdate:
        # End whichever utterance is being revealed, so the normal reveal path
        # finalizes it instead of inventing a second source of truth.
        for (t0, t1, _text, _c) in self.script:
            if t0 <= self.covered_sec < t1:
                self.covered_sec = t1
                break
        text, _stable, conf = self._reveal()
        return ASRUpdate(text=text, stable_chars=len(text), confidence=conf,
                         finalized=True)


def _join_text(left: str, right: str) -> str:
    if left and right:
        return f"{left} {right}".strip()
    return (left or right).strip()


def _tokens_to_words(tokens, timestamps):
    """Group sub-word tokens into words, each with its end time.

    sherpa's zipformer marks a word start with '▁'. Returns None when the token
    stream does not follow that convention, so callers fall back to text only.
    """
    if not tokens or not timestamps or len(tokens) != len(timestamps):
        return None
    words = []
    cur: list = []
    cur_t = 0.0
    for tok, t in zip(tokens, timestamps):
        if not isinstance(tok, str):
            return None
        starts_word = tok.startswith("▁") or tok.startswith(" ")
        piece = tok.lstrip("▁ ").strip()
        if starts_word and cur:
            words.append(("".join(cur), cur_t))
            cur = []
        if piece:
            cur.append(piece)
        cur_t = float(t)
    if cur:
        words.append(("".join(cur), cur_t))
    return words or None


class SherpaASR(ASRBackend):
    """Streaming transducer ASR through sherpa-onnx.

    Install ``sherpa-onnx`` and point ``model_dir`` at an extracted streaming
    zipformer package (``tokens.txt`` plus encoder/decoder/joiner ONNX files).
    The recognizer emits partial text for the current utterance; this adapter
    keeps the cumulative contract expected by ``StablePrefixSegmenter``:
    ``ASRUpdate.text`` is the whole session transcript and ``stable_chars``
    covers finalized utterances plus the longest common prefix of the current
    partial.

    The stream is reset at sherpa's own endpoint. The pipeline's VAD still
    decides when a segment ends; the two are allowed to fire independently.
    """

    def __init__(
        self,
        model_dir: Optional[str] = None,
        tokens: Optional[str] = None,
        encoder: Optional[str] = None,
        decoder: Optional[str] = None,
        joiner: Optional[str] = None,
        cfg: Optional[BLineConfig] = None,
        num_threads: int = 2,
        decoding_method: str = "greedy_search",
        provider: str = "cpu",
        **backend_kwargs,
    ):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "SherpaASR needs the sherpa-onnx package. Install it with "
                "`pip install sherpa-onnx` and download a streaming zipformer "
                "model package (see https://github.com/k2-fsa/sherpa-onnx)."
            ) from exc

        self.cfg = cfg
        self.sample_rate = int(getattr(cfg, "sample_rate", 16000))
        paths = self._resolve_paths(model_dir, tokens, encoder, decoder, joiner)
        self._rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=paths["tokens"],
            encoder=paths["encoder"],
            decoder=paths["decoder"],
            joiner=paths["joiner"],
            num_threads=int(num_threads),
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method=decoding_method,
            provider=provider,
            **backend_kwargs,
        )
        self.reset()

    @staticmethod
    def _resolve_paths(model_dir, tokens, encoder, decoder, joiner) -> dict:
        if model_dir:
            root = Path(model_dir)

            def pick(pattern: str, explicit: Optional[str]) -> str:
                if explicit:
                    return str(explicit)
                matches = sorted(root.glob(pattern))
                if not matches:
                    raise FileNotFoundError(
                        f"{root}: no file matching {pattern!r}; pass explicit paths "
                        "or download a sherpa-onnx streaming model package"
                    )
                for match in matches:
                    if "int8" not in match.name:
                        return str(match)
                return str(matches[0])

            return {
                "tokens": pick("tokens.txt", tokens),
                "encoder": pick("encoder*.onnx", encoder),
                "decoder": pick("decoder*.onnx", decoder),
                "joiner": pick("joiner*.onnx", joiner),
            }
        if not all((tokens, encoder, decoder, joiner)):
            raise ValueError(
                "provide model_dir or all four paths: tokens, encoder, decoder, joiner"
            )
        return {
            "tokens": str(tokens),
            "encoder": str(encoder),
            "decoder": str(decoder),
            "joiner": str(joiner),
        }

    def reset(self) -> None:
        self._stream = self._rec.create_stream()
        self._session_text = ""
        self._session_words: list = []

    def _decode(self) -> str:
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        result = self._rec.get_result(self._stream)
        # sherpa-onnx has returned both a plain string and a result object
        # across versions; accept either.
        if isinstance(result, str):
            return result.strip()
        return (getattr(result, "text", "") or "").strip()

    def _result(self):
        """Return (text, words) where words carries per-word end times.

        get_result_all() exposes tokens + timestamps; older wrappers only have
        get_result(), in which case words is None and the segmenter works on
        plain text.
        """
        while self._rec.is_ready(self._stream):
            self._rec.decode_stream(self._stream)
        get_all = getattr(self._rec, "get_result_all", None)
        if get_all is not None:
            try:
                r = get_all(self._stream)
                text = (getattr(r, "text", "") or "").strip()
                words = _tokens_to_words(
                    getattr(r, "tokens", None), getattr(r, "timestamps", None)
                )
                if text:
                    return text, words
            except Exception:
                pass
        return self._decode(), None

    def _update(self, finalized: bool) -> ASRUpdate:
        current, cur_words = self._result()
        # The finalized prefix is the whole sentences sherpa has ended itself.
        # The current partial is a separate draft: it is *not* folded into
        # stable via a longest-common-prefix, because a transducer rewrites its
        # tail and that made captions look chopped mid-sentence.
        if self._rec.is_endpoint(self._stream):
            if current:
                self._session_text = _join_text(self._session_text, current)
                self._session_words = list(self._session_words) + list(cur_words or [])
            self._rec.reset(self._stream)
            current, cur_words = "", None
        elif finalized and current:
            self._session_text = _join_text(self._session_text, current)
            self._session_words = list(self._session_words) + list(cur_words or [])
            current, cur_words = "", None

        text = _join_text(self._session_text, current)
        words = list(self._session_words) + list(cur_words or [])
        return ASRUpdate(
            text=text,
            stable_chars=len(self._session_text),
            confidence=1.0,
            finalized=finalized,
            words=words or None,
        )

    def stream(self, samples, start_sec: float, end_sec: float) -> ASRUpdate:
        self._stream.accept_waveform(
            self.sample_rate, np.asarray(samples, dtype=np.float32)
        )
        return self._update(finalized=False)

    def finalize(self) -> ASRUpdate:
        return self._update(finalized=True)

    def force_finalize(self) -> ASRUpdate:
        """End the current utterance at the current partial, even if sherpa has
        not endpointed. Keeps the bookkeeping inside one place: the text moves
        from draft to session_text, so the segmenter can never see it twice."""
        current, cur_words = self._result()
        if current:
            self._session_text = _join_text(self._session_text, current)
            self._session_words = list(self._session_words) + list(cur_words or [])
        self._rec.reset(self._stream)
        text = self._session_text
        return ASRUpdate(text=text, stable_chars=len(text), confidence=1.0,
                         finalized=True, words=self._session_words or None)


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
