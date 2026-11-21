"""Punctuation restoration through sherpa-onnx.

Two model families are supported, and ``PunctRestorer`` picks one from the
path it is given:

* **online** (English, recommended for live use): Edge-Punct-Casing CNN-BiLSTM,
  ~30 MB. A directory holding ``model.onnx`` plus ``bpe.vocab``. Uses
  ``sherpa_onnx.OnlinePunctuation``. It also restores casing, which is
  harmless because the display normalises case anyway.
    https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/
    sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
* **offline** (Chinese + English, ~279 MB): CT-Transformer, via
  ``sherpa_onnx.OfflinePunctuation``.
    https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/
    sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

Detection: a directory containing ``bpe.vocab`` is an online model, anything
else is offline. A path to a single ``.onnx`` file is always offline.

The streaming zipformer emits bare upper-case words, so this module is what
turns "SO NOW WE SAMPLE THE SIGNAL" into "So now, we sample the signal."
Nothing else in the pipeline may invent punctuation from pauses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_ONLINE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/"
    "sherpa-onnx-online-punct-en-2024-08-06.tar.bz2"
)
_OFFLINE_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/"
    "sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2"
)


def _repair_online_spacing(src: str, out: str) -> str:
    """Re-insert spaces the online decoder occasionally drops.

    Observed with sherpa-onnx 2.0.1: on a long input one space disappears
    ("we filter" -> "wefilter"). The word order is unchanged, so when the word
    counts differ, split each merged token using the source words. Equal counts
    return the model output untouched.
    """
    src_words = src.split()
    out_words = out.split()
    if len(src_words) == len(out_words):
        return out

    fixed = []
    pending = ""
    oi = 0
    for word in src_words:
        if pending:
            token = pending
            pending = ""
        elif oi < len(out_words):
            token = out_words[oi]
            oi += 1
        else:
            return out

        if (len(token) > len(word)
                and token.casefold().startswith(word.casefold())):
            # Keep the model's casing for the visible word, then split off the
            # next words that were glued onto it. Punctuation stays attached.
            shown = token[:len(word)]
            rest = token[len(word):]
            marks = ""
            while rest and rest[0] in ",.?!;:":
                marks += rest[0]
                rest = rest[1:]
            fixed.append(shown + marks)
            if rest:
                pending = rest
        else:
            fixed.append(token)

    if pending:
        fixed.append(pending)
    if len(fixed) != len(src_words):
        return out
    return " ".join(fixed)


class PunctRestorer:
    def __init__(self, model_path: str, num_threads: int = 1,
                 provider: str = "cpu", mode: str = "auto"):
        try:
            import sherpa_onnx
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                "punctuation restoration needs sherpa-onnx: `pip install sherpa-onnx`."
            ) from exc

        mode = (mode or "auto").lower()
        if mode not in ("auto", "online", "offline"):
            raise ValueError(
                f"punct mode must be auto, online or offline; got {mode!r}")

        path = Path(model_path)
        if not path.exists():
            raise RuntimeError(
                f"punctuation model not found: {path}\n"
                "Download and extract one of:\n"
                f"  online  (English, ~30 MB): {_ONLINE_URL}\n"
                f"  offline (zh+en, ~279 MB):  {_OFFLINE_URL}\n"
                "then pass the extracted directory (or its model.onnx) as "
                "--punct-model"
            )

        self._sherpa = sherpa_onnx
        self.kind = "offline"
        bpe_vocab: Optional[Path] = None

        if path.is_dir():
            bpe = path / "bpe.vocab"
            if mode == "online" or (mode == "auto" and bpe.exists()):
                self.kind = "online"
                bpe_vocab = bpe
            onnx = self._pick_onnx(path)
        else:
            if mode == "online":
                raise RuntimeError(
                    "an online punctuation model needs a directory containing "
                    "both model.onnx and bpe.vocab"
                )
            onnx = path

        if self.kind == "online":
            if bpe_vocab is None or not bpe_vocab.exists():
                raise RuntimeError(
                    f"{path}: online punctuation needs bpe.vocab next to the "
                    "model; if this is the CT-Transformer, pass its .onnx file "
                    "or use a directory without bpe.vocab"
                )
            config = sherpa_onnx.OnlinePunctuationConfig(
                sherpa_onnx.OnlinePunctuationModelConfig(
                    cnn_bilstm=str(onnx),
                    bpe_vocab=str(bpe_vocab),
                    num_threads=int(num_threads),
                    provider=provider,
                )
            )
            self._punct = sherpa_onnx.OnlinePunctuation(config)
            # The C++ API exposes only AddPunctuationWithCase; accept either
            # name so a future Python binding with add_punctuation also works.
            self._apply = (getattr(self._punct, "add_punctuation", None)
                           or self._punct.add_punctuation_with_case)
        else:
            config = sherpa_onnx.OfflinePunctuationConfig(
                sherpa_onnx.OfflinePunctuationModelConfig(
                    ct_transformer=str(onnx),
                    num_threads=int(num_threads),
                    provider=provider,
                )
            )
            self._punct = sherpa_onnx.OfflinePunctuation(config)
            self._apply = self._punct.add_punctuation

        self.model_path = str(onnx)
        self.bpe_vocab = str(bpe_vocab) if bpe_vocab else ""

    @staticmethod
    def _pick_onnx(root: Path) -> Path:
        """Prefer the full-precision model over the int8 one, as elsewhere."""
        matches = sorted(root.glob("*.onnx"))
        if not matches:
            raise RuntimeError(f"{root}: no .onnx file inside")
        for match in matches:
            if "int8" not in match.name:
                return match
        return matches[0]

    def describe(self) -> str:
        """Short identity for the config snapshot."""
        return f"{self.kind}:{Path(self.model_path).name}"

    def apply(self, text: Optional[str]) -> str:
        """Punctuate one string, letting inference errors propagate.

        The online model's BPE vocabulary is lower-case only (checked: zero
        upper-case tokens), so ASR output such as "SO NOW WE" must be
        lower-cased before it is tokenised. The model restores casing itself.
        """
        t = (text or "").strip()
        if not t:
            return ""
        if self.kind == "online":
            t = t.lower()
        out = self._apply(t).strip()
        if self.kind == "online":
            out = _repair_online_spacing(t, out)
        return out

    def restore(self, text: Optional[str]) -> str:
        t = (text or "").strip()
        if not t:
            return ""
        try:
            return self.apply(t)
        except Exception:
            # Restoration is an enhancement: never lose the transcript over it.
            return t
