"""Semantic sentence cuts on the live draft.

The ASR keeps decoding one stream; nothing is reset. On a fixed cadence the
current draft (the text after the stable prefix) is punctuated. A
sentence-final mark that is *not* at the live edge -- at least
``semantic_lookahead_words`` words remain after it -- and that lands at the same
character offset on ``semantic_confirm`` consecutive runs becomes a cut. The
pipeline then reports that prefix as stable to the segmenter, which commits it;
the tail stays on screen and keeps growing as the draft.

The added cost is one run of a small punctuation model per cadence step
(~15-25 ms for the English CNN-BiLSTM) plus the confirmation wait. No audio is
re-decoded, and no recognised text is rewritten: the cut only moves the
boundary between "committed" and "draft".
"""
from __future__ import annotations

from typing import Optional

from .config import BLineConfig

_TERMINAL = ".!?"


class SemanticCutter:
    def __init__(self, punct, cfg: BLineConfig):
        self.punct = punct
        self.cfg = cfg
        self._warned = False
        self.reset()

    def reset(self) -> None:
        self._pending: Optional[int] = None
        self._hits = 0
        self._last_len = -1
        self._last_run: Optional[float] = None

    def feed(self, draft: str, now: float) -> int:
        """Return how many characters of ``draft`` may be committed (0 = none).

        ``draft`` is the raw tail of the ASR text; it may start with the space
        that followed the previous cut, and the returned offset counts it.
        """
        stripped = draft.strip()
        if len(stripped) < max(1, self.cfg.semantic_min_chars):
            self._pending, self._hits = None, 0
            return 0
        if not self._due(len(stripped), now):
            return 0
        self._last_len, self._last_run = len(stripped), now

        try:
            punctuated = self.punct.apply(stripped)
        except Exception as exc:
            # A punctuation failure may not take the pipeline down; semantic
            # cuts are an enhancement, not the only way to end a caption.
            if not self._warned:
                print(f"[semantic] punctuation failed, cuts disabled for now: {exc}")
                self._warned = True
            self._pending, self._hits = None, 0
            return 0
        raw_words = stripped.split()
        out_words = punctuated.split()
        # The model may only add marks and change case; a different word count
        # means the spacing repair failed, and no offset can be trusted.
        if not raw_words or len(raw_words) != len(out_words):
            self._pending, self._hits = None, 0
            return 0

        lookahead = max(0, int(self.cfg.semantic_lookahead_words))
        n = len(raw_words)
        ends = [i for i, w in enumerate(out_words)
                if w and w[-1] in _TERMINAL and n - 1 - i >= lookahead]
        if not ends:
            self._pending, self._hits = None, 0
            return 0

        # Rightmost sentence end that still leaves the required lookahead.
        cut = self._offset_after_word(stripped, raw_words, ends[-1])
        if cut is None or cut < self.cfg.semantic_min_chars:
            self._pending, self._hits = None, 0
            return 0

        if cut == self._pending:
            self._hits += 1
        else:
            self._pending, self._hits = cut, 1
        if self._hits < max(1, int(self.cfg.semantic_confirm)):
            return 0

        self._pending, self._hits = None, 0
        lead = len(draft) - len(draft.lstrip())
        return lead + cut

    def _due(self, length: int, now: float) -> bool:
        if self._last_run is None:
            return True
        if length - self._last_len >= max(1, int(self.cfg.semantic_step_chars)):
            return True
        return now - self._last_run >= max(0.0, float(self.cfg.semantic_step_sec))

    @staticmethod
    def _offset_after_word(text: str, words, index: int) -> Optional[int]:
        """Character offset just past word ``index`` in ``text``."""
        pos = 0
        for k, word in enumerate(words):
            found = text.find(word, pos)
            if found < 0:
                return None
            pos = found + len(word)
            if k == index:
                return pos
        return None
