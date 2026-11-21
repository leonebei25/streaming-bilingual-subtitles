"""Segmentation driven by the recognizer's finalized text.

The recognizer streams a partial transcript and reports how much of it is
*finalized* -- text that will never be rewritten. The segmenter watches that
number grow: each time the finalized text lengthens, the new slice is one whole
sentence and is committed as a segment.

Sentence boundaries therefore come from the ASR, not from an energy VAD:

  * SherpaASR finalizes a sentence when its own endpoint fires (language-model
    based, ~1.2 s of trailing silence by default, with a ~20 s hard cap), so a
    mid-sentence breath does not chop a caption.
  * ScriptedASR finalizes an utterance `stability_delay_ms` after it ends.

The VAD only supplies `note_speech` so a segment's t_start can be stamped with
the frame-accurate speech onset. The draft (the text after the finalized prefix)
is exposed as `partial` and shown live; it becomes a real segment only when the
ASR finalizes it or the stream ends.
"""
from __future__ import annotations

from typing import List, Optional

from .config import BLineConfig
from .types import ASRUpdate, Segment


def _slice_words(words, text: str, start: int, end: int):
    """The words covering text[start:end], or None when they do not line up."""
    if not words:
        return None
    pos, out = 0, []
    for w, t in words:
        w0, w1 = pos, pos + len(w)
        pos = w1 + 1  # the space after this word
        if w1 <= start:
            continue
        if w0 >= end:
            break
        out.append((w, t))
    if not out:
        return None
    joined = " ".join(w for w, _ in out)
    if joined.casefold() != text[start:end].strip().casefold():
        return None
    return out


class StablePrefixSegmenter:
    def __init__(self, cfg: BLineConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.emitted = 0          # characters of finalized text already committed
        self.cur_start: Optional[float] = None   # speech onset for t_start
        self.partial = ""         # the current draft (text after finalized)
        self.last_conf = 1.0
        self.seg_id = 0
        self.pending: List[Segment] = []
        self._used_onset: Optional[float] = None  # onset already spent on a segment
        self._last_end = 0.0      # end of the last committed segment
        self._draft_since: Optional[float] = None  # when the current draft began

    def note_speech(self, is_speech: bool, now_sec: float,
                    onset_sec: Optional[float] = None) -> None:
        """Track the speech onset at frame granularity (VAD keeps t_start exact).

        The VAD's hangover keeps reporting is_speech after a sentence is
        committed, with the *same* onset, so a stale onset must not be reused
        for the next sentence: only an onset that has not already been spent
        starts a new t_start.
        """
        onset = onset_sec if onset_sec is not None else now_sec
        if is_speech and self.cur_start is None and onset != self._used_onset:
            self.cur_start = onset

    def update(self, update: ASRUpdate, now_sec: float,
               end_sec: Optional[float] = None) -> None:
        """Advance from one recognizer update.

        ``stable_chars`` may be pushed past what the recognizer itself
        finalized -- that is how a semantic cut commits a prefix while the rest
        is still a draft. ``end_sec`` is the audio time of that boundary when
        known, so latency is not measured from the end of the untrimmed tail.
        """
        self.last_conf = update.confidence

        # Never move backwards: after a semantic cut, the recognizer still
        # reports the committed text as part of its draft.
        stable = max(update.stable_chars, self.emitted)

        committed = False
        finalized = update.text[:stable]
        if len(finalized) > self.emitted:
            start = self.emitted
            new_text = finalized[start:].strip()
            words = _slice_words(getattr(update, "words", None), update.text,
                                 start, stable)
            self.emitted = len(finalized)
            if new_text:
                self._commit_sentences(new_text, words, now_sec, end_sec)
                committed = True

        # Everything after the finalized prefix is the live draft.
        self.partial = update.text[stable:].strip()
        if self.partial:
            if committed or self._draft_since is None:
                self._draft_since = now_sec
        else:
            self._draft_since = None

    def draft_age(self, now_sec: float) -> float:
        """How long the current draft has been growing, in seconds."""
        if self._draft_since is None:
            return 0.0
        return max(0.0, now_sec - self._draft_since)

    def flush(self, te: Optional[float] = None) -> None:
        """Commit whatever draft is left (end of stream / gap)."""
        if self.partial.strip():
            t = te if te is not None else 0.0
            self._commit(self._sentence_text(self.partial.strip()), t)
            self.partial = ""
            self._draft_since = None

    def pop(self) -> List[Segment]:
        out = self.pending
        self.pending = []
        return out

    def _commit_sentences(self, text: str, words, now_sec: float,
                          end_sec: Optional[float] = None) -> None:
        """Commit one finalized slice.

        By default the slice goes to the screen whole: the boundary that ended
        the draft (the ASR endpoint, a semantic cut, or a forced commit) is the
        only splitter, so one draft run on screen becomes exactly one caption.
        With ``split_on_pauses`` enabled the slice is instead cut at the largest
        inter-word pauses, so each caption still ends where the speaker
        breathed.

        Word timestamps are absolute audio seconds (SherpaASR shifts them by
        the stream origin), so a caption's t_end is the end of its last word,
        not the moment the commit happened. ``end_sec`` overrides that when the
        caller knows the boundary exactly (semantic and pause cuts).
        """
        if not getattr(self.cfg, "split_on_pauses", False):
            self._commit(self._sentence_text(text, words),
                         self._boundary_time(words, end_sec, now_sec))
            return

        groups = self._group_by_gaps(words)
        if len(groups) <= 1:
            self._commit(self._sentence_text(text, words),
                         self._boundary_time(words, end_sec, now_sec))
            return

        pieces = [(self._sentence_text(" ".join(w for w, _ in grp).strip(), grp), grp)
                  for grp in groups]
        pieces = [(p, grp) for p, grp in pieces if p]
        if len(pieces) <= 1:
            self._commit(self._sentence_text(text, words),
                         self._boundary_time(words, end_sec, now_sec))
            return

        # With real word times, every piece ends where its own last word ended.
        if all(grp and grp[-1][1] is not None for _, grp in pieces):
            for piece, grp in pieces:
                self._commit(piece, grp[-1][1])
            return

        # No timings: spread the slice over [start, end] by character count.
        span_end = self._boundary_time(words, end_sec, now_sec)
        start = self.cur_start if self.cur_start is not None else self._last_end
        start = max(start, self._last_end)
        span = max(span_end - start, 0.2 * len(pieces))
        total = sum(len(p) for p, _ in pieces) or 1
        acc = 0
        for piece, _grp in pieces:
            acc += len(piece)
            self._commit(piece, start + span * acc / total)

    @staticmethod
    def _boundary_time(words, end_hint: Optional[float], now_sec: float) -> float:
        """When the committed text actually ended, in audio seconds."""
        if end_hint is not None:
            return float(end_hint)
        if words:
            t = words[-1][1]
            if t is not None:
                return float(t)
        return float(now_sec)

    def _group_by_gaps(self, words):
        if not words or len(words) < 4:
            return []
        gap_s = max(0.05, self.cfg.sentence_gap_ms / 1000.0)
        max_chars = int(self.cfg.max_seg_chars)      # 0 = no length cap
        min_chars = max(1, int(getattr(self.cfg, "sentence_min_chars", 25)))
        groups, cur, cur_len = [], [], 0
        for i, (w, t) in enumerate(words):
            cur.append((w, t))
            cur_len += len(w) + 1
            has_next = i + 1 < len(words)
            gap = (words[i + 1][1] - t) if has_next else 0.0
            # A clock reset inside the utterance shows up as a wild difference;
            # never treat that as a pause.
            usable_gap = has_next and -0.1 <= gap <= 5.0
            if usable_gap and gap >= gap_s and cur_len >= min_chars:
                groups.append(cur)
                cur, cur_len = [], 0
            elif max_chars > 0 and cur_len >= max_chars:
                # Over the length limit: cut at the biggest pause inside the
                # group rather than at whatever word happens to hit the limit,
                # so the caption still ends where the speaker breathed.
                cut = self._best_cut(cur, min_chars)
                if 0 < cut < len(cur):
                    groups.append(cur[:cut])
                    cur = cur[cut:]
                    cur_len = sum(len(x) + 1 for x, _ in cur)
                else:
                    groups.append(cur)
                    cur, cur_len = [], 0
        if cur:
            groups.append(cur)
        return groups

    @staticmethod
    def _best_cut(group, min_chars: int) -> int:
        """Index just after the largest pause that leaves a long enough piece."""
        best_i, best_gap, chars = 0, -1.0, 0
        for i in range(len(group) - 1):
            chars += len(group[i][0]) + 1
            if chars < min_chars:
                continue
            gap = group[i + 1][1] - group[i][1]
            if gap > best_gap:
                best_gap, best_i = gap, i + 1
        return best_i

    def _sentence_text(self, text: str, words=None) -> str:
        """Finish one committed caption with a terminal mark.

        The streaming recognizer returns bare words: 'THE AMPLIFIER DRIVES THE
        SPEAKER'. A caption with no full stop is indistinguishable from a
        fragment, so every committed sentence gets one. Nothing else is
        invented: a comma guessed from a pause is wrong far too often, so the
        only thing allowed to place marks *inside* a sentence is the
        punctuation model (`--punct-model`).

        `words` is accepted for interface stability and deliberately unused.
        """
        t = " ".join((text or "").split())
        if not t:
            return t
        if not getattr(self.cfg, "rule_punctuation", False) or self.cfg.punct_model:
            return t
        if t[-1] in ".!?…,;:":
            return t
        return t + "."

    def _commit(self, text: str, end_sec: float) -> None:
        # With no fresh VAD onset (two sentences inside one speech run), a
        # segment starts where the previous one ended: contiguous spans beat a
        # zero-length segment stamped at its own commit time.
        if self.cur_start is not None:
            self._used_onset = self.cur_start
            s = self.cur_start
        else:
            s = self._last_end
        # A VAD onset can be older than the previous caption's end (the speaker
        # began before the endpoint fired). Captions must never overlap.
        s = max(s, self._last_end)
        e = max(float(end_sec), s + 0.05)   # never a zero-length caption
        self.pending.append(Segment(self.seg_id, s, e, text, self.last_conf))
        self.seg_id += 1
        self._last_end = e
        self.cur_start = None
