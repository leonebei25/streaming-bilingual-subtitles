"""End-to-end orchestration: source -> VAD -> ASR -> segmenter -> MT -> events.

The entry point takes a FrameSource (a stream), never a decoded buffer. The
source decides what audio has actually arrived; the pipeline only processes what
it is handed.
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from .asr import ASRBackend
from .config import BLineConfig
from .mt import MTBackend
from .segmenter import StablePrefixSegmenter, _slice_words
from .semantic import SemanticCutter
from .sentence import split_sentences
from .sources import FrameSource
from .types import ASRUpdate, AudioFrame, Gap, SubtitleEvent
from .vad import EnergyVAD


class SubtitlePipeline:
    def __init__(self, cfg: BLineConfig, asr: ASRBackend, mt: MTBackend,
                 vad=None, segmenter=None, punct=None):
        self.cfg = cfg
        self.asr = asr
        self.mt = mt
        self.punct = punct
        self.semantic = None
        self.semantic_cut_count = 0
        self.pause_cut_count = 0
        self._last_update: Optional[ASRUpdate] = None
        if getattr(cfg, "semantic_cuts", False):
            if punct is None:
                print("[pipeline] semantic cuts need a punctuation model; disabled")
            else:
                self.semantic = SemanticCutter(punct, cfg)
        self.vad = vad or EnergyVAD(cfg)
        self.segmenter = segmenter or StablePrefixSegmenter(cfg)
        self.events: List[SubtitleEvent] = []
        self.context: List[str] = []
        self._event_seq = 0
        # Transcript tail that has been recognized but not yet committed as a
        # segment. The display shows it as a draft line so words appear while
        # they are spoken instead of after the sentence ends.
        self.live_text = ""
        # True once the recognizer has supplied word timings, which is what
        # lets the segmenter cut at pauses instead of at a character count.
        self.words_available = False
        self._cautions: dict = {}
        self._chunk: List[AudioFrame] = []
        self._wall0 = 0.0
        self._proc0 = 0.0
        self._stream_clock = 0.0
        self._realtime = False
        self.duration_sec = 0.0

    @property
    def processing_sec(self) -> float:
        return time.perf_counter() - self._proc0

    def run(self, source: FrameSource) -> List[SubtitleEvent]:
        """Consume a frame stream until the source reports it has ended.

        The stream may be a live capture, a file replay or a synthetic signal;
        the pipeline does not know or care which.
        """
        self.events = []
        self.context = []
        self._event_seq = 0
        self.live_text = ""
        self.words_available = False
        self.semantic_cut_count = 0
        self.pause_cut_count = 0
        self._last_update = None
        self._cautions = {}
        self._chunk = []
        self.asr.reset()
        self.vad.reset()
        self.segmenter.reset()
        if self.semantic is not None:
            self.semantic.reset()

        self.duration_sec = source.duration_sec
        self._proc0 = time.perf_counter()
        self._realtime = source.realtime

        try:
            while True:
                items = source.take()
                self._stream_clock = source.released_sec
                if not items:
                    if source.ended():
                        break
                    if source.realtime:
                        time.sleep(0.004)
                        continue
                    break
                for it in items:
                    self._handle(it)
        except KeyboardInterrupt:
            # Live capture (mic / pipe): Ctrl+C flushes what was produced so far
            # and lets the caller write its report, instead of losing the run.
            pass

        # A live stream has no known length up front; its duration is however
        # much audio actually arrived. Refresh it so metrics divide by the real
        # span instead of the zero seen at start-up.
        self.duration_sec = source.duration_sec
        self._finish()
        return self.events

    def _handle(self, it) -> None:
        if isinstance(it, Gap):
            # Always flush the recognizer chunk: audio on the two sides of a
            # hole must never be concatenated. A short hole, however, is not
            # the end of the sentence -- keep the VAD and segmenter state so a
            # 20 ms lost packet does not shift every later subtitle.
            self._flush_chunk(force=True)
            gap_ms = max(0.0, it.end_sec - it.start_sec) * 1000.0
            if gap_ms >= self.cfg.gap_reset_ms:
                self.segmenter.flush(it.start_sec)
                for seg in self.segmenter.pop():
                    self._emit(seg)
                self.vad.reset()
                self._last_update = None
                if self.semantic is not None:
                    self.semantic.reset()
            self.events.append(SubtitleEvent(
                seg_id=-1, t_start=it.start_sec, t_end=it.end_sec,
                text_en="", text_zh="", conf_asr=0.0, conf_mt=0.0,
                revision=0, gap_flag=True, emitted_at=self._emit_at(),
            ))
            return

        frame: AudioFrame = it
        is_speech, _ = self.vad.update(frame)
        self.segmenter.note_speech(is_speech, frame.end_sec, self.vad.seg_start)
        self._chunk.append(frame)
        if len(self._chunk) >= self.cfg.chunk_frames:
            self._flush_chunk(force=False)

    def _flush_chunk(self, force: bool) -> None:
        if not self._chunk:
            return
        if not force and len(self._chunk) < self.cfg.chunk_frames:
            return
        data = np.concatenate([f.data for f in self._chunk])
        start = self._chunk[0].start_sec
        end = self._chunk[-1].end_sec
        self._chunk = []
        update = self.asr.stream(data, start, end)
        self._last_update = update
        self.words_available = getattr(update, "words", None) is not None
        self._feed_segmenter(update, end)
        self.live_text = self.segmenter.partial
        for seg in self.segmenter.pop():
            self._emit(seg)
        self._maybe_force_finalize(end)

    def _feed_segmenter(self, update: ASRUpdate, end: float) -> None:
        """Hand one recognizer update to the segmenter, with semantic cuts.

        A confirmed sentence end inside the draft becomes a shorter
        ``stable_chars`` for this update only: the prefix is committed and the
        tail remains the draft. The recognizer is never reset, so no text or
        audio has to be re-decoded.
        """
        stable = max(update.stable_chars, self.segmenter.emitted)
        if self.semantic is not None and len(update.text) > stable:
            draft = update.text[stable:]
            cut = self.semantic.feed(draft, end)
            if cut > 0:
                cut = min(cut, len(draft))
                self.semantic_cut_count += 1
                start = self.segmenter.cur_start
                if start is None:
                    start = self.segmenter._last_end
                cut_sec = self._cut_time(update, stable, cut, end, start)
                update = ASRUpdate(
                    text=update.text,
                    stable_chars=stable + cut,
                    confidence=update.confidence,
                    finalized=update.finalized,
                    words=getattr(update, "words", None),
                )
                self.segmenter.update(update, end, end_sec=cut_sec)
                return
        self.segmenter.update(update, end)

    @staticmethod
    def _cut_time(update: ASRUpdate, stable: int, cut_rel: int,
                  end_sec: float, start_sec: float) -> float:
        """Audio time of a character-level cut inside the draft.

        Prefer the end time of the last whole word before the cut; fall back to
        a character-proportional estimate when the backend gives no timings.
        """
        words = _slice_words(getattr(update, "words", None), update.text,
                             stable, len(update.text))
        if words:
            draft = update.text[stable:]
            lead = len(draft) - len(draft.lstrip())
            acc, best = 0, None
            for i, (w, t) in enumerate(words):
                acc += len(w) + (1 if i else 0)
                if lead + acc <= cut_rel:
                    best = float(t)
            if best is not None and start_sec - 0.1 <= best <= end_sec + 0.5:
                return best
        total = max(1, len(update.text) - stable)
        return min(end_sec, start_sec + max(0.0, end_sec - start_sec)
                   * (cut_rel / total))

    def _maybe_force_finalize(self, now_sec: float) -> None:
        """Bound how long one caption may stay a draft.

        Sherpa endpoints on its own after ~1.2 s of silence, but continuous
        speech, music or a noisy room can keep a single partial alive far too
        long -- and then nothing is ever committed and the screen stays empty.
        When that happens, ask the recognizer to end the utterance now. The
        bounds are the draft's age (`max_seg_seconds`) and, optionally, a
        character cap (`max_seg_chars`, 0 disables it).
        """
        seg = self.segmenter
        if not seg.partial:
            return
        max_chars = int(self.cfg.max_seg_chars)   # 0 = no character cap
        max_sec = float(self.cfg.max_seg_seconds)
        due = ((max_chars > 0 and len(seg.partial) > max_chars)
               or (max_sec > 0 and seg.draft_age(now_sec) >= max_sec))
        if not due:
            return
        # First choice: commit up to the latest real pause. A hard cut at the
        # current end lands mid-phrase ("... one big NVIDIA | to see if ...").
        if self._pause_cut(now_sec):
            return
        force = getattr(self.asr, "force_finalize", None)
        hit = False
        if force is not None:
            try:
                update = force()
            except Exception as exc:
                if not self._cautions.get("force"):
                    print(f"[pipeline] force_finalize failed, captions may stall: {exc}")
                    self._cautions["force"] = True
                update = None
            if isinstance(update, ASRUpdate):
                hit = True
                seg.update(update, now_sec)
                self.live_text = seg.partial
                for s in seg.pop():
                    self._emit(s)
                if self.semantic is not None:
                    self.semantic.reset()
        if not hit and not self._cautions.get("no_force"):
            print("[pipeline] the ASR backend cannot end a long draft; if captions "
                  "stall, that is why")
            self._cautions["no_force"] = True

    def _pause_cut(self, now_sec: float) -> bool:
        """Commit the draft up to the latest real pause before a hard cut.

        Called only when a draft is due for a forced commit. Word timestamps
        are absolute audio seconds, so the cut can land exactly on the pause
        and the caption's t_end is the end of its last word. Returns False when
        there is no usable pause (or no word timings), letting the caller fall
        back to ``force_finalize``.
        """
        seg = self.segmenter
        update = self._last_update
        if update is None or not seg.partial:
            return False

        stable = max(update.stable_chars, seg.emitted)
        draft = update.text[stable:]
        if len(draft.strip()) < max(1, int(self.cfg.force_min_chars)):
            return False
        words = _slice_words(getattr(update, "words", None), update.text,
                             stable, len(update.text))
        if not words or len(words) < 2:
            return False

        # The words were matched against this slice by _slice_words, and ASR
        # text separates words with single spaces.
        lead = len(draft) - len(draft.lstrip())
        offsets, acc = [], 0
        for i, (w, _t) in enumerate(words):
            acc += len(w) + (1 if i else 0)
            offsets.append(lead + acc)

        threshold = max(0.05, float(self.cfg.force_pause_ms) / 1000.0)
        best_i = None
        for i in range(len(words) - 1):
            gap = float(words[i + 1][1]) - float(words[i][1])
            # 0.4 s computed as 2.4 - 2.0 is 0.3999999999999999; do not let
            # floating point reject a pause that is exactly at the threshold.
            if gap + 1e-6 < threshold or gap > 5.0:
                continue
            if offsets[i] < self.cfg.force_min_chars:
                continue
            best_i = i          # keep the latest qualifying pause
        if best_i is None:
            return False

        cut_rel = offsets[best_i]
        cut_sec = float(words[best_i][1])
        # A backend that still reports stream-relative times would produce an
        # impossible boundary; let the hard-cut path handle that case.
        if not (seg._last_end - 0.05 <= cut_sec <= now_sec + 0.5):
            return False

        cut_update = ASRUpdate(
            text=update.text,
            stable_chars=stable + cut_rel,
            confidence=update.confidence,
            finalized=update.finalized,
            words=getattr(update, "words", None),
        )
        seg.update(cut_update, now_sec, end_sec=cut_sec)
        self.live_text = seg.partial
        for s in seg.pop():
            self._emit(s)
        if self.semantic is not None:
            self.semantic.reset()
        self.pause_cut_count += 1
        return True

    def _finish(self) -> None:
        self._flush_chunk(force=True)
        update = self.asr.finalize()
        self.segmenter.update(update, self.duration_sec)
        self.segmenter.flush(self.duration_sec)
        for seg in self.segmenter.pop():
            self._emit(seg)
        self.live_text = ""

    def _emit(self, seg) -> None:
        """Turn one finalized ASR slice into subtitle events.

        By default the slice is one event: whatever boundary ended the draft
        (the ASR endpoint or a forced commit) is the only splitter, so the line
        the viewer just watched grow on screen becomes exactly one caption.
        With ``split_on_pauses`` enabled the slice is split into sentence-sized
        parts instead, and its time span is spread over them by length.
        """
        text = self.punct.restore(seg.text_en) if self.punct is not None else seg.text_en
        if getattr(self.cfg, "split_on_pauses", False):
            parts = split_sentences(text, max_chars=self.cfg.max_seg_chars)
        else:
            parts = [text] if text.strip() else []
        if not parts:
            return
        for part, (t0, t1) in zip(parts, self._spread(seg.t_start, seg.t_end, parts)):
            zh, conf_mt = self.mt.translate(part, tuple(self.context))
            self.events.append(SubtitleEvent(
                seg_id=self._event_seq, t_start=t0, t_end=t1,
                text_en=part, text_zh=zh,
                conf_asr=seg.conf_asr, conf_mt=conf_mt,
                revision=0, gap_flag=False, emitted_at=self._emit_at(),
            ))
            self._event_seq += 1
            self.context.append(part)
            if len(self.context) > self.cfg.mt_context_sentences:
                self.context.pop(0)

    @staticmethod
    def _spread(t0: float, t1: float, parts):
        """Give each part a sub-interval of [t0, t1] proportional to its length."""
        total = sum(len(p) for p in parts) or 1
        span = max(0.0, t1 - t0)
        out, acc, cur = [], 0, t0
        for p in parts:
            acc += len(p)
            end = t0 + span * acc / total
            out.append((cur, max(end, cur)))
            cur = end
        return out

    def _emit_at(self):
        """Stream-clock timestamp (audio seconds) when an event is produced.

        Only meaningful in realtime mode: latency = emitted_at - t_start is then
        measured in audio seconds and is independent of the replay speed.
        """
        return self._stream_clock if self._realtime else None
