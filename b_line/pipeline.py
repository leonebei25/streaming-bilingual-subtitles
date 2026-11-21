"""End-to-end orchestration: source -> VAD -> ASR -> segmenter -> MT -> events.

The entry point takes a FrameSource (a stream), never a decoded buffer. The
source decides what audio has actually arrived; the pipeline only processes what
it is handed.
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from .asr import ASRBackend
from .config import BLineConfig
from .mt import MTBackend
from .segmenter import StablePrefixSegmenter
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
        self._cautions = {}
        self._chunk = []
        self.asr.reset()
        self.vad.reset()
        self.segmenter.reset()

        self.duration_sec = source.duration_sec
        self._proc0 = time.perf_counter()
        self._realtime = source.realtime

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
        self.words_available = getattr(update, "words", None) is not None
        self.segmenter.update(update, end)
        self.live_text = self.segmenter.partial
        for seg in self.segmenter.pop():
            self._emit(seg)
        self._maybe_force_finalize(end)

    def _maybe_force_finalize(self, now_sec: float) -> None:
        """Bound how long one caption may stay a draft.

        Sherpa endpoints on its own after ~1.2 s of silence, but continuous
        speech, music or a noisy room can keep a single partial alive far too
        long -- and then nothing is ever committed and the screen stays empty.
        When that happens, ask the recognizer to end the utterance now.
        """
        seg = self.segmenter
        if not seg.partial:
            return
        max_chars = max(1, int(self.cfg.max_seg_chars))
        max_sec = float(self.cfg.max_seg_seconds)
        due = (len(seg.partial) > max_chars
               or (max_sec > 0 and seg.draft_age(now_sec) >= max_sec))
        if not due:
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
        if not hit and not self._cautions.get("no_force"):
            print("[pipeline] the ASR backend cannot end a long draft; if captions "
                  "stall, that is why")
            self._cautions["no_force"] = True

    def _finish(self) -> None:
        self._flush_chunk(force=True)
        update = self.asr.finalize()
        self.segmenter.update(update, self.duration_sec)
        self.segmenter.flush(self.duration_sec)
        for seg in self.segmenter.pop():
            self._emit(seg)
        self.live_text = ""

    def _emit(self, seg) -> None:
        """Turn one finalized ASR slice into one event per sentence.

        The recognizer finalizes a stretch of speech that may hold several
        sentences (and has no punctuation of its own), so restore punctuation
        when a model is configured, split into sentences, and spread the
        slice's time span over them by length.
        """
        text = self.punct.restore(seg.text_en) if self.punct is not None else seg.text_en
        parts = split_sentences(text, max_chars=self.cfg.max_seg_chars)
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
