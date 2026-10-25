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
from .sources import FrameSource
from .types import AudioFrame, Gap, SubtitleEvent
from .vad import EnergyVAD


class SubtitlePipeline:
    def __init__(self, cfg: BLineConfig, asr: ASRBackend, mt: MTBackend,
                 vad=None, segmenter=None):
        self.cfg = cfg
        self.asr = asr
        self.mt = mt
        self.vad = vad or EnergyVAD(cfg)
        self.segmenter = segmenter or StablePrefixSegmenter(cfg)
        self.events: List[SubtitleEvent] = []
        self.context: List[str] = []
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
            self._flush_chunk(force=True)
            self.segmenter.flush(it.start_sec)
            for seg in self.segmenter.pop():
                self._emit(seg)
            self.events.append(SubtitleEvent(
                seg_id=-1, t_start=it.start_sec, t_end=it.end_sec,
                text_en="", text_zh="", conf_asr=0.0, conf_mt=0.0,
                revision=0, gap_flag=True, emitted_at=self._emit_at(),
            ))
            self.vad.reset()
            return

        frame: AudioFrame = it
        is_speech, endpoints = self.vad.update(frame)
        self.segmenter.note_speech(is_speech, frame.end_sec, self.vad.seg_start)
        self._chunk.append(frame)
        if len(self._chunk) >= self.cfg.chunk_frames:
            self._flush_chunk(force=False)
        for (ts, te) in endpoints:
            self.segmenter.on_endpoint(te)

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
        self.segmenter.update(update, end)
        for seg in self.segmenter.pop():
            self._emit(seg)

    def _finish(self) -> None:
        self._flush_chunk(force=True)
        update = self.asr.finalize()
        self.segmenter.update(update, self.duration_sec)
        self.segmenter.flush(self.duration_sec)
        for seg in self.segmenter.pop():
            self._emit(seg)

    def _emit(self, seg) -> None:
        zh, conf_mt = self.mt.translate(seg.text_en, tuple(self.context))
        self.events.append(SubtitleEvent(
            seg_id=seg.seg_id, t_start=seg.t_start, t_end=seg.t_end,
            text_en=seg.text_en, text_zh=zh,
            conf_asr=seg.conf_asr, conf_mt=conf_mt,
            revision=0, gap_flag=False, emitted_at=self._emit_at(),
        ))
        self.context.append(seg.text_en)
        if len(self.context) > self.cfg.mt_context_sentences:
            self.context.pop(0)

    def _emit_at(self):
        """Stream-clock timestamp (audio seconds) when an event is produced.

        Only meaningful in realtime mode: latency = emitted_at - t_start is then
        measured in audio seconds and is independent of the replay speed.
        """
        return self._stream_clock if self._realtime else None
