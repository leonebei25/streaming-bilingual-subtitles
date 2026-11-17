import unittest

import numpy as np

from a_line.enhance import BypassEnhancer, SpectralSubtractionEnhancer
from a_line.enhance.source import EnhancedFrameSource
from b_line.asr import ScriptedASR
from b_line.config import BLineConfig
from b_line.metrics import evaluate
from b_line.mt import IdentityMT
from b_line.pipeline import SubtitlePipeline
from b_line.sources import SyntheticFrameSource
from b_line.types import AudioFrame, Gap

SCRIPT = [
    (0.5, 1.9, "Welcome to digital signal processing.", "欢迎来到数字信号处理。", 0.95),
    (2.9, 4.3, "Today we study streaming speech recognition.", "今天我们学习流式语音识别。", 0.95),
]


def drain(source, max_items=256):
    items = []
    while not source.ended():
        batch = source.take(max_items)
        if not batch and not source.ended():
            break
        items.extend(batch)
    return items


class TestEnhancedFrameSource(unittest.TestCase):
    def setUp(self):
        self.cfg = BLineConfig(stability_delay_ms=200.0, chunk_ms=320.0,
                               vad_hangover_ms=300.0, max_seg_chars=60)

    def make(self, enhancer, loss=None, duration=2.0):
        utterances = [(0.2, 0.8, 200.0), (1.2, 1.8, 400.0)]
        inner = SyntheticFrameSource(
            self.cfg, utterances, duration, loss_windows=loss, seed=3
        )
        return EnhancedFrameSource(inner, enhancer, self.cfg.sample_rate)

    def test_bypass_preserves_timestamps_sequences_and_samples(self):
        source = self.make(BypassEnhancer())
        items = drain(source)
        frames = [i for i in items if isinstance(i, AudioFrame)]
        self.assertEqual(len(frames), 200)
        self.assertEqual([f.seq for f in frames], list(range(200)))
        self.assertEqual(sum(len(f.data) for f in frames), 32000)
        self.assertAlmostEqual(frames[0].start_sec, 0.0, places=6)
        self.assertAlmostEqual(frames[-1].end_sec, 2.0, places=6)

    def test_gap_is_forwarded_once_and_not_filled(self):
        source = self.make(BypassEnhancer(), loss=[(0.9, 1.1)])
        items = drain(source)
        gaps = [i for i in items if isinstance(i, Gap)]
        frames = [i for i in items if isinstance(i, AudioFrame)]
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start_sec, 0.9, places=3)
        self.assertAlmostEqual(gaps[0].end_sec, 1.1, places=3)
        self.assertEqual(sum(len(f.data) for f in frames), 28800)

    def test_spectral_output_is_finite_and_same_length(self):
        source = self.make(SpectralSubtractionEnhancer())
        items = drain(source)
        frames = [i for i in items if isinstance(i, AudioFrame)]
        self.assertEqual(sum(len(f.data) for f in frames), 32000)
        self.assertTrue(all(np.all(np.isfinite(f.data)) for f in frames))

    def test_no_frames_beyond_released_horizon(self):
        inner = SyntheticFrameSource(
            self.cfg, [(0.2, 0.8, 200.0)], 1.0, realtime=True, speed=40.0, seed=3
        )
        source = EnhancedFrameSource(inner, BypassEnhancer(), self.cfg.sample_rate)
        while not source.ended():
            for item in source.take():
                self.assertLessEqual(item.end_sec, inner.released_sec + 1e-9)

    def test_pipeline_a_and_b_produce_the_same_transcript(self):
        entries = [(t0, t1, en, conf) for (t0, t1, en, zh, conf) in SCRIPT]
        refs = [(t0, t1, en, zh) for (t0, t1, en, zh, conf) in SCRIPT]
        results = {}
        for name, enhancer in (("A", BypassEnhancer()),
                               ("B", SpectralSubtractionEnhancer())):
            utterances = [(t0, t1, 190.0 + 40.0 * i)
                          for i, (t0, t1, _, _, _) in enumerate(SCRIPT)]
            inner = SyntheticFrameSource(self.cfg, utterances, 5.0, seed=7)
            source = EnhancedFrameSource(inner, enhancer, self.cfg.sample_rate)
            pipe = SubtitlePipeline(
                self.cfg, ScriptedASR(entries, self.cfg), IdentityMT()
            )
            events = pipe.run(source)
            metrics = evaluate(
                events, references=refs, duration_sec=pipe.duration_sec,
                processing_sec=pipe.processing_sec,
            )
            results[name] = ([e.text_en for e in events if not e.gap_flag], metrics)
        self.assertEqual(results["A"][0], results["B"][0])
        self.assertEqual(results["A"][1]["wer"], 0.0)
        self.assertEqual(results["B"][1]["wer"], 0.0)


if __name__ == "__main__":
    unittest.main()
