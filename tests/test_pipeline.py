import unittest

from b_line.config import BLineConfig
from b_line.feeder import FrameFeeder
from b_line.segmenter import StablePrefixSegmenter
from b_line.types import ASRUpdate
from b_line.audio import synthesize_test_audio


class TestSegmenter(unittest.TestCase):
    def test_forced_split_reconstructs_text(self):
        cfg = BLineConfig(max_seg_chars=10, stability_delay_ms=100.0)
        seg = StablePrefixSegmenter(cfg)
        text = "alpha beta gamma delta epsilon"
        seg.note_speech(True, 0.0, 0.0)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=0.9), 0.0)
        seg.on_endpoint(1.0)
        seg.note_speech(False, 1.3)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=0.9), 1.3)
        out = seg.pop()
        self.assertGreaterEqual(len(out), 1)
        self.assertEqual(" ".join(s.text_en for s in out), text)


class TestFeeder(unittest.TestCase):
    def test_never_read_future(self):
        cfg = BLineConfig()
        x = synthesize_test_audio([(0.5, 1.0, 200.0)], 2.0, sr=cfg.sample_rate, seed=0)
        f = FrameFeeder(x, cfg.sample_rate, cfg)
        f.release(1.0)
        items = f.drain()
        self.assertTrue(items)
        self.assertTrue(all(i.end_sec <= 1.0 + 1e-9 for i in items))
        self.assertFalse(f.exhausted())

    def test_loss_window_becomes_gap(self):
        from b_line.types import Gap
        cfg = BLineConfig()
        x = synthesize_test_audio([(0.2, 1.8, 200.0)], 2.0, sr=cfg.sample_rate, seed=0)
        f = FrameFeeder(x, cfg.sample_rate, cfg, loss_windows=[(0.5, 0.7)])
        f.release_all()
        items = f.drain()
        gaps = [i for i in items if isinstance(i, Gap)]
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start_sec, 0.5, places=2)
        self.assertAlmostEqual(gaps[0].end_sec, 0.7, places=2)


if __name__ == "__main__":
    unittest.main()
