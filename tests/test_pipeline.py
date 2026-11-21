import os
import tempfile
import unittest
import wave

import numpy as np

from b_line.config import BLineConfig
from b_line.mt import IdentityMT
from b_line.pipeline import SubtitlePipeline
from b_line.segmenter import StablePrefixSegmenter
from b_line.sources import SyntheticFrameSource, WavFrameSource
from b_line.types import ASRUpdate, AudioFrame, Gap


def write_wav(path, samples, sr=16000, channels=1):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


class TestSegmenter(unittest.TestCase):
    def test_finalized_text_commits_whole_sentence(self):
        cfg = BLineConfig()
        seg = StablePrefixSegmenter(cfg)
        text = "alpha beta gamma delta epsilon"
        seg.note_speech(True, 0.0, 0.0)
        seg.update(ASRUpdate(text=text, stable_chars=len(text), confidence=0.9), 0.0)
        out = seg.pop()
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text_en, text)


class _NoEndpointASR:
    """Finalizes nothing on its own: stable_chars stays 0, like sherpa on
    continuous speech that never hits its silence endpoint."""

    def __init__(self):
        self.text = ""
        self.forced = 0

    def reset(self):
        self.text = ""
        self.forced = 0

    def stream(self, samples, start_sec, end_sec):
        words = max(0, int(end_sec * 8))
        self.text = " ".join(["WORD"] * words)
        return ASRUpdate(text=self.text, stable_chars=0, confidence=1.0)

    def finalize(self):
        return ASRUpdate(text=self.text, stable_chars=len(self.text),
                         confidence=1.0, finalized=True)

    def force_finalize(self):
        self.forced += 1
        return ASRUpdate(text=self.text, stable_chars=len(self.text),
                         confidence=1.0, finalized=True)


class TestStallRecovery(unittest.TestCase):
    def test_long_draft_is_force_finalized(self):
        cfg = BLineConfig(max_seg_chars=40, max_seg_seconds=8.0)
        asr = _NoEndpointASR()
        asr.text = "WORD " * 10
        pipe = SubtitlePipeline(cfg, asr, IdentityMT())
        pipe.segmenter.partial = "WORD " * 10       # 50 chars > 40
        pipe.segmenter._draft_since = 1.0
        pipe._maybe_force_finalize(1.5)
        self.assertEqual(asr.forced, 1)
        self.assertGreaterEqual(len(pipe.events), 1,
                                "a stalled draft must still reach the screen")

    def test_backend_without_force_finalize_does_not_crash(self):
        cfg = BLineConfig(max_seg_chars=40)
        asr = _NoEndpointASR()
        asr.force_finalize = None
        pipe = SubtitlePipeline(cfg, asr, IdentityMT())
        pipe.segmenter.partial = "WORD " * 10
        pipe.segmenter._draft_since = 1.0
        pipe._maybe_force_finalize(1.5)   # must simply do nothing
        self.assertEqual(pipe.events, [])


class TestWavSource(unittest.TestCase):
    def setUp(self):
        self.cfg = BLineConfig()
        n = int(2.0 * self.cfg.sample_rate)
        t = np.arange(n) / self.cfg.sample_rate
        x = 0.3 * np.sin(2 * np.pi * 200.0 * t).astype(np.float32)
        fd, self.path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        write_wav(self.path, x, self.cfg.sample_rate)
        self.addCleanup(lambda: os.path.exists(self.path) and os.unlink(self.path))

    def test_construction_reads_nothing(self):
        src = WavFrameSource(self.path, self.cfg)
        self.assertEqual(src.samples_read, 0)
        self.assertFalse(src.ended())

    def test_rejects_non_16k(self):
        fd, p = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.addCleanup(lambda: os.path.exists(p) and os.unlink(p))
        write_wav(p, np.zeros(8000, dtype=np.float32), 8000)
        with self.assertRaises(ValueError):
            WavFrameSource(p, self.cfg)

    def test_loss_window_becomes_gap(self):
        src = WavFrameSource(self.path, self.cfg, loss_windows=[(0.5, 0.7)])
        items = []
        while not src.ended():
            items.extend(src.take())
        gaps = [i for i in items if isinstance(i, Gap)]
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start_sec, 0.5, places=3)
        self.assertAlmostEqual(gaps[0].end_sec, 0.7, places=3)

    def test_never_reads_past_horizon(self):
        src = WavFrameSource(self.path, self.cfg, realtime=True, speed=40.0)
        while not src.ended():
            for it in src.take():
                self.assertLessEqual(it.end_sec, src.released_sec + 1e-9)

    def test_one_gap_per_loss_window_under_pacing(self):
        # A moving horizon must not fragment one dropped interval into many gaps.
        src = WavFrameSource(self.path, self.cfg, loss_windows=[(0.5, 0.9)],
                             realtime=True, speed=40.0)
        gaps = []
        while not src.ended():
            gaps.extend(i for i in src.take() if isinstance(i, Gap))
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start_sec, 0.5, places=3)
        self.assertAlmostEqual(gaps[0].end_sec, 0.9, places=3)


class TestSyntheticSource(unittest.TestCase):
    def test_covers_whole_duration(self):
        cfg = BLineConfig()
        src = SyntheticFrameSource(cfg, [(0.2, 0.8, 200.0)], 1.0)
        frames = []
        while not src.ended():
            frames.extend(i for i in src.take() if isinstance(i, AudioFrame))
        self.assertEqual(sum(len(f.data) for f in frames), int(1.0 * cfg.sample_rate))

    def test_noise_is_position_seeded(self):
        cfg = BLineConfig()
        a = SyntheticFrameSource(cfg, [], 0.5, seed=1)
        b = SyntheticFrameSource(cfg, [], 0.5, seed=1)
        fa = [i for i in a.take() if isinstance(i, AudioFrame)]
        fb = [i for i in b.take() if isinstance(i, AudioFrame)]
        self.assertEqual(len(fa), len(fb))
        np.testing.assert_allclose(fa[3].data, fb[3].data)


if __name__ == "__main__":
    unittest.main()
