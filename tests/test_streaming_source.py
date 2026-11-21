"""StreamingFrameSource: push-fed, bounded, real-time, never preloaded."""
import unittest

import numpy as np

from b_line.config import BLineConfig
from b_line.sources import StreamingFrameSource, _normalize_device
from b_line.types import AudioFrame, Gap


class NormalizeDeviceTest(unittest.TestCase):
    def test_numeric_string_is_an_index(self):
        self.assertEqual(_normalize_device("10"), 10)
        self.assertEqual(_normalize_device("  -3 "), -3)

    def test_name_passes_through(self):
        self.assertEqual(_normalize_device("麦克风阵列 1"), "麦克风阵列 1")
        self.assertEqual(_normalize_device("Realtek HD Audio"), "Realtek HD Audio")

    def test_empty_and_none_and_int(self):
        self.assertIsNone(_normalize_device(""))
        self.assertIsNone(_normalize_device(None))
        self.assertEqual(_normalize_device(5), 5)


class StreamingFrameSourceTest(unittest.TestCase):
    def setUp(self):
        self.cfg = BLineConfig()
        self.hop = self.cfg.hop_len
        self.sr = self.cfg.sample_rate

    def _frames(self, items):
        return [i for i in items if isinstance(i, AudioFrame)]

    def _gaps(self, items):
        return [i for i in items if isinstance(i, Gap)]

    def test_hops_are_contiguous_with_monotonic_seq(self):
        src = StreamingFrameSource(self.cfg)
        src.push(np.zeros(self.hop * 3, dtype=np.float32))
        frames = self._frames(src.take())
        self.assertEqual(len(frames), 3)
        self.assertAlmostEqual(frames[0].start_sec, 0.0)
        self.assertAlmostEqual(frames[2].end_sec, 3 * self.hop / self.sr)
        self.assertEqual([f.seq for f in frames], [0, 1, 2])
        self.assertAlmostEqual(src.released_sec, 3 * self.hop / self.sr)

    def test_partial_hop_held_until_finish(self):
        src = StreamingFrameSource(self.cfg)
        src.push(np.ones(self.hop // 2, dtype=np.float32))
        self.assertEqual(src.take(), [])
        self.assertFalse(src.ended())
        src.finish()
        items = src.take()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].data.size, self.hop // 2)
        self.assertTrue(src.ended())

    def test_int16_stereo_is_downmixed(self):
        src = StreamingFrameSource(self.cfg)
        inter = np.empty(self.hop * 2, dtype=np.int16)
        inter[0::2] = 1000
        inter[1::2] = 3000
        src.push(inter, channels=2)
        f = self._frames(src.take())[0]
        self.assertAlmostEqual(float(f.data.mean()), 2000 / 32768.0, places=4)

    def test_overflow_drops_oldest_and_reports_a_gap(self):
        src = StreamingFrameSource(self.cfg, max_buffer_sec=0.2)
        src.push(np.zeros(int(0.5 * self.sr), dtype=np.float32))
        self.assertGreater(src.dropped_samples, 0)
        items = src.take()
        self.assertTrue(self._gaps(items))
        self.assertLessEqual(src._buffered, src._max_buffered + self.hop)

    def test_rate_mismatch_raises(self):
        src = StreamingFrameSource(self.cfg)
        with self.assertRaises(ValueError):
            src.push(np.zeros(100, dtype=np.float32), sample_rate=8000)

    def test_limit_sec_finishes_automatically(self):
        src = StreamingFrameSource(self.cfg, limit_sec=0.05)
        src.push(np.zeros(int(self.sr * 0.1), dtype=np.float32))
        self.assertTrue(src._finished)

    def test_stream_reconstructs_input_exactly(self):
        src = StreamingFrameSource(self.cfg)
        x = np.sin(np.arange(self.hop * 5) * 0.01).astype(np.float32)
        step = self.hop // 2
        for i in range(0, len(x), step):       # uneven producer chunks
            src.push(x[i:i + step])
        src.finish()
        out = []
        for _ in range(100):
            out += [it.data for it in src.take() if isinstance(it, AudioFrame)]
            if src.ended():
                break
        y = np.concatenate(out)
        self.assertEqual(len(y), len(x))
        self.assertTrue(np.allclose(y, x))

    def test_empty_push_is_noop(self):
        src = StreamingFrameSource(self.cfg)
        src.push(np.zeros(0, dtype=np.float32))
        self.assertEqual(src.take(), [])
        self.assertAlmostEqual(src.released_sec, 0.0)
