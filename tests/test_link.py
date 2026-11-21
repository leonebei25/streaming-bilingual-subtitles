import unittest

from a_line.link import LinkConfig, LinkSource
from b_line.config import BLineConfig
from b_line.sources import SyntheticFrameSource
from b_line.types import AudioFrame, Gap


def drain(source, max_items=256):
    items = []
    while not source.ended():
        batch = source.take(max_items)
        if not batch and not source.ended():
            break
        items.extend(batch)
    return items


def signature(items):
    return [
        (type(i).__name__, round(i.start_sec, 4), round(i.end_sec, 4),
         getattr(i, "seq", None))
        for i in items
    ]


class TestLinkSource(unittest.TestCase):
    def setUp(self):
        self.cfg = BLineConfig()
        self.utterances = [(0.2, 0.8, 200.0), (1.2, 1.8, 400.0)]

    def make(self, link, duration=2.0):
        inner = SyntheticFrameSource(
            self.cfg, self.utterances, duration, seed=3
        )
        return LinkSource(inner, link, self.cfg.sample_rate), inner

    def test_clean_link_passes_every_frame(self):
        source, _ = self.make(LinkConfig(seed=1))
        items = drain(source)
        frames = [i for i in items if isinstance(i, AudioFrame)]
        gaps = [i for i in items if isinstance(i, Gap)]
        self.assertEqual(len(frames), 200)
        self.assertEqual(gaps, [])
        self.assertEqual([f.seq for f in frames], list(range(200)))

    def test_loss_burst_becomes_one_gap(self):
        source, _ = self.make(LinkConfig(burst_windows=[(0.8, 1.0)], seed=1))
        items = drain(source)
        gaps = [i for i in items if isinstance(i, Gap)]
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0].start_sec, 0.8, places=3)
        self.assertAlmostEqual(gaps[0].end_sec, 1.0, places=3)
        frames = [i for i in items if isinstance(i, AudioFrame)]
        self.assertEqual(sum(len(f.data) for f in frames), 28800)

    def test_small_jitter_buffer_drops_late_packets(self):
        source, _ = self.make(
            LinkConfig(jitter_ms=100.0, jitter_buffer_ms=40.0, seed=1)
        )
        items = drain(source)
        gaps = [i for i in items if isinstance(i, Gap)]
        self.assertGreater(len(gaps), 0)
        frames = [i for i in items if isinstance(i, AudioFrame)]
        self.assertLess(sum(len(f.data) for f in frames), 32000)

    def test_same_seed_produces_same_trace(self):
        a, _ = self.make(LinkConfig(loss_rate=0.1, seed=42))
        b, _ = self.make(LinkConfig(loss_rate=0.1, seed=42))
        self.assertEqual(signature(drain(a)), signature(drain(b)))

    def test_realtime_link_never_reads_ahead(self):
        inner = SyntheticFrameSource(
            self.cfg, [(0.2, 0.8, 200.0)], 1.0,
            realtime=True, speed=40.0, seed=3,
        )
        source = LinkSource(inner, LinkConfig(loss_rate=0.05, seed=1),
                            self.cfg.sample_rate)
        while not source.ended():
            for item in source.take(64):
                self.assertLessEqual(item.end_sec, inner.released_sec + 1e-9)


if __name__ == "__main__":
    unittest.main()
