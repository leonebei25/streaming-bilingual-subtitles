import unittest

from b_line.metrics import chrf, edit_distance, evaluate, wer
from b_line.types import SubtitleEvent


def event(t0, t1, emitted):
    return SubtitleEvent(
        seg_id=0, t_start=t0, t_end=t1, text_en="a b", text_zh="啊",
        conf_asr=1.0, conf_mt=1.0, revision=0, gap_flag=False, emitted_at=emitted,
    )


class TestMetrics(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_wer_exact(self):
        self.assertEqual(wer("a b c", "a b c"), 0.0)

    def test_wer_substitution(self):
        self.assertAlmostEqual(wer("a b c", "a x c"), 1 / 3, places=6)

    def test_wer_empty_reference(self):
        self.assertIsNone(wer("", "x"))

    def test_chrf_identity(self):
        self.assertAlmostEqual(chrf("你好世界", "你好世界"), 1.0, places=6)

    def test_chrf_mismatch(self):
        self.assertLess(chrf("你好世界", "完全不同的内容"), 1.0)

    def test_latency_reports_both_views(self):
        events = [event(0.0, 1.0, 1.6), event(1.0, 2.0, 2.4)]
        out = evaluate(events, duration_sec=2.0, processing_sec=0.1, realtime=True)
        self.assertAlmostEqual(out["latency_p50_ms"], 1500.0, delta=80.0)
        self.assertAlmostEqual(out["latency_from_end_p50_ms"], 500.0, delta=50.0)


if __name__ == "__main__":
    unittest.main()
