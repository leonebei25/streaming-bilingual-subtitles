import unittest
from types import SimpleNamespace

from b_line.display import CaptionQueue, dwell_sec, reading_time


def ev(n, en="", zh=""):
    return SimpleNamespace(seg_id=n, text_en=en, text_zh=zh, revision=0)


class TestDwell(unittest.TestCase):
    def test_reading_time_grows_with_text(self):
        self.assertLess(reading_time("short"), reading_time("a much longer line"))
        self.assertGreater(reading_time("中文四个字"), 0.4)

    def test_dwell_is_clamped(self):
        self.assertAlmostEqual(dwell_sec("", min_hold_s=1.2), 1.2)
        self.assertAlmostEqual(dwell_sec("ok", min_hold_s=1.2), 1.2)
        long_text = "x" * 400
        self.assertAlmostEqual(dwell_sec(long_text, max_hold_s=4.5), 4.5)

    def test_cjk_reads_slower_than_latin_per_char(self):
        self.assertGreater(reading_time("字" * 10), reading_time("a" * 10))


class TestCaptionQueue(unittest.TestCase):
    def test_first_event_shows_immediately(self):
        q = CaptionQueue()
        q.push(ev(0, "hello"), 0.0)
        cur, prev = q.poll(0.0)
        self.assertEqual(cur.seg_id, 0)
        self.assertIsNone(prev)

    def test_caption_is_not_replaced_before_it_is_read(self):
        q = CaptionQueue(min_hold_s=1.2)
        q.push(ev(0, "first caption"), 0.0)
        cur, _ = q.poll(0.0)
        q.push(ev(1, "second caption"), 1.0)
        cur, _ = q.poll(1.1)
        self.assertEqual(cur.seg_id, 0, "the first caption must stay up")
        cur, prev = q.poll(1.25)
        self.assertEqual(cur.seg_id, 1)
        self.assertEqual(prev.seg_id, 0, "the replaced caption stays as context")

    def test_keeps_last_caption_when_nothing_follows(self):
        q = CaptionQueue(min_hold_s=0.5)
        q.push(ev(0, "only one"), 0.0)
        q.poll(0.0)
        cur, _ = q.poll(30.0)
        self.assertIsNotNone(cur)
        self.assertEqual(cur.seg_id, 0)

    def test_backlog_is_compressed_not_dropped(self):
        q = CaptionQueue(min_hold_s=1.2, backlog_floor_s=0.6)
        for i in range(4):
            q.push(ev(i, f"caption number {i} with some words"), 0.0)
        order = []
        switch_times = []
        last = None
        t = 0.0
        while t < 8.0:
            cur, _ = q.poll(t)
            if cur is not None and (last is None or cur.seg_id != last.seg_id):
                order.append(cur.seg_id)
                switch_times.append(t)
                last = cur
            t = round(t + 0.02, 2)
        self.assertEqual(order, [0, 1, 2, 3], "no caption may be skipped")
        gaps = [b - a for a, b in zip(switch_times, switch_times[1:])]
        for gap in gaps:
            self.assertGreaterEqual(gap, 0.6 - 0.03, f"caption flashed for {gap:.2f}s")
            self.assertLessEqual(gap, 4.5 + 0.03)

    def test_consecutive_fragments_join_into_one_caption(self):
        q = CaptionQueue(merge_gap_s=1.2, merge_max_chars=200)
        e1 = ev(0, "Joining us from all")
        e1.t_start, e1.t_end = 1.0, 2.0
        e2 = ev(1, "around the world")
        e2.t_start, e2.t_end = 2.1, 2.9
        q.push(e1, 0.0)
        q.push(e2, 0.1)
        cur, _ = q.poll(0.1)
        self.assertEqual(cur.text_en, "Joining us from all around the world")
        self.assertEqual(q.pending_count, 0, "the fragment must not queue separately")

    def test_join_stops_at_a_real_pause(self):
        q = CaptionQueue(merge_gap_s=0.5, merge_max_chars=200)
        e1 = ev(0, "first part")
        e1.t_start, e1.t_end = 0.0, 1.0
        e2 = ev(1, "second part")
        e2.t_start, e2.t_end = 2.5, 3.0
        q.push(e1, 0.0)
        q.push(e2, 0.1)
        cur, _ = q.poll(0.1)
        self.assertEqual(cur.text_en, "first part")
        self.assertEqual(q.pending_count, 1)

    def test_join_respects_the_length_cap(self):
        q = CaptionQueue(merge_gap_s=1.2, merge_max_chars=20)
        e1 = ev(0, "aaaaaaaaaa")
        e1.t_start, e1.t_end = 0.0, 1.0
        e2 = ev(1, "bbbbbbbbbbbbbb")
        e2.t_start, e2.t_end = 1.1, 2.0
        q.push(e1, 0.0)
        q.push(e2, 0.1)
        cur, _ = q.poll(0.1)
        self.assertEqual(cur.text_en, "aaaaaaaaaa")
        self.assertEqual(q.pending_count, 1)

    def test_remaining_counts_down(self):
        q = CaptionQueue(min_hold_s=1.0, max_hold_s=1.0)
        q.push(ev(0, "text"), 0.0)
        self.assertAlmostEqual(q.remaining(0.25), 0.75)
        self.assertEqual(q.remaining(5.0), 0.0)

    def test_display_lag_is_bounded(self):
        """Speech faster than reading speed must not build a growing backlog."""
        def run(spacing, seconds=25.0):
            q = CaptionQueue(min_hold_s=1.0, max_hold_s=4.0, max_lag_s=3.0)
            worst = 0.0
            pushed = 0
            for step in range(int(seconds / 0.05)):
                t = round(step * 0.05, 2)
                if pushed * spacing <= t:
                    e = ev(pushed, f"caption number {pushed}")
                    e.t_end = pushed * spacing
                    q.push(e, t)
                    pushed += 1
                cur, _ = q.poll(t)
                if cur is not None:
                    worst = max(worst, t - cur.t_end)
            return q, worst

        _, worst = run(0.5)
        self.assertLessEqual(worst, 3.6, f"display fell {worst:.1f}s behind")

        q, worst = run(0.15)
        self.assertGreater(q.dropped, 0, "a hopeless backlog must be skipped")
        self.assertLessEqual(worst, 3.6, f"display fell {worst:.1f}s behind")


if __name__ == "__main__":
    unittest.main()
