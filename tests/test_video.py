import os
import tempfile
import unittest

from a_line.video import make_test_video, render_subtitles


class TestVideoRenderer(unittest.TestCase):
    def test_renders_expected_frame_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.mp4")
            dst = os.path.join(tmp, "out.mp4")
            make_test_video(src, seconds=1.0, fps=5)
            events = [
                {"seg_id": 0, "t_start": 0.1, "t_end": 0.7,
                 "text_en": "Hello world.", "text_zh": "你好，世界。",
                 "gap_flag": False},
                {"seg_id": -1, "t_start": 0.7, "t_end": 0.9,
                 "text_en": "", "text_zh": "", "gap_flag": True},
            ]
            stats = render_subtitles(src, events, dst)
            self.assertEqual(stats["frames_in"], 5)
            self.assertEqual(stats["frames_out"], 5)
            self.assertGreater(os.path.getsize(dst), 0)

    def test_gap_is_not_drawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "in.mp4")
            dst = os.path.join(tmp, "out.mp4")
            make_test_video(src, seconds=0.6, fps=5)
            events = [{"seg_id": -1, "t_start": 0.0, "t_end": 0.6,
                       "text_en": "", "text_zh": "", "gap_flag": True}]
            stats = render_subtitles(src, events, dst)
            self.assertEqual(stats["frames_out"], 3)


if __name__ == "__main__":
    unittest.main()
