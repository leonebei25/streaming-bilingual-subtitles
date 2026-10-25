import unittest

from b_line.metrics import chrf, edit_distance, wer


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


if __name__ == "__main__":
    unittest.main()
