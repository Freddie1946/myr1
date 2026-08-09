#!/usr/bin/env python3

import unittest

from analyze_stage3_pathmmu_badcases import jaccard, mcnemar_exact


class Stage3BadcaseAnalysisTests(unittest.TestCase):
    def test_jaccard(self):
        self.assertEqual(jaccard("a b", "b c"), 1 / 3)

    def test_mcnemar_symmetry(self):
        self.assertAlmostEqual(mcnemar_exact(60, 55), mcnemar_exact(55, 60))
        self.assertEqual(mcnemar_exact(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
