#!/usr/bin/env python3

import unittest

from cluster_bootstrap_pathmmu import percentile


class ClusterBootstrapTests(unittest.TestCase):
    def test_percentile_endpoints(self):
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0), 1.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0], 1), 3.0)

    def test_percentile_interpolates(self):
        self.assertEqual(percentile([0.0, 10.0], 0.25), 2.5)


if __name__ == "__main__":
    unittest.main()
