import math
import unittest


class ReadinessStatisticsTests(unittest.TestCase):
    def test_binary_population_std(self):
        self.assertAlmostEqual(math.sqrt((3 / 8) * (5 / 8)), 0.4841229182759271)


if __name__ == "__main__":
    unittest.main()
