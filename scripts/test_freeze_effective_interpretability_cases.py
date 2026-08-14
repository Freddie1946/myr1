import unittest

from freeze_effective_interpretability_cases import is_effective


class FreezeEffectiveCasesTest(unittest.TestCase):
    def test_strict_effective_rule(self):
        base = {"clean_reasoning_correct": True, "clean_reasoning_forced_agreement": True,
                "reference_target_margin_drop": 0.2, "reference_minus_random_extra_drop": 0.1}
        self.assertTrue(is_effective(base))
        for key in base:
            row = dict(base)
            row[key] = False if isinstance(base[key], bool) else 0.0
            self.assertFalse(is_effective(row), key)


if __name__ == "__main__":
    unittest.main()
