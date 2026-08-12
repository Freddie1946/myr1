import unittest

from prepare_visual_adaptation_sft_pilot import target_modules


class FormalSelectedSftTests(unittest.TestCase):
    def test_l_and_a_language_lora_are_matched(self):
        self.assertEqual(target_modules("l"), target_modules("a"))
        self.assertEqual(len(target_modules("l")), 196)

    def test_a_mechanism_arm_keeps_language_targets_matched(self):
        self.assertEqual(target_modules("a"), target_modules("l"))
        self.assertFalse(any(name.startswith("visual.") for name in target_modules("a")))


if __name__ == "__main__":
    unittest.main()
