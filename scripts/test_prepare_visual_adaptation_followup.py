import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from prepare_visual_adaptation_sft_pilot import target_modules


class FollowupConfigTests(unittest.TestCase):
    def test_l_has_language_targets_only(self):
        modules = target_modules("l")
        self.assertEqual(len(modules), 28 * 7)
        self.assertFalse(any(module.startswith("visual.") for module in modules))

    def test_a_uses_same_lora_targets(self):
        self.assertEqual(target_modules("l"), target_modules("a"))


if __name__ == "__main__":
    unittest.main()
