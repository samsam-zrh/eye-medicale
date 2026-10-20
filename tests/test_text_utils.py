import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_utils import clean


class CleanTests(unittest.TestCase):
    def test_short_negative_sentence_is_kept(self):
        self.assertEqual(clean("No mass. No effusion."), "No mass. No effusion.")

    def test_opposite_findings_are_not_merged(self):
        text = "Effusion present. No effusion."
        self.assertEqual(clean(text), text)

    def test_repetition_ignores_case_and_spacing(self):
        self.assertEqual(clean("No mass. no   mass. Clear lungs."), "No mass. Clear lungs.")

    def test_decimal_measurement_is_kept(self):
        self.assertEqual(clean("A 1.5 mm focus. No mass."), "A 1.5 mm focus. No mass.")

    def test_model_tags_and_empty_output(self):
        self.assertEqual(clean("<start_of_turn>No mass.<end_of_turn>"), "No mass.")
        self.assertEqual(clean(" \n "), "")


if __name__ == "__main__":
    unittest.main()
