import unittest
from external_metrics import material_metrics


class ExternalMetricsTests(unittest.TestCase):
    def test_trash_output_remains_an_error(self):
        metrics, pred, _ = material_metrics([0, 1, 2, 3, 4], [
            [.9, .02, .02, .02, .02, .02], [.02, .9, .02, .02, .02, .02],
            [.02, .02, .9, .02, .02, .02], [.02, .02, .02, .9, .02, .02],
            [.01, .01, .01, .01, .35, .61]])
        self.assertEqual(pred[-1], 5)
        self.assertAlmostEqual(metrics['accuracy'], .8)
        self.assertAlmostEqual(metrics['macro_f1_five_materials'], .8)
        self.assertEqual(metrics['classification_report']['plastic']['recall'], 0)

    def test_high_score_wrong_answer_is_not_reliably_flagged(self):
        metrics, _, review = material_metrics([4], [[.01, .91, .02, .02, .03, .01]])
        self.assertFalse(review[0])
        self.assertEqual(metrics['plastic_as_glass_count'], 1)
        self.assertEqual(metrics['unflagged_error_count'], 1)
        self.assertEqual(metrics['unflagged_error_rate'], 1)

    def test_unmapped_labels_and_five_output_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            material_metrics([5], [[0, 0, 0, 0, 0, 1]])
        with self.assertRaises(ValueError):
            material_metrics([4], [[0, 0, 0, 0, 1]])


if __name__ == '__main__':
    unittest.main()
