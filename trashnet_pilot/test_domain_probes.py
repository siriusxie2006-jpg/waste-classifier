import unittest
import numpy as np
from PIL import Image
from image_framing import fit_image
from domain_probes import probe_frame, MEAN_RGB, agreement_metrics


class DomainProbeTests(unittest.TestCase):
    def test_existing_padding_change_preserves_every_content_pixel(self):
        content = np.random.default_rng(4).integers(0, 256, (168, 224, 3), dtype=np.uint8)
        frame = fit_image(Image.fromarray(content))
        result = np.asarray(probe_frame(frame, 'existing_mean_pad', (0, 28, 224, 196)))
        np.testing.assert_array_equal(result[28:196], content)
        self.assertTrue((result[:28] == MEAN_RGB).all())
        self.assertTrue((result[196:] == MEAN_RGB).all())
        square = Image.fromarray(np.random.default_rng(2).integers(0, 256, (224, 224, 3), dtype=np.uint8))
        np.testing.assert_array_equal(probe_frame(square, 'existing_mean_pad', (0, 0, 224, 224)), square)

    def test_shrink_pair_differs_only_in_added_border(self):
        frame = Image.fromarray(np.random.default_rng(3).integers(0, 256, (224, 224, 3), dtype=np.uint8))
        white = np.asarray(probe_frame(frame, 'shrink75_white', (0, 0, 224, 224)))
        mean = np.asarray(probe_frame(frame, 'shrink75_mean', (0, 0, 224, 224)))
        np.testing.assert_array_equal(white[28:196, 28:196], mean[28:196, 28:196])
        self.assertTrue((white[:28] == 255).all())
        self.assertTrue((mean[:28] == MEAN_RGB).all())

    def test_rotations_are_reversible_without_pixel_removal(self):
        frame = Image.fromarray(np.random.default_rng(5).integers(0, 256, (224, 224, 3), dtype=np.uint8))
        rotated = probe_frame(frame, 'rotate90', (0, 0, 224, 224))
        np.testing.assert_array_equal(rotated.transpose(Image.Transpose.ROTATE_270), frame)
        rotated = probe_frame(frame, 'rotate180', (0, 0, 224, 224))
        np.testing.assert_array_equal(rotated.transpose(Image.Transpose.ROTATE_180), frame)

    def test_prediction_transitions_count_both_directions(self):
        reference = [[.9, .1], [.9, .1]]
        changed = [[.1, .9], [.1, .9]]
        metrics = agreement_metrics([0, 1], changed, reference)
        self.assertEqual(metrics['original_correct_now_wrong'], 1)
        self.assertEqual(metrics['original_wrong_now_correct'], 1)
        self.assertEqual(metrics['prediction_agreement'], 0)


if __name__ == '__main__':
    unittest.main()
