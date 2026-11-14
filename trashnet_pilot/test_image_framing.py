"""Image-extent and user-crop regression tests, independent of labels."""
import unittest
import numpy as np
from PIL import Image, ImageDraw
from inference import crop_image, prepare_image


class ImageFramingTests(unittest.TestCase):
    def test_fit_preserves_both_edges_of_wide_image(self):
        image = Image.new('RGB', (900, 300), (0, 200, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 80, 299), fill=(255, 0, 0))
        draw.rectangle((819, 0, 899, 299), fill=(0, 0, 255))
        full = np.array(prepare_image(image, framing='fit'))
        center = np.array(prepare_image(image, framing='center'))
        self.assertEqual(full.shape, (224, 224, 3))
        self.assertTrue(np.any(np.all(full == [255, 0, 0], axis=2)))
        self.assertTrue(np.any(np.all(full == [0, 0, 255], axis=2)))
        self.assertFalse(np.any(np.all(center == [255, 0, 0], axis=2)))
        self.assertFalse(np.any(np.all(center == [0, 0, 255], axis=2)))

    def test_no_user_crop_means_whole_image(self):
        image = Image.new('RGB', (300, 800), 'red')
        self.assertEqual(crop_image(image).size, image.size)
        full = prepare_image(image, framing='fit')
        self.assertEqual(full.getpixel((112, 0)), (255, 0, 0))
        self.assertEqual(full.getpixel((112, 223)), (255, 0, 0))

    def test_only_explicit_user_rectangle_is_cropped(self):
        image = Image.new('RGB', (100, 80))
        self.assertEqual(crop_image(image, (.1, .25, .9, .75)).size, (80, 40))
        for box in [(-.1, 0, 1, 1), (0, 0, 0, 1), (0, 0, float('nan'), 1)]:
            with self.assertRaises(ValueError):
                crop_image(image, box)


if __name__ == '__main__':
    unittest.main(verbosity=2)
