import unittest

import numpy as np

from scripts.preprocessing_utils import (
    choose_split,
    normalize_db,
    rgb_to_hsv,
    should_keep_tile,
)


class PreprocessingHelperTests(unittest.TestCase):
    def test_normalize_db_clips_to_unit_interval(self):
        values = np.array([-40.0, -30.0, -15.0, 0.0, 5.0], dtype=np.float32)
        out = normalize_db(values)
        np.testing.assert_allclose(out, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_rgb_to_hsv_sets_invalid_pixels_to_zero(self):
        rgb = np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        valid = np.array([[True, True], [True, False]])
        hsv = rgb_to_hsv(rgb, valid)
        self.assertEqual(hsv.shape, (3, 2, 2))
        np.testing.assert_allclose(hsv[:, 0, 0], [0.0, 1.0, 1.0])
        np.testing.assert_allclose(hsv[:, 0, 1], [1.0 / 3.0, 1.0, 1.0])
        np.testing.assert_allclose(hsv[:, 1, 0], [2.0 / 3.0, 1.0, 1.0])
        np.testing.assert_allclose(hsv[:, 1, 1], [0.0, 0.0, 0.0])

    def test_should_keep_tile_keeps_positive_even_with_low_coverage(self):
        label_valid = np.zeros((4, 4), dtype=bool)
        feature_valid = np.ones((4, 4), dtype=bool)
        flood = np.zeros((4, 4), dtype=bool)
        flood[0, 0] = True
        self.assertTrue(should_keep_tile(label_valid, feature_valid, flood))

    def test_should_keep_tile_drops_background_with_low_coverage(self):
        label_valid = np.zeros((4, 4), dtype=bool)
        feature_valid = np.ones((4, 4), dtype=bool)
        flood = np.zeros((4, 4), dtype=bool)
        self.assertFalse(should_keep_tile(label_valid, feature_valid, flood))

    def test_choose_split_matches_region_policy(self):
        self.assertEqual(choose_split("Pidie_Jaya"), "val")
        self.assertEqual(choose_split("Pidie"), "test")
        self.assertEqual(choose_split("Aceh_Timur"), "train")


if __name__ == "__main__":
    unittest.main()
