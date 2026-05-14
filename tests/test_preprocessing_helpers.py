import unittest

import numpy as np

from scripts.make_procanet_tiles import build_procanet_payload
from scripts.preprocessing_utils import (
    CHANNELS_7CH,
    PROCANET_ENCODER1_CHANNELS,
    PROCANET_ENCODER2_CHANNELS,
    choose_split,
    normalize_db,
    rgb_to_hsv,
    should_keep_tile,
    split_procanet_encoders,
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

    def test_split_procanet_encoders_repeats_sar_as_water_modality(self):
        stack = np.arange(7 * 2 * 2, dtype=np.float32).reshape(7, 2, 2)
        encoder1, encoder2 = split_procanet_encoders(stack)

        np.testing.assert_array_equal(encoder1, stack)
        np.testing.assert_array_equal(encoder2, stack[:2])
        self.assertEqual(PROCANET_ENCODER1_CHANNELS, CHANNELS_7CH)
        self.assertEqual(PROCANET_ENCODER2_CHANNELS, ("VV", "VH"))

    def test_build_procanet_payload_preserves_masks_and_metadata(self):
        source = {
            "x": np.arange(7 * 2 * 2, dtype=np.float32).reshape(7, 2, 2),
            "y": np.ones((1, 2, 2), dtype=np.uint8),
            "valid_mask": np.ones((1, 2, 2), dtype=np.uint8),
            "water_river_mask": np.zeros((1, 2, 2), dtype=np.uint8),
            "feature_valid_mask": np.ones((1, 2, 2), dtype=np.uint8),
            "s2_valid_mask": np.zeros((1, 2, 2), dtype=np.uint8),
            "region": np.array("Aceh_Timur"),
            "row": np.array(512),
            "col": np.array(1024),
            "channels": np.array(CHANNELS_7CH),
        }

        payload = build_procanet_payload(source)

        np.testing.assert_array_equal(payload["x_encoder1"], source["x"])
        np.testing.assert_array_equal(payload["x_encoder2"], source["x"][:2])
        np.testing.assert_array_equal(payload["y"], source["y"])
        np.testing.assert_array_equal(payload["valid_mask"], source["valid_mask"])
        self.assertEqual(tuple(payload["encoder1_channels"].tolist()), CHANNELS_7CH)
        self.assertEqual(tuple(payload["encoder2_channels"].tolist()), ("VV", "VH"))


if __name__ == "__main__":
    unittest.main()
