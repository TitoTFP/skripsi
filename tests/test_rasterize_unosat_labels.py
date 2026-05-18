import unittest
from pathlib import Path

from osgeo import ogr

from scripts.rasterize_unosat_labels import (
    LabelLayerSets,
    apply_roi_mask,
    classify_unosat_layers,
    find_admin_boundary,
    find_s1_reference,
    merge_binary_masks,
    output_region_name,
    repair_layer_geometries,
)


class RasterizeUnosatLabelTests(unittest.TestCase):
    def test_classify_unosat_layers_keeps_flood_valid_and_auxiliary_masks_separate(self):
        layers = [
            "S1_20251127_AnalysisExtent_WestAcehProvince",
            "S1_20251127_FloodExtent_WestAcehProvince",
            "S1_20251127_WaterExtent_WestAcehProvince",
            "S1_20251128_FloodExtent_AcehProvince_SumateraProvince",
            "S1_20251128_WaterExtent_AcehProvince_SumateraProvince",
            "S1_20251128_AnalysisExtent_AcehProvince_SumateraProvince",
            "ST2_20251129_AnalysisExtent_AcehProvince",
            "ST2_20251129_River_AcehProvince",
            "ST2_20251129_FloodExtent_AcehProvince",
        ]

        classified = classify_unosat_layers(layers)

        self.assertEqual(
            classified,
            LabelLayerSets(
                flood=(
                    "S1_20251127_FloodExtent_WestAcehProvince",
                    "S1_20251128_FloodExtent_AcehProvince_SumateraProvince",
                    "ST2_20251129_FloodExtent_AcehProvince",
                ),
                valid=(
                    "S1_20251127_AnalysisExtent_WestAcehProvince",
                    "S1_20251128_AnalysisExtent_AcehProvince_SumateraProvince",
                    "ST2_20251129_AnalysisExtent_AcehProvince",
                ),
                water_river=(
                    "S1_20251127_WaterExtent_WestAcehProvince",
                    "S1_20251128_WaterExtent_AcehProvince_SumateraProvince",
                    "ST2_20251129_River_AcehProvince",
                ),
            ),
        )

    def test_region_output_names_match_existing_dataset_layout(self):
        self.assertEqual(output_region_name("Kota Banda Aceh"), "Banda_Aceh")
        self.assertEqual(output_region_name("Kota Langsa"), "Langsa")
        self.assertEqual(output_region_name("Aceh Tamiang"), "Aceh_Tamiang")

    def test_find_admin_boundary_matches_region_geojson_name(self):
        root = Path("dataset/batas admin indo")

        self.assertEqual(
            find_admin_boundary("Kota Langsa", root),
            root / "Kota Langsa-KAB_KOTA.geojson",
        )
        self.assertEqual(
            find_admin_boundary("Aceh Tamiang", root),
            root / "Kabupaten Aceh Tamiang-KAB_KOTA.geojson",
        )

    def test_find_admin_boundary_fails_when_region_roi_is_missing(self):
        with self.assertRaisesRegex(FileNotFoundError, "No admin boundary"):
            find_admin_boundary("Aceh Tamiang", Path("missing-root"))

    def test_apply_roi_mask_intersects_valid_mask_with_region_roi(self):
        valid = [[1, 1, 0], [1, 0, 1]]
        roi = [[1, 0, 1], [1, 1, 0]]

        self.assertEqual(apply_roi_mask(valid, roi), [[1, 0, 0], [1, 0, 0]])

    def test_merge_binary_masks_uses_pixelwise_or_after_per_layer_rasterization(self):
        masks = [
            [[1, 0, 0], [0, 1, 0]],
            [[0, 1, 0], [0, 1, 1]],
            [[0, 0, 0], [1, 0, 0]],
        ]

        self.assertEqual(merge_binary_masks(masks), [[1, 1, 0], [1, 1, 1]])

    def test_merge_binary_masks_rejects_empty_layer_group(self):
        with self.assertRaisesRegex(ValueError, "No raster masks"):
            merge_binary_masks([])

    def test_repair_layer_geometries_makes_invalid_polygon_rasterizable(self):
        driver = ogr.GetDriverByName("MEM")
        source = driver.CreateDataSource("source")
        layer = source.CreateLayer("invalid", geom_type=ogr.wkbPolygon)
        feature = ogr.Feature(layer.GetLayerDefn())
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for x, y in [(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]:
            ring.AddPoint(x, y)
        polygon = ogr.Geometry(ogr.wkbPolygon)
        polygon.AddGeometry(ring)
        self.assertFalse(polygon.IsValid())
        feature.SetGeometry(polygon)
        layer.CreateFeature(feature)

        repaired_ds, repaired_layer = repair_layer_geometries(layer, "repaired")
        repaired_feature = next(iter(repaired_layer))
        repaired_geometry = repaired_feature.GetGeometryRef()

        self.assertIsNotNone(repaired_ds)
        self.assertTrue(repaired_geometry.IsValid())

    def test_find_s1_reference_requires_exactly_one_s1_tif(self):
        with self.subTest("one S1"):
            files = [
                Path("S2_Aceh_Besar.tif"),
                Path("S1_Aceh_Besar.tif"),
                Path("notes.txt"),
            ]
            self.assertEqual(find_s1_reference(files), Path("S1_Aceh_Besar.tif"))

        with self.subTest("none"):
            with self.assertRaisesRegex(FileNotFoundError, "No S1 GeoTIFF"):
                find_s1_reference([Path("S2_Aceh_Besar.tif")])

        with self.subTest("many"):
            with self.assertRaisesRegex(ValueError, "Multiple S1 GeoTIFF"):
                find_s1_reference([Path("S1_a.tif"), Path("S1_b.tif")])


if __name__ == "__main__":
    unittest.main()
