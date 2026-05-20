from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from osgeo import gdal

from scripts.export_demnas import crop_dem_to_boundary


class ExportDemnasTests(unittest.TestCase):
    def setUp(self) -> None:
        gdal.UseExceptions()
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_crop_dem_to_boundary(self) -> None:
        # Create a mock geojson boundary (coordinates in EPSG:4326/UTM or dummy CRS coordinates)
        geojson_path = self.tmp_dir / "boundary.geojson"
        boundary = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [100.0, 100.0],
                                [110.0, 100.0],
                                [110.0, 110.0],
                                [100.0, 110.0],
                                [100.0, 100.0],
                            ]
                        ],
                    },
                    "properties": {},
                }
            ],
        }
        with geojson_path.open("w") as f:
            json.dump(boundary, f)

        # Create a larger input DEM (e.g., covering 90 to 120, resolution 2 units)
        input_dem = self.tmp_dir / "input_dem.tif"
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(input_dem), 20, 20, 1, gdal.GDT_Float32)
        # origin_x=90, width=2, origin_y=120, height=-2
        ds.SetGeoTransform([90.0, 2.0, 0.0, 120.0, 0.0, -2.0])
        # WGS 84
        sr = gdal.osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        ds.SetProjection(sr.ExportToWkt())
        band = ds.GetRasterBand(1)
        band.WriteArray(np.ones((20, 20), dtype=np.float32) * 5.0)
        ds.FlushCache()
        ds = None

        # Output path
        output_path = self.tmp_dir / "output_dem.tif"
        crop_dem_to_boundary(input_dem, geojson_path, output_path)

        # Verify output exists
        self.assertTrue(output_path.exists())

        # Verify size and extent: it should be cropped to the geojson cutline
        out_ds = gdal.Open(str(output_path), gdal.GA_ReadOnly)
        self.assertIsNotNone(out_ds)
        
        # Checking cutline output bounds:
        # GeoJSON is [100.0, 100.0, 110.0, 110.0]
        # Pixel size is 2.0. So it should crop to approximately 5x5 pixels (size 10x10)
        gt = out_ds.GetGeoTransform()
        self.assertAlmostEqual(gt[0], 100.0)
        self.assertAlmostEqual(gt[3], 110.0)
        self.assertAlmostEqual(gt[1], 2.0)
        self.assertAlmostEqual(gt[5], -2.0)
        self.assertEqual(out_ds.RasterXSize, 5)
        self.assertEqual(out_ds.RasterYSize, 5)

        band = out_ds.GetRasterBand(1)
        arr = band.ReadAsArray()
        # All pixels inside the cutline should have value 5.0
        np.testing.assert_allclose(arr, 5.0)

        out_ds = None


if __name__ == "__main__":
    unittest.main()
