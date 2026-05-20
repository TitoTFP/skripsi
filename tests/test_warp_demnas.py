from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from osgeo import gdal

from scripts.warp_demnas import find_s1_reference, warp_dem


class WarpDemnasTests(unittest.TestCase):
    def setUp(self) -> None:
        gdal.UseExceptions()
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_find_s1_reference_returns_matching_file(self) -> None:
        region_dir = self.tmp_dir / "region"
        region_dir.mkdir()
        s1_file = region_dir / "S1_Aceh_Besar.tif"
        s1_file.touch()
        (region_dir / "S2_Aceh_Besar.tif").touch()
        (region_dir / "notes.txt").touch()

        ref = find_s1_reference(region_dir)
        self.assertEqual(ref, s1_file)

    def test_find_s1_reference_raises_if_missing(self) -> None:
        region_dir = self.tmp_dir / "region"
        region_dir.mkdir()
        (region_dir / "S2_Aceh_Besar.tif").touch()

        with self.assertRaisesRegex(FileNotFoundError, "No S1 GeoTIFF found"):
            find_s1_reference(region_dir)

    def test_find_s1_reference_raises_if_multiple(self) -> None:
        region_dir = self.tmp_dir / "region"
        region_dir.mkdir()
        (region_dir / "S1_a.tif").touch()
        (region_dir / "S1_b.tif").touch()

        with self.assertRaisesRegex(ValueError, "Multiple S1 GeoTIFF files"):
            find_s1_reference(region_dir)

    def test_warp_dem_aligns_to_reference_grid(self) -> None:
        # Create a mock reference S1 image (e.g. 10x10)
        ref_path = self.tmp_dir / "ref.tif"
        driver = gdal.GetDriverByName("GTiff")
        ref_ds = driver.Create(str(ref_path), 10, 10, 1, gdal.GDT_Float32)
        ref_ds.SetGeoTransform([100000.0, 10.0, 0.0, 200000.0, 0.0, -10.0])
        # WGS 84 / UTM zone 47N
        sr = gdal.osr.SpatialReference()
        sr.ImportFromEPSG(32647)
        ref_ds.SetProjection(sr.ExportToWkt())
        import numpy as np
        band = ref_ds.GetRasterBand(1)
        band.WriteArray(np.ones((10, 10), dtype=np.float32))
        ref_ds.FlushCache()
        ref_ds = None

        # Create a mock raw DEM (e.g. 5x5, larger extent, different resolution)
        dem_raw_path = self.tmp_dir / "dem_raw.tif"
        dem_ds = driver.Create(str(dem_raw_path), 5, 5, 1, gdal.GDT_Float32)
        dem_ds.SetGeoTransform([99950.0, 20.0, 0.0, 200050.0, 0.0, -20.0])
        dem_ds.SetProjection(sr.ExportToWkt())
        band = dem_ds.GetRasterBand(1)
        import numpy as np
        band.WriteArray(np.arange(25, dtype=np.float32).reshape(5, 5))
        dem_ds.FlushCache()
        dem_ds = None

        # Warp
        dem_out_path = self.tmp_dir / "dem_out.tif"
        warp_dem(dem_raw_path, ref_path, dem_out_path)

        # Verify output
        self.assertTrue(dem_out_path.exists())
        out_ds = gdal.Open(str(dem_out_path), gdal.GA_ReadOnly)
        self.assertEqual(out_ds.RasterXSize, 10)
        self.assertEqual(out_ds.RasterYSize, 10)
        
        # Check GeoTransform matches
        out_gt = out_ds.GetGeoTransform()
        self.assertEqual(out_gt, (100000.0, 10.0, 0.0, 200000.0, 0.0, -10.0))
        
        # Check Projection matches
        self.assertEqual(out_ds.GetProjection(), sr.ExportToWkt())
        
        out_ds = None


if __name__ == "__main__":
    unittest.main()
