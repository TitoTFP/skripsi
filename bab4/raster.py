from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Prefer the project dependency from pyproject.toml.
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
except ModuleNotFoundError:  # pragma: no cover - exercised on system-python fallback envs
    rasterio = None
    Resampling = None
    Window = None

try:  # Fallback for environments that have python-gdal but not rasterio.
    from osgeo import gdal

    gdal.UseExceptions()
except ModuleNotFoundError:  # pragma: no cover - depends on local env
    gdal = None


@dataclass
class RasterDataset:
    path: Path
    RasterXSize: int
    RasterYSize: int
    RasterCount: int
    crs: str
    transform: tuple[float, float, float, float, float, float]
    dtypes: tuple[str, ...]
    descriptions: tuple[str, ...]
    backend: str
    _gdal_dataset: object | None = None

    def GetProjection(self) -> str:
        return self.crs

    def GetGeoTransform(self) -> tuple[float, float, float, float, float, float]:
        return self.transform

    def GetRasterBand(self, index: int) -> "RasterBand":
        return RasterBand(self, index)


@dataclass
class RasterBand:
    dataset: RasterDataset
    index: int

    @property
    def DataType(self) -> str | int:
        if self.dataset.backend == "gdal" and self.dataset._gdal_dataset is not None:
            return self.dataset._gdal_dataset.GetRasterBand(self.index).DataType
        return self.dataset.dtypes[self.index - 1]

    def GetDescription(self) -> str:
        if self.dataset.backend == "gdal" and self.dataset._gdal_dataset is not None:
            return self.dataset._gdal_dataset.GetRasterBand(self.index).GetDescription()
        return self.dataset.descriptions[self.index - 1] or ""

    def GetStatistics(self, approx_ok: bool = True, force: bool = True) -> tuple[float, float, float, float]:
        if self.dataset.backend == "gdal" and self.dataset._gdal_dataset is not None:
            return tuple(self.dataset._gdal_dataset.GetRasterBand(self.index).GetStatistics(approx_ok, force))
        arr = self.ReadAsArray()
        finite = np.isfinite(arr)
        if not finite.any():
            return 0.0, 0.0, 0.0, 0.0
        values = arr[finite]
        return float(values.min()), float(values.max()), float(values.mean()), float(values.std())

    def ReadAsArray(
        self,
        xoff: int = 0,
        yoff: int = 0,
        xsize: int | None = None,
        ysize: int | None = None,
        *,
        buf_xsize: int | None = None,
        buf_ysize: int | None = None,
    ) -> np.ndarray:
        if self.dataset.backend == "gdal" and self.dataset._gdal_dataset is not None:
            band = self.dataset._gdal_dataset.GetRasterBand(self.index)
            if xsize is None or ysize is None:
                return band.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
            return band.ReadAsArray(xoff, yoff, xsize, ysize, buf_xsize=buf_xsize, buf_ysize=buf_ysize)
        if rasterio is None or Window is None:
            raise ModuleNotFoundError("rasterio or osgeo is required to read GeoTIFF rasters")
        with rasterio.open(self.dataset.path) as src:
            window = None
            if xsize is not None and ysize is not None:
                window = Window(xoff, yoff, xsize, ysize)
            out_shape = None
            if buf_xsize is not None and buf_ysize is not None:
                out_shape = (buf_ysize, buf_xsize)
            return src.read(
                self.index,
                window=window,
                out_shape=out_shape,
                resampling=Resampling.bilinear if Resampling is not None else 1,
            )


def raster_metadata(path: Path) -> dict[str, Any]:
    ds = open_dataset(path)
    return {
        "path": str(path),
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "count": ds.RasterCount,
        "crs": ds.GetProjection(),
        "transform": ds.GetGeoTransform(),
        "pixel_width": abs(ds.GetGeoTransform()[1]),
        "dtype": ds.GetRasterBand(1).DataType if ds.RasterCount else None,
    }


def npz_keys(path: Path) -> tuple[str, ...]:
    with np.load(path) as payload:
        return tuple(payload.keys())


def open_dataset(path: Path) -> RasterDataset:
    path = Path(path)
    if rasterio is not None:
        with rasterio.open(path) as src:
            return RasterDataset(
                path=path,
                RasterXSize=src.width,
                RasterYSize=src.height,
                RasterCount=src.count,
                crs=src.crs.to_wkt() if src.crs else "",
                transform=tuple(float(value) for value in src.transform.to_gdal()),
                dtypes=tuple(str(dtype) for dtype in src.dtypes),
                descriptions=tuple(description or "" for description in src.descriptions),
                backend="rasterio",
            )
    if gdal is not None:
        ds = gdal.Open(str(path), gdal.GA_ReadOnly)
        if ds is None:
            raise FileNotFoundError(path)
        dtypes = tuple(_gdal_dtype_name(ds.GetRasterBand(index).DataType) for index in range(1, ds.RasterCount + 1))
        descriptions = tuple(ds.GetRasterBand(index).GetDescription() or "" for index in range(1, ds.RasterCount + 1))
        return RasterDataset(
            path=path,
            RasterXSize=ds.RasterXSize,
            RasterYSize=ds.RasterYSize,
            RasterCount=ds.RasterCount,
            crs=ds.GetProjection(),
            transform=tuple(float(value) for value in ds.GetGeoTransform()),
            dtypes=dtypes,
            descriptions=descriptions,
            backend="gdal",
            _gdal_dataset=ds,
        )
    raise ModuleNotFoundError("rasterio or osgeo is required to read GeoTIFF rasters")


def band_stats(path: Path, band_index: int = 1) -> dict[str, float | int]:
    ds = open_dataset(path)
    band = ds.GetRasterBand(band_index)
    min_val, max_val, mean_val, std_val = band.GetStatistics(True, True)
    return {
        "count": int(ds.RasterXSize * ds.RasterYSize),
        "min": float(min_val),
        "max": float(max_val),
        "mean": float(mean_val),
        "std": float(std_val),
    }


def read_raster(path: Path, band_index: int = 1, *, max_size: int = 700) -> np.ndarray:
    ds = open_dataset(path)
    width, height = ds.RasterXSize, ds.RasterYSize
    scale = min(max_size / width, max_size / height, 1.0)
    out_w = max(1, int(width * scale))
    out_h = max(1, int(height * scale))
    return ds.GetRasterBand(band_index).ReadAsArray(buf_xsize=out_w, buf_ysize=out_h).astype(np.float32)


def read_raster_window(path: Path, row: int, col: int, size: int = 512, band_index: int = 1) -> np.ndarray:
    ds = open_dataset(path)
    width = min(size, ds.RasterXSize - col)
    height = min(size, ds.RasterYSize - row)
    return ds.GetRasterBand(band_index).ReadAsArray(col, row, width, height).astype(np.float32)


def read_stack_rgb(feature_dir: Path, *, max_size: int = 700) -> np.ndarray:
    hue = read_raster(feature_dir / "hue.tif", max_size=max_size)
    sat = read_raster(feature_dir / "saturation.tif", max_size=max_size)
    val = read_raster(feature_dir / "value.tif", max_size=max_size)
    from bab4.plots import hsv_to_rgb

    return hsv_to_rgb(np.stack([hue, sat, val]))


def load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def effective_valid_mask(tile: dict[str, Any]) -> np.ndarray:
    valid = np.asarray(tile["valid_mask"]).astype(bool)
    feature = np.asarray(tile.get("feature_valid_mask", valid)).astype(bool)
    return valid & feature


def tile_region(path: Path) -> str:
    with np.load(path, allow_pickle=False) as payload:
        if "region" in payload.files:
            return str(payload["region"])
    return path.stem.split("_r", 1)[0]


def dtype_name(dtype: str | int) -> str:
    if isinstance(dtype, str):
        return dtype
    return _gdal_dtype_name(dtype)


def _gdal_dtype_name(dtype: int) -> str:
    if gdal is None:
        return str(dtype)
    return gdal.GetDataTypeName(dtype)
