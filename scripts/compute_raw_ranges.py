from pathlib import Path

import numpy as np
from osgeo import gdal

ROOT = Path(__file__).resolve().parents[1] / "dataset"
BLOCK_SIZE = 512


def update_range(path: Path, band_number: int, minimum: float, maximum: float) -> tuple[float, float]:
    try:
        dataset = gdal.Open(str(path))
        if dataset is None:
            raise FileNotFoundError(path)
        band = dataset.GetRasterBand(band_number)
        nodata = band.GetNoDataValue()

        for yoff in range(0, band.YSize, BLOCK_SIZE):
            for xoff in range(0, band.XSize, BLOCK_SIZE):
                values = band.ReadAsArray(
                    xoff,
                    yoff,
                    min(BLOCK_SIZE, band.XSize - xoff),
                    min(BLOCK_SIZE, band.YSize - yoff),
                )
                valid = np.isfinite(values)
                if nodata is not None:
                    valid &= values != nodata
                if valid.any():
                    minimum = min(minimum, float(values[valid].min()))
                    maximum = max(maximum, float(values[valid].max()))
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Failed reading band {band_number} from {path}") from error

    return minimum, maximum


def find_range(paths: list[Path], band_number: int = 1) -> tuple[float, float]:
    minimum, maximum = np.inf, -np.inf
    for path in paths:
        minimum, maximum = update_range(path, band_number, minimum, maximum)
    return minimum, maximum


def main() -> None:
    s1_paths = list((ROOT / "satelit raw").glob("*/S1_*.tif"))
    ranges = {
        "VV": find_range(s1_paths, 1),
        "VH": find_range(s1_paths, 2),
        "Hue": find_range(list((ROOT / "features_preprocessed").glob("*/hue.tif"))),
        "Saturation": find_range(list((ROOT / "features_preprocessed").glob("*/saturation.tif"))),
        "Value": find_range(list((ROOT / "features_preprocessed").glob("*/value.tif"))),
        "Slope": find_range(list((ROOT / "features_preprocessed").glob("*/slope_degrees.tif"))),
        "HAND": find_range(list((ROOT / "features_preprocessed").glob("*/hand_meters.tif"))),
        "Label banjir": (0.0, 1.0),
    }

    print("| Variabel | Rentang raw dataset |")
    print("|---|---:|")
    for variable, (minimum, maximum) in ranges.items():
        print(f"| {variable} | {minimum:.15g} sampai {maximum:.15g} |")


if __name__ == "__main__":
    main()
