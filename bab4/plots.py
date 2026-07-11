from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def normalize_image(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [2, 98])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h = np.asarray(hsv[0], dtype=np.float32)
    s = np.asarray(hsv[1], dtype=np.float32)
    v = np.asarray(hsv[2], dtype=np.float32)
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = i % 6
    rgb = np.zeros((h.shape[0], h.shape[1], 3), dtype=np.float32)
    choices = [
        (v, t, p),
        (q, v, p),
        (p, v, t),
        (p, q, v),
        (t, p, v),
        (v, p, q),
    ]
    for idx, channels in enumerate(choices):
        mask = i_mod == idx
        for band, values in enumerate(channels):
            rgb[..., band][mask] = values[mask]
    return np.clip(rgb, 0, 1)


def hsv_to_display_rgb(hsv: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert HSV to pseudo-RGB for figures with display-only value stretching."""
    hsv_arr = np.asarray(hsv, dtype=np.float32).copy()
    value = hsv_arr[2]
    if valid_mask is None:
        valid = np.isfinite(value) & (value > 0)
    else:
        valid = np.asarray(valid_mask).astype(bool) & np.isfinite(value)
    if valid.any():
        stretched = np.zeros_like(value, dtype=np.float32)
        stretched[valid] = np.power(normalize_image(value[valid]), 0.6)
        hsv_arr[2] = stretched
    return hsv_to_rgb(hsv_arr)


def simple_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(labels, values, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, path)


def plot_osm_cache_lines(ax, cache_paths: Iterable[Path], dataset, *, max_ways: int = 700) -> int:
    node_map: dict[int, tuple[float, float]] = {}
    ways = []
    for cache_path in cache_paths:
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for elem in payload.get("elements", []):
            if elem.get("type") == "node" and "lat" in elem and "lon" in elem:
                node_map[int(elem["id"])] = (float(elem["lon"]), float(elem["lat"]))
            elif elem.get("type") == "way" and elem.get("nodes"):
                tags = elem.get("tags", {})
                if any(key in tags for key in ("highway", "waterway", "natural", "building")):
                    ways.append(elem)
        if ways:
            break
    if not ways or not node_map:
        return 0

    transformer = _lonlat_to_dataset_transformer(dataset)
    gt = dataset.GetGeoTransform()
    inv_ok, inv_gt = _invert_geotransform(gt)
    if not inv_ok:
        return 0
    plotted = 0
    for way in ways[:max_ways]:
        coords = [node_map.get(int(node_id)) for node_id in way.get("nodes", [])]
        coords = [coord for coord in coords if coord is not None]
        if len(coords) < 2:
            continue
        pixels = []
        for lon, lat in coords:
            x, y = transformer(lon, lat)
            px = inv_gt[0] + inv_gt[1] * x + inv_gt[2] * y
            py = inv_gt[3] + inv_gt[4] * x + inv_gt[5] * y
            pixels.append((px, py))
        xs, ys = zip(*pixels)
        tags = way.get("tags", {})
        color = "#1f2937"
        lw = 0.35
        if "waterway" in tags or tags.get("natural") == "water":
            color, lw = "#0ea5e9", 0.6
        elif "highway" in tags:
            color, lw = "#f97316", 0.45
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.7)
        plotted += 1
    return plotted


def _lonlat_to_dataset_transformer(dataset):
    projection = dataset.GetProjection()
    try:
        from pyproj import CRS, Transformer

        transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_wkt(projection), always_xy=True)
        return lambda lon, lat: transformer.transform(lon, lat)
    except Exception:
        from osgeo import osr

        source = osr.SpatialReference()
        source.ImportFromEPSG(4326)
        target = osr.SpatialReference()
        target.ImportFromWkt(projection)
        if hasattr(osr, "OAMS_TRADITIONAL_GIS_ORDER"):
            source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        transformer = osr.CoordinateTransformation(source, target)
        return lambda lon, lat: transformer.TransformPoint(lon, lat)[:2]


def _invert_geotransform(gt):
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if det == 0:
        return False, gt
    inv_det = 1.0 / det
    inv_gt = (
        (gt[2] * gt[3] - gt[0] * gt[5]) * inv_det,
        gt[5] * inv_det,
        -gt[2] * inv_det,
        (gt[0] * gt[4] - gt[1] * gt[3]) * inv_det,
        -gt[4] * inv_det,
        gt[1] * inv_det,
    )
    return True, inv_gt
