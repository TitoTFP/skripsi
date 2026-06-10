from __future__ import annotations

from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UNET_PRED_DIR = ROOT / "runs" / "final" / "unet" / "eval_test" / "predictions" / "Aceh_Utara"
PROCANET_PRED_DIR = ROOT / "runs" / "final" / "procanet" / "eval_test" / "predictions" / "Aceh_Utara"
TILE_DIR = ROOT / "dataset" / "tiles" / "7ch" / "by_region" / "Aceh_Utara"
OUTPUT_PATH = ROOT / "runs" / "final" / "evaluation_plots.png"


def find_best_tile() -> str:
    """Find a tile with high flood pixels in the valid mask."""
    best_tile = None
    max_flood_pixels = -1

    # Search through UNet predictions
    for pred_path in sorted(UNET_PRED_DIR.glob("*.npz")):
        data = np.load(pred_path)
        y = data["y"]  # target flood label
        eff_mask = data["effective_valid_mask"]

        # Only count flood pixels in effective valid area
        valid_flood_count = np.sum((y == 1) & eff_mask)

        # We want a tile that has a good amount of flood but not entirely flooded,
        # e.g., at least some background as well.
        total_valid = np.sum(eff_mask)
        if total_valid > 100000:  # Must have decent coverage
            # Prefer tiles with a mix (e.g. 10% to 50% flooded)
            ratio = valid_flood_count / total_valid
            if 0.1 < ratio < 0.6:
                if valid_flood_count > max_flood_pixels:
                    max_flood_pixels = valid_flood_count
                    best_tile = pred_path.name

    if best_tile is None:
        # Fallback to the one with the absolute most flood pixels
        for pred_path in sorted(UNET_PRED_DIR.glob("*.npz")):
            data = np.load(pred_path)
            y = data["y"]
            eff_mask = data["effective_valid_mask"]
            valid_flood_count = np.sum((y == 1) & eff_mask)
            if valid_flood_count > max_flood_pixels:
                max_flood_pixels = valid_flood_count
                best_tile = pred_path.name

    if best_tile is None:
        raise FileNotFoundError("Could not find any suitable tile with predictions.")

    print(f"Selected tile for visualization: {best_tile} with {max_flood_pixels} flood pixels")
    return best_tile


def main() -> None:
    # 1. Find a good tile
    tile_name = find_best_tile()

    # 2. Load feature tile (contains SAR, HSV, DEM)
    feature_path = TILE_DIR / tile_name
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature tile not found: {feature_path}")

    feature_data = np.load(feature_path)
    x = feature_data["x"]  # shape (7, 512, 512)

    # Urutan channel: VV, VH, Hue, Saturation, Value, Slope, HAND
    vv = x[0]
    vh = x[1]
    hue = x[2]
    sat = x[3]
    val = x[4]
    slope = x[5]
    hand = x[6]

    # Reconstruct pseudo-RGB from HSV
    # Matplotlib's hsv_to_rgb expects values in range [0, 1]
    # Check bounds first and normalize if necessary
    hsv_img = np.stack([hue, sat, val], axis=-1)
    # Clip to safety range [0, 1]
    hsv_img = np.clip(hsv_img, 0.0, 1.0)
    rgb = mcolors.hsv_to_rgb(hsv_img)

    # 3. Load Predictions
    unet_path = UNET_PRED_DIR / tile_name
    procanet_path = PROCANET_PRED_DIR / tile_name

    unet_data = np.load(unet_path)
    procanet_data = np.load(procanet_path)

    y_gt = unet_data["y"][0]  # shape (512, 512) or similar, let's extract 2D
    eff_mask = unet_data["effective_valid_mask"][0]

    unet_pred = unet_data["prediction"][0]
    procanet_pred = procanet_data["prediction"][0]

    # Create visual overlays where invalid area is greyed out
    # 0: non-flood, 1: flood, NaN/Masked: invalid
    def mask_invalid(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked = np.copy(image).astype(float)
        masked[~mask] = np.nan
        return masked

    y_gt_masked = mask_invalid(y_gt, eff_mask)
    unet_pred_masked = mask_invalid(unet_pred, eff_mask)
    procanet_pred_masked = mask_invalid(procanet_pred, eff_mask)

    # 4. Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel A: SAR (VV)
    im_sar = axes[0, 0].imshow(vv, cmap="gray")
    axes[0, 0].set_title("(a)", fontsize=30, y=-0.15)
    fig.colorbar(im_sar, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Panel B: Optical (Sentinel-2 HSV reconstructed to RGB)
    # Mask invalid S2 areas (if s2_valid_mask is available and 0)
    s2_mask = feature_data.get("s2_valid_mask")
    if s2_mask is not None:
        s2_mask_2d = s2_mask[0].astype(bool)
        rgb_masked = np.copy(rgb)
        rgb_masked[~s2_mask_2d] = 0.5  # color invalid areas gray
        axes[0, 1].imshow(rgb_masked)
    else:
        axes[0, 1].imshow(rgb)
    axes[0, 1].set_title("(b)", fontsize=30, y=-0.15)

    # Panel C: Topography (DEMNAS Slope)
    im_slope = axes[0, 2].imshow(slope, cmap="terrain")
    axes[0, 2].set_title("(c)", fontsize=30, y=-0.15)
    fig.colorbar(im_slope, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Define a custom color map for segmentation: 0 = Dark Green (non-flood), 1 = Blue (flood)
    cmap_seg = mcolors.ListedColormap(["#2ca02c", "#1f77b4"])
    cmap_seg.set_bad(color="#d3d3d3")  # Light gray for invalid area

    # Panel D: Ground Truth
    im_gt = axes[1, 0].imshow(y_gt_masked, cmap=cmap_seg, vmin=0, vmax=1)
    axes[1, 0].set_title("(d)", fontsize=30, y=-0.15)
    # Custom colorbar/legend
    cbar_gt = fig.colorbar(im_gt, ax=axes[1, 0], fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar_gt.ax.set_yticklabels(["Non-Flood", "Flood"])

    # Panel E: U-Net Prediction
    im_unet = axes[1, 1].imshow(unet_pred_masked, cmap=cmap_seg, vmin=0, vmax=1)
    axes[1, 1].set_title("(e)", fontsize=30, y=-0.15)
    cbar_unet = fig.colorbar(im_unet, ax=axes[1, 1], fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar_unet.ax.set_yticklabels(["Non-Flood", "Flood"])

    # Panel F: ProCANet Prediction
    im_pro = axes[1, 2].imshow(procanet_pred_masked, cmap=cmap_seg, vmin=0, vmax=1)
    axes[1, 2].set_title("(f)", fontsize=30, y=-0.15)
    cbar_pro = fig.colorbar(im_pro, ax=axes[1, 2], fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar_pro.ax.set_yticklabels(["Non-Flood", "Flood"])

    # Clean up axes ticks
    for row in axes:
        for ax in row:
            ax.set_xticks([])
            ax.set_yticks([])

    # fig.suptitle(f"Visualisasi Hasil Segmentasi Banjir Wilayah Aceh Utara\nTile: {tile_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
