from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TILE_ROOT = ROOT / "dataset/tiles/7ch"


def load_tile(tile_path: Path) -> dict[str, np.ndarray]:
    if not tile_path.exists():
        raise FileNotFoundError(f"Tile tidak ditemukan: {tile_path}")

    data = np.load(tile_path)

    required_keys = ["valid_mask", "feature_valid_mask"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Key '{key}' tidak ada di {tile_path}")

    return {
        "label_valid_mask": data["valid_mask"][0].astype(bool),
        "feature_valid_mask": data["feature_valid_mask"][0].astype(bool),
    }


def compare_masks(label_valid_mask: np.ndarray, feature_valid_mask: np.ndarray) -> dict[str, float]:
    both_valid = label_valid_mask & feature_valid_mask
    label_only = label_valid_mask & ~feature_valid_mask
    feature_only = ~label_valid_mask & feature_valid_mask
    both_invalid = ~label_valid_mask & ~feature_valid_mask

    total = label_valid_mask.size
    union = label_valid_mask | feature_valid_mask
    intersection = both_valid

    iou = intersection.sum() / union.sum() if union.sum() > 0 else 0.0
    agreement = (label_valid_mask == feature_valid_mask).sum() / total

    return {
        "total_pixels": total,
        "both_valid": int(both_valid.sum()),
        "label_only": int(label_only.sum()),
        "feature_only": int(feature_only.sum()),
        "both_invalid": int(both_invalid.sum()),
        "label_valid_ratio": float(label_valid_mask.mean()),
        "feature_valid_ratio": float(feature_valid_mask.mean()),
        "iou": float(iou),
        "agreement": float(agreement),
    }


def make_comparison_map(label_valid_mask: np.ndarray, feature_valid_mask: np.ndarray) -> np.ndarray:
    """
    Kode kategori:
    0 = sama-sama invalid
    1 = hanya label valid
    2 = hanya feature valid
    3 = sama-sama valid
    """
    comparison = np.zeros(label_valid_mask.shape, dtype=np.uint8)
    comparison[label_valid_mask & ~feature_valid_mask] = 1
    comparison[~label_valid_mask & feature_valid_mask] = 2
    comparison[label_valid_mask & feature_valid_mask] = 3
    return comparison


def visualize(tile_path: Path, output_path: Path | None = None) -> None:
    masks = load_tile(tile_path)

    label_valid_mask = masks["label_valid_mask"]
    feature_valid_mask = masks["feature_valid_mask"]
    comparison = make_comparison_map(label_valid_mask, feature_valid_mask)
    stats = compare_masks(label_valid_mask, feature_valid_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(label_valid_mask, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("label_valid_mask\n(key: valid_mask)")

    axes[1].imshow(feature_valid_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("feature_valid_mask")

    im = axes[2].imshow(comparison, vmin=0, vmax=3)
    axes[2].set_title(
        "Perbandingan\n"
        "0=both invalid, 1=label only, 2=feature only, 3=both valid"
    )

    for ax in axes:
        ax.axis("off")

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{tile_path.name}\n"
        f"IoU valid area: {stats['iou']:.4f} | "
        f"Agreement: {stats['agreement']:.4f} | "
        f"Label valid: {stats['label_valid_ratio']:.2%} | "
        f"Feature valid: {stats['feature_valid_ratio']:.2%}",
        fontsize=11,
    )

    fig.tight_layout()

    print("=== Statistik Perbandingan Mask ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\nVisualisasi disimpan ke: {output_path}")
    else:
        plt.show()


def get_tile_from_split(split: str, index: int) -> Path:
    split_dir = TILE_ROOT / split
    tiles = sorted(split_dir.glob("*.npz"))

    if not tiles:
        raise FileNotFoundError(f"Tidak ada tile .npz di {split_dir}")

    if index < 0 or index >= len(tiles):
        raise IndexError(f"Index {index} di luar range. Total tile: {len(tiles)}")

    return tiles[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualisasi perbandingan label_valid_mask dan feature_valid_mask."
    )

    parser.add_argument(
        "--tile",
        type=Path,
        default=None,
        help="Path langsung ke file tile .npz.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Split dataset jika tidak memakai --tile.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index tile pada split yang dipilih.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path output PNG. Jika kosong, visualisasi langsung ditampilkan.",
    )

    args = parser.parse_args()

    tile_path = args.tile if args.tile is not None else get_tile_from_split(args.split, args.index)
    visualize(tile_path=tile_path, output_path=args.output)


if __name__ == "__main__":
    main()
