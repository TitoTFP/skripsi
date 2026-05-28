from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.preprocessing_utils import CV_REGIONS
from training.datasets import FloodTileDataset
from training.models import ProCANet, UNet
from training.train import train_one_epoch, write_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train final flood segmentation models on all train+val regions.")
    parser.add_argument("--architecture", choices=("unet", "procanet"), required=True)
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs. Default: unet=21, procanet=18.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size. Default: 8.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate. Default: unet=5e-5, procanet=1e-4.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay. Default: 1e-4.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps. Default: 2.")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision (AMP). Default: True.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    parser.add_argument("--water-river", "--water_river", dest="water_river_as_flood", action="store_true", help="Include water/river mask as flood label.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers. Default: 2.")
    parser.add_argument("--device", default="auto", help="Device to use (e.g., cuda, cpu, auto).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: runs/final/{architecture}")
    parser.add_argument("--tile-root", type=Path, default=None, help="Directory containing tiles.")
    parser.add_argument("--base-channels", type=int, default=32, help="Base channels for the model. Default: 32.")
    parser.add_argument("--max-batches", type=int, default=None)

    args = parser.parse_args()

    # Set optimal defaults based on architecture if not specified
    if args.epochs is None:
        args.epochs = 21 if args.architecture == "unet" else 18
    if args.lr is None:
        args.lr = 5e-5 if args.architecture == "unet" else 1e-4

    return args


def build_model(architecture: str, base_channels: int) -> torch.nn.Module:
    if architecture == "unet":
        return UNet(in_channels=7, out_channels=1, base_channels=base_channels)
    if architecture == "procanet":
        return ProCANet(encoder1_channels=7, encoder2_channels=2, out_channels=1, base_channels=base_channels)
    raise ValueError(f"unknown architecture {architecture!r}")


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_amp_enabled(amp_requested: bool, device: torch.device) -> bool:
    return bool(amp_requested and device.type == "cuda")


def save_final_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    epoch: int,
    train_iou: float,
    architecture: str,
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_iou": train_iou,
        "architecture": architecture,
        "config": config,
    }
    torch.save(checkpoint, output_dir / "final.pt")
    print(f"✔ Final checkpoint saved to {output_dir / 'final.pt'}")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir or Path("runs") / "final" / args.architecture
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Final Training: {args.architecture} ===")
    print(f"  - Regions (10 CV): {CV_REGIONS}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Weight Decay: {args.weight_decay}")
    print(f"  - Batch Size (Effective): {args.batch_size} (x{args.gradient_accumulation_steps} = {args.batch_size * args.gradient_accumulation_steps})")
    print(f"  - Output Dir: {output_dir}")

    # Load dataset using all 10 CV regions
    train_dataset = FloodTileDataset(
        split="train",
        architecture=args.architecture,
        root=args.tile_root,
        water_river_as_flood=args.water_river_as_flood,
        regions=CV_REGIONS,
        augment=True,  # Enable augmentations for final training
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(args.architecture, args.base_channels).to(device)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("✔ Model compiled successfully using torch.compile")
        except Exception as e:
            print(f"Compilation skipped: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_effective = resolve_amp_enabled(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_effective)

    # Write training config
    config = vars(args).copy()
    config["device"] = str(device)
    config["output_dir"] = str(output_dir)
    config["tile_root"] = str(args.tile_root) if args.tile_root is not None else None
    config["optimizer"] = "AdamW"
    config["amp_effective"] = amp_effective
    config["regions"] = list(CV_REGIONS)
    write_training_config(output_dir, config)

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_iou",
                "train_dice",
                "train_accuracy",
                "lr",
            ],
        )
        writer.writeheader()

        last_metrics = None
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_batches=args.max_batches,
                progress_desc=f"{args.architecture} epoch {epoch}/{args.epochs} final_train",
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                amp_enabled=amp_effective,
                scaler=scaler,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "train_accuracy": train_metrics["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            f.flush()
            print(row)
            last_metrics = train_metrics

        # Save the final checkpoint
        if last_metrics is not None:
            save_final_checkpoint(
                model=model,
                optimizer=optimizer,
                output_dir=output_dir,
                epoch=args.epochs,
                train_iou=last_metrics["iou"],
                architecture=args.architecture,
                config=config,
            )


if __name__ == "__main__":
    main()
