from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.datasets import FloodTileDataset
from training.models import ProCANet, UNet
from training.train import EarlyStopping, evaluate, save_checkpoint_if_best, train_one_epoch, write_training_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flood segmentation models.")
    parser.add_argument("--architecture", choices=("unet", "procanet"), required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("reduce-on-plateau", "none"), default="reduce-on-plateau")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--water-river", "--water_river", dest="water_river_as_flood", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tile-root", type=Path, default=None)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = args.output_dir or Path("runs") / args.architecture
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = FloodTileDataset(
        "train",
        architecture=args.architecture,
        root=args.tile_root,
        water_river_as_flood=args.water_river_as_flood,
    )
    val_dataset = FloodTileDataset(
        "val",
        architecture=args.architecture,
        root=args.tile_root,
        augment=False,
        water_river_as_flood=args.water_river_as_flood,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(args.architecture, args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.lr_scheduler, args.lr_factor, args.lr_patience)
    amp_effective = resolve_amp_enabled(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_effective)
    config = vars(args).copy()
    config["device"] = str(device)
    config["output_dir"] = str(output_dir)
    config["tile_root"] = str(args.tile_root) if args.tile_root is not None else None
    config["optimizer"] = "AdamW"
    config["amp_effective"] = amp_effective
    write_training_config(output_dir, config)

    metrics_path = output_dir / "metrics.csv"
    best_val_iou = -1.0
    early_stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_iou",
                "train_dice",
                "val_loss",
                "val_iou",
                "val_dice",
                "lr",
                "best_val_iou",
                "saved",
                "bad_epochs",
                "stopped_early",
            ],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                max_batches=args.max_batches,
                progress_desc=f"{args.architecture} epoch {epoch}/{args.epochs} train",
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                amp_enabled=amp_effective,
                scaler=scaler,
            )
            val_metrics = evaluate(
                model,
                val_loader,
                device,
                max_batches=args.max_batches,
                progress_desc=f"{args.architecture} epoch {epoch}/{args.epochs} val",
            )
            if scheduler is not None:
                scheduler.step(val_metrics["iou"])
            best_val_iou, saved = save_checkpoint_if_best(
                model,
                optimizer,
                output_dir,
                epoch=epoch,
                val_iou=val_metrics["iou"],
                best_val_iou=best_val_iou,
                architecture=args.architecture,
                config=config,
            )
            stopped_early = early_stopping.step(val_metrics["iou"])
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "val_loss": val_metrics["loss"],
                "val_iou": val_metrics["iou"],
                "val_dice": val_metrics["dice"],
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_iou": best_val_iou,
                "saved": int(saved),
                "bad_epochs": early_stopping.bad_epochs,
                "stopped_early": int(stopped_early),
            }
            writer.writerow(row)
            f.flush()
            print(row)
            if stopped_early:
                break


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


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    factor: float,
    patience: int,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "reduce-on-plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
        )
    raise ValueError(f"unknown lr scheduler {scheduler_name!r}")


if __name__ == "__main__":
    main()
