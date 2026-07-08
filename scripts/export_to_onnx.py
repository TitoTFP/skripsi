from __future__ import annotations

import os
import sys

# Add the root directory to path to resolve 'training' imports
sys.path.append(os.getcwd())

import torch
from torch import nn

from training.models import ProCANet, UNet


# Wrapper to bypass dict-input during ONNX export for ProCANet
class ProCANetExportWrapper(nn.Module):
    def __init__(self, model: ProCANet) -> None:
        super().__init__()
        self.model = model

        # Ensure the underlying model is also set to evaluation mode
        self.model.eval()

    def forward(self, encoder1: torch.Tensor, encoder2: torch.Tensor) -> torch.Tensor:
        features = {
            "encoder1": encoder1,
            "encoder2": encoder2
        }
        return self.model(features)


def main() -> None:
    # 1. Create output directory in artifacts
    output_dir = "artifacts/models"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------
    # U-Net Export
    # ----------------------------------------------------
    print("Mengekspor model U-Net ke ONNX...")
    unet_model = UNet(in_channels=7, out_channels=1, base_channels=32)
    unet_model.eval()
    
    # Input shape: (Batch, Channels, Height, Width) -> (1, 7, 256, 256)
    dummy_input_unet = torch.randn(1, 7, 256, 256)
    unet_onnx_path = os.path.join(output_dir, "unet_architecture.onnx")
    
    # Export without parameter weights so we get a single, lightweight .onnx file
    torch.onnx.export(
        unet_model,
        dummy_input_unet,
        unet_onnx_path,
        export_params=False,
        opset_version=18,
        input_names=['input_features'],
        output_names=['output_logits']
    )
    print(f"✔ U-Net model exported to {unet_onnx_path}")

    # ----------------------------------------------------
    # ProCANet Export
    # ----------------------------------------------------
    print("\nMengekspor model ProCANet ke ONNX...")
    procanet_model = ProCANet(encoder1_channels=7, encoder2_channels=2, out_channels=1, base_channels=32)
    procanet_wrapper = ProCANetExportWrapper(procanet_model)
    procanet_wrapper.eval()
    
    # Inputs:
    # - Encoder 1 (7 channels): Sentinel-1 + Sentinel-2 HSV + DEMNAS
    # - Encoder 2 (2 channels): Sentinel-1 VV & VH
    dummy_enc1 = torch.randn(1, 7, 256, 256)
    dummy_enc2 = torch.randn(1, 2, 256, 256)
    procanet_onnx_path = os.path.join(output_dir, "procanet_architecture.onnx")
    
    # Export without parameter weights so we get a single, lightweight .onnx file
    torch.onnx.export(
        procanet_wrapper,
        (dummy_enc1, dummy_enc2),
        procanet_onnx_path,
        export_params=False,
        opset_version=18,
        input_names=['encoder1_input', 'encoder2_input'],
        output_names=['output_logits']
    )
    print(f"✔ ProCANet model exported to {procanet_onnx_path}")

    # Clean up any leftover external data files if they exist
    for fn in ["unet_architecture.onnx.data", "procanet_architecture.onnx.data"]:
        path = os.path.join(output_dir, fn)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed temporary data file: {path}")


if __name__ == "__main__":
    main()
