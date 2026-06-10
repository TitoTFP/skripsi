import json
from pathlib import Path

notebook_path = Path("notebooks/hyperparameter_tuning_analysis.ipynb")

# Load the notebook
with notebook_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the new comparative code for cell 9
new_code = [
    "if not df_results.empty and not best_unet_curve.empty and not best_procanet_curve.empty:\n",
    "    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)\n",
    "    \n",
    "    metrics = {\n",
    "        \"U-Net (lr=5e-5, wd=1e-4)\": best_unet_curve,\n",
    "        \"ProCANet (lr=1e-4, wd=1e-4)\": best_procanet_curve\n",
    "    }\n",
    "    \n",
    "    specs = [\n",
    "        (axes[0, 0], \"Loss (BCE + Dice)\", \"mean_train_loss\", \"mean_val_loss\"),\n",
    "        (axes[0, 1], \"Mean IoU\", \"mean_train_iou\", \"mean_val_iou\"),\n",
    "        (axes[1, 0], \"F1-Score / Dice\", \"mean_train_dice\", \"mean_val_dice\"),\n",
    "    ]\n",
    "    \n",
    "    for ax, title, train_key, val_key in specs:\n",
    "        for name, df_curve in metrics.items():\n",
    "            epochs = df_curve[\"epoch\"]\n",
    "            ax.plot(epochs, df_curve[train_key], linestyle=\"--\", label=f\"{name} train\")\n",
    "            ax.plot(epochs, df_curve[val_key], label=f\"{name} val\")\n",
    "        ax.set_title(title, fontsize=14, pad=10)\n",
    "        ax.set_xlabel(\"Epoch\", fontsize=12)\n",
    "        ax.grid(True, alpha=0.3)\n",
    "        ax.legend(fontsize=10)\n",
    "\n",
    "    lr_ax = axes[1, 1]\n",
    "    for name, df_curve in metrics.items():\n",
    "        if \"mean_lr\" in df_curve.columns:\n",
    "            lr_ax.plot(df_curve[\"epoch\"], df_curve[\"mean_lr\"], label=name, linewidth=2)\n",
    "    lr_ax.set_title(\"Learning Rate\", fontsize=14, pad=10)\n",
    "    lr_ax.set_xlabel(\"Epoch\", fontsize=12)\n",
    "    lr_ax.grid(True, alpha=0.3)\n",
    "    lr_ax.legend(fontsize=10)\n",
    "\n",
    "    fig.savefig(\"../runs/training_curves.png\", dpi=160)\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Data kurva kosong atau eksperimen belum dimuat.\")"
]

modified = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("plt.subplots(2, 2, figsize=(16, 11))" in line for line in source) or any("fig.savefig" in line and "training_curves.png" in line for line in source):
            cell["source"] = new_code
            modified = True
            print("Replaced cell 9 source code with Option B comparative layout.")
            break

if modified:
    # Save the modified notebook
    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Notebook updated successfully.")
else:
    print("Plotting cell was not found or modified.")
