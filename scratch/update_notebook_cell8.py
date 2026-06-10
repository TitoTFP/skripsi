import json
from pathlib import Path

notebook_path = Path("notebooks/hyperparameter_tuning_analysis.ipynb")

# Load the notebook
with notebook_path.open("r", encoding="utf-8") as f:
    nb = json.load(f)

modified = False
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        # Find cell 8 (get_average_learning_curves definition)
        if any("def get_average_learning_curves" in line for line in source):
            # Let's find where mean_val_dice is defined and insert mean_lr after it
            for i, line in enumerate(source):
                if 'mean_val_dice' in line and 'mean_val_dice' in line:
                    # Check if mean_lr is already there
                    if not any("mean_lr" in l for l in source):
                        # Replace the line to add a comma at the end before newline
                        source[i] = line.replace('mean_val_dice": df_epoch["val_dice"].mean()\n', 'mean_val_dice": df_epoch["val_dice"].mean(),\n')
                        source.insert(i + 1, '            "mean_lr": df_epoch["lr"].mean()\n')
                        modified = True
                        print("Inserted mean_lr into get_average_learning_curves cell.")
                    break

if modified:
    # Save the modified notebook
    with notebook_path.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print("Notebook cell 8 updated successfully.")
else:
    print("Cell 8 not found or already modified.")
