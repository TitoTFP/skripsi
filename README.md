# Skripsi Flood Segmentation Dataset

Repository ini berisi pipeline preprocessing dataset banjir untuk deep learning
segmentation berbasis Sentinel-1, Sentinel-2, DEMNAS, dan label UNOSAT.

Target utama pipeline saat ini adalah dataset **7-channel U-Net style**:

```text
X = [VV, VH, Hue, Saturation, Value, Slope, HAND]
y = flood binary label
```

Label utama berasal dari UNOSAT, bukan pseudo-label otomatis.

## Dataset Status

Dataset siap dipakai untuk training model segmentation satu input 7-channel.

Hasil akhir:

```text
dataset/features_preprocessed/
dataset/tiles/7ch/
dataset/tiles/procanet/
dataset/preprocessing_summary.csv
dataset/feature_preprocessing_summary.csv
dataset/preprocessing_verification_report.csv
```

Ringkasan tile:

| Split | Tile |
|---|---:|
| Train | 983 |
| Validation | 36 |
| Test | 102 |
| Total | 1121 |

Semua 11 wilayah masuk dataset:

```text
Aceh_Besar
Aceh_Tamiang
Aceh_Timur
Aceh_Utara
Agam
Banda_Aceh
Bireuen
Langsa
Pasaman_Barat
Pidie
Pidie_Jaya
```

## Data Sources

### Sentinel-1 and Sentinel-2

Raw export dari Google Earth Engine:

```text
dataset/satelit raw/
```

Per wilayah tersedia:

- Sentinel-1: `VV`, `VH`
- Sentinel-2: `B2`, `B3`, `B4`, `B8`, `B11`, `B12`

S1 dan S2 per wilayah sudah align pada grid 10 m.

### DEMNAS

Raw DEMNAS:

```text
dataset/DEMNAS_Exports/
```

DEMNAS yang sudah di-warp ke grid Sentinel:

```text
dataset/DEMNAS_warped_to_sentinel/
```

Output DEMNAS aligned dipakai untuk membuat:

- `Slope`
- `HAND`

### UNOSAT

Raw UNOSAT FileGDB:

```text
dataset/unosat/FL20251126IDN.gdb
```

Label rasterized:

```text
dataset/labels_unosat_rasterized/
```

Per wilayah tersedia:

```text
label_flood_binary.tif
label_valid_mask.tif
label_water_river_mask.tif
```

Interpretasi:

| Raster | Value | Meaning |
|---|---:|---|
| `label_flood_binary.tif` | 1 | Flood, dari `FloodExtent_*` UNOSAT |
| `label_flood_binary.tif` | 0 | Non-flood atau outside flood polygon |
| `label_valid_mask.tif` | 1 | Area valid analisis UNOSAT |
| `label_valid_mask.tif` | 0 | Ignore saat training/evaluasi |
| `label_water_river_mask.tif` | 1 | WaterExtent/River auxiliary mask |

`WaterExtent_*` dan `ST2_20251129_River_AcehProvince` **tidak** masuk label flood `1`.
Keduanya hanya menjadi auxiliary mask.

## Preprocessing

Pipeline preprocessing sekarang:

1. Normalize Sentinel-1 `VV/VH`
2. Convert Sentinel-2 pseudo-RGB ke HSV
3. Generate S2 valid mask
4. Generate DEMNAS slope
5. Generate DEM-derived HAND
6. Stack 7 channel
7. Tile 512 x 512
8. Attach label + masks
9. Verify raster alignment, value range, mask values, tile shapes

### Sentinel-1

Input `VV/VH` dianggap sudah dB-like dari GEE export.

Preprocessing:

```text
clip [-30, 0]
normalize = (value + 30) / 30
```

Output:

```text
vv_norm.tif
vh_norm.tif
```

### Sentinel-2

Pseudo-RGB:

```text
R = B12
G = B8
B = B4
```

RGB di-clip ke `[0,1]`, lalu dikonversi ke HSV.

Output:

```text
hue.tif
saturation.tif
value.tif
s2_valid_mask.tif
```

Wilayah dengan S2 kosong/hampir kosong tetap masuk dataset.
HSV invalid diisi `0`, dan status validitasnya disimpan di `s2_valid_mask`.

Wilayah S2 bermasalah:

| Region | S2 valid |
|---|---:|
| Aceh_Tamiang | 0.0004% |
| Agam | 0.0000% |
| Langsa | 0.0000% |
| Pasaman_Barat | 0.0016% |

Ini sengaja dipertahankan agar semua wilayah tetap masuk dataset.
Risikonya: model bisa belajar pola `HSV=0` sebagai artefak data kosong.
Gunakan `s2_valid_mask` untuk audit atau ablation.

### DEMNAS

Slope dibuat dengan GDAL:

```text
gdaldem slope
clip [0,45]
normalize = slope / 45
```

HAND dibuat dengan WhiteboxTools dari DEM saja:

1. Breach depressions
2. D8 pointer
3. D8 flow accumulation
4. Extract streams dengan threshold `1000` cells
5. Elevation above stream
6. Clip `[0,50]`
7. Normalize `/50`

UNOSAT river mask tidak dipakai untuk HAND agar tidak terjadi label leakage.

Output:

```text
slope_norm.tif
hand_norm.tif
feature_valid_mask.tif
```

### Stack

Per wilayah, stack akhir:

```text
stack_7ch.tif
```

Band order:

| Band | Channel |
|---:|---|
| 1 | VV |
| 2 | VH |
| 3 | Hue |
| 4 | Saturation |
| 5 | Value |
| 6 | Slope |
| 7 | HAND |

Semua channel Float32 dan berada dalam range `[0,1]`.

## Tile Format

Tiles disimpan sebagai compressed NumPy `.npz`:

```text
dataset/tiles/7ch/{train,val,test}/*.npz
```

Setiap file berisi:

| Key | Shape | Meaning |
|---|---|---|
| `x` | `7 x 512 x 512` | Feature tensor |
| `y` | `1 x 512 x 512` | Flood binary label |
| `valid_mask` | `1 x 512 x 512` | UNOSAT valid analysis mask |
| `water_river_mask` | `1 x 512 x 512` | Auxiliary water/river mask |
| `feature_valid_mask` | `1 x 512 x 512` | S1 + DEM feature valid mask |
| `s2_valid_mask` | `1 x 512 x 512` | Sentinel-2 valid mask |
| `region` | scalar | Region name |
| `row` | scalar | Source raster row offset |
| `col` | scalar | Source raster column offset |
| `channels` | `7` | Channel names |

Training interpretation:

```text
valid_mask=1 and y=1 -> flood
valid_mask=1 and y=0 -> non-flood
valid_mask=0          -> ignore
```

Saat training/evaluasi, mask efektif adalah intersection:

```text
effective_valid_mask = valid_mask & feature_valid_mask
```

Artinya piksel di luar coverage UNOSAT atau memiliki feature no-data tidak ikut loss/metrik.

Default training tetap memakai `y` dari `label_flood_binary` saja.
`water_river_mask` bukan feature utama dan tidak otomatis menjadi label flood.

Untuk eksperimen sensitivitas, training CLI bisa menganggap water/river sebagai
flood positif tanpa mengubah tile:

```bash
uv run python -m scripts.train_segmentation --architecture unet --water-river
```

Saat flag `--water-river` atau `--water_river` aktif, target train/validation
menjadi:

```text
y_effective = y | water_river_mask
```

Mask valid tetap sama:

```text
effective_valid_mask = valid_mask & feature_valid_mask
```

## ProCANet Tile Format

ProCANet export tersedia di:

```text
dataset/tiles/procanet/{train,val,test}/*.npz
```

Export ini mengikuti insight paper ProCANet (`docs/referensi/2501.11923v1.pdf`):
encoder pertama membawa konteks multisensor lengkap, sedangkan encoder kedua
mengulang modalitas paling informatif untuk batas air. Pada paper, modalitas
air yang diulang adalah NIR. Untuk dataset ini, modalitas air paling stabil saat
banjir dan awan adalah SAR, sehingga encoder kedua memakai `VV/VH`.

Setiap file berisi:

| Key | Shape | Meaning |
|---|---|---|
| `x_encoder1` | `7 x 512 x 512` | `VV`, `VH`, `Hue`, `Saturation`, `Value`, `Slope`, `HAND` |
| `x_encoder2` | `2 x 512 x 512` | `VV`, `VH` SAR-focused branch |
| `y` | `1 x 512 x 512` | Flood binary proxy label dari UNOSAT |
| `valid_mask` | `1 x 512 x 512` | UNOSAT valid analysis mask |
| `water_river_mask` | `1 x 512 x 512` | Auxiliary water/river mask |
| `feature_valid_mask` | `1 x 512 x 512` | S1 + DEM feature valid mask |
| `s2_valid_mask` | `1 x 512 x 512` | Sentinel-2 valid mask |
| `encoder1_channels` | `7` | Encoder 1 channel names |
| `encoder2_channels` | `2` | Encoder 2 channel names |
| `region` | scalar | Region name |
| `row` | scalar | Source raster row offset |
| `col` | scalar | Source raster column offset |

Tile counts sama dengan export 7-channel:

| Split | Tile |
|---|---:|
| Train | 983 |
| Validation | 36 |
| Test | 102 |
| Total | 1121 |

## Split Policy

Split berbasis wilayah untuk mengurangi spatial leakage:

| Split | Regions |
|---|---|
| Train | Aceh_Besar, Aceh_Tamiang, Aceh_Timur, Aceh_Utara, Agam, Bireuen, Banda_Aceh, Langsa, Pasaman_Barat |
| Validation | Pidie_Jaya |
| Test | Pidie |

Tile size:

```text
512 x 512
```

Tile positive selalu dipertahankan.
Background-only tile disampling deterministik agar kelas tidak terlalu timpang.

## Summary Statistics

Tile summary:

| Region | Split | Tiles | Positive | Background | Flood px | Valid px | S2 valid px |
|---|---|---:|---:|---:|---:|---:|---:|
| Aceh_Besar | train | 128 | 64 | 64 | 628789 | 33341952 | 21347199 |
| Aceh_Tamiang | train | 116 | 83 | 33 | 1223088 | 29644288 | 168 |
| Aceh_Timur | train | 267 | 129 | 138 | 4599286 | 68784000 | 27833115 |
| Aceh_Utara | train | 122 | 80 | 42 | 4194022 | 31562240 | 19914945 |
| Agam | train | 96 | 36 | 60 | 359611 | 25003008 | 0 |
| Banda_Aceh | train | 7 | 7 | 0 | 23904 | 709340 | 277784 |
| Bireuen | train | 73 | 40 | 33 | 546763 | 19130368 | 14371660 |
| Langsa | train | 13 | 13 | 0 | 264247 | 2893273 | 0 |
| Pasaman_Barat | train | 161 | 72 | 89 | 385168 | 41810944 | 1332 |
| Pidie_Jaya | val | 36 | 18 | 18 | 325779 | 9154560 | 6873287 |
| Pidie | test | 102 | 51 | 51 | 1052181 | 26336256 | 18592985 |

Full CSV:

```text
dataset/preprocessing_summary.csv
dataset/feature_preprocessing_summary.csv
dataset/preprocessing_verification_report.csv
```

## Scripts

### Feature preprocessing

```bash
uv run python -m scripts.preprocess_features
```

Creates:

```text
dataset/features_preprocessed/
dataset/feature_preprocessing_summary.csv
```

### Tile generation

```bash
uv run python -m scripts.make_tiles
```

Creates:

```text
dataset/tiles/7ch/
dataset/preprocessing_summary.csv
```

### ProCANet tile generation

```bash
uv run python -m scripts.make_procanet_tiles
```

Creates:

```text
dataset/tiles/procanet/
```

Expected output:

```text
procanet_tile_counts {'train': 983, 'val': 36, 'test': 102}
```

### Verification

```bash
uv run python -m scripts.verify_preprocessing
```

Expected output:

```text
feature_regions 11
tile_counts {'train': 983, 'val': 36, 'test': 102}
```

### Unit tests

```bash
uv run python -m unittest tests.test_preprocessing_helpers tests.test_training_data tests.test_models tests.test_training_loop -v
```

Expected:

```text
Ran 30 tests
OK
```

### Training

Baseline U-Net:

```bash
uv run python -m scripts.train_segmentation \
  --architecture unet \
  --epochs 50 \
  --batch-size 8 \
  --output-dir runs/baseline_unet
```

ProCANet:

```bash
uv run python -m scripts.train_segmentation \
  --architecture procanet \
  --epochs 50 \
  --batch-size 4 \
  --output-dir runs/procanet
```

Default training uses AdamW, `25` epochs, batch size `2`, learning rate `1e-4`,
weight decay `1e-4`, early stopping patience `5`, `ReduceLROnPlateau` on
validation IoU, gradient accumulation step `1`, AMP disabled, and auto-selects
CUDA when available. Best checkpoints are saved by validation IoU:

```text
runs/{architecture}/best.pt
```

The current finished experiment outputs are:

```text
runs/baseline_unet/best.pt
runs/procanet/best.pt
```

Per-epoch metrics are written to:

```text
runs/{architecture}/metrics.csv
```

Metrics columns:

```text
epoch, train_loss, train_iou, train_dice, val_loss, val_iou, val_dice,
lr, best_val_iou, saved, bad_epochs, stopped_early
```

Existing finished run CSV files under `runs/baseline_unet/` and `runs/procanet/`
were generated before the `lr` column was added, so those files keep the older
schema without `lr`.

The resolved training config is written to:

```text
runs/{architecture}/config.json
```

Useful overrides:

```bash
uv run python -m scripts.train_segmentation \
  --architecture unet \
  --epochs 50 \
  --batch-size 4 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --lr-scheduler reduce-on-plateau \
  --lr-factor 0.5 \
  --lr-patience 2 \
  --gradient-accumulation-steps 2 \
  --amp \
  --early-stopping-patience 8 \
  --early-stopping-min-delta 0.001
```

Water/river-as-flood experiment:

```bash
uv run python -m scripts.train_segmentation \
  --architecture unet \
  --water-river \
  --amp \
  --gradient-accumulation-steps 2
```

`--amp` only takes effect on CUDA. On CPU, training falls back to FP32 and
`runs/{architecture}/config.json` records `amp_effective: false`.

### Current Training Results

Finished training runs:

| Model | Output dir | Epoch stopped | Best validation IoU | Best validation Dice |
|---|---|---:|---:|---:|
| U-Net baseline | `runs/baseline_unet/` | 10 | 0.6062394985 | 0.7548556726 |
| ProCANet | `runs/procanet/` | 17 | 0.6224360219 | 0.7672857524 |

The current comparison plot is stored at:

```text
runs/training_curves.png
```

These are validation results only. Final test-set inference, confusion matrix,
IoU, Dice/F1, and qualitative overlay visualizations are still pending.

### Mask Visualization

Inspect a tile and its label/masks:

```bash
uv run python -m scripts.visualize_valid_masks \
  dataset/tiles/7ch/train/<tile>.npz \
  --output runs/<tile>_masks.png
```

## Environment

Project dependencies are listed in:

```text
pyproject.toml
uv.lock
```

Important packages:

```text
numpy
torch
whitebox
earthengine-api
requests
```

Runtime uses `uv` with Python 3.14. GDAL Python bindings come from the system
Python environment in this workspace, so the local uv environment must keep
system site packages visible.

```text
uv run python
```

The project pins Python 3.14 in `.python-version` and `pyproject.toml`.

If recreating locally:

```bash
uv venv --python /usr/bin/python3.14 --system-site-packages .venv
uv sync
```

## Repository Layout

```text
.
├── dataset/
│   ├── preprocess_steps.md
│   ├── preprocessing_todo.md
│   ├── labels_unosat_rasterized/
│   ├── DEMNAS_warped_to_sentinel/
│   ├── features_preprocessed/
│   └── tiles/
│       ├── 7ch/
│       └── procanet/
├── scripts/
│   ├── preprocessing_utils.py
│   ├── preprocess_features.py
│   ├── make_tiles.py
│   ├── make_procanet_tiles.py
│   ├── train_segmentation.py
│   └── verify_preprocessing.py
├── training/
│   ├── datasets.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   └── models/
├── tests/
│   ├── test_preprocessing_helpers.py
│   ├── test_training_data.py
│   ├── test_models.py
│   └── test_training_loop.py
├── pyproject.toml
└── README.md
```

Large raw and generated geospatial artifacts are ignored by `.gitignore`.

## Jupyter MCP

Project dependencies include JupyterLab plus the collaboration and MCP helper
packages required by the Datalayer Jupyter MCP STDIO provider.

Start JupyterLab from this repo before using the MCP server:

```bash
uv run jupyter lab --port 8888 --IdentityProvider.token skripsi-mcp --ip 127.0.0.1
```

Codex is configured in `/home/nozomi/.codex/config.toml` with:

```toml
[mcp_servers.jupyter]
command = "uvx"
args = ["jupyter-mcp-server@latest"]
env = { JUPYTER_URL = "http://localhost:8888", JUPYTER_TOKEN = "skripsi-mcp", ALLOW_IMG_OUTPUT = "true" }
```

If the MCP client starts before JupyterLab, restart the client after JupyterLab
is reachable, or reconnect with the server's `connect_to_jupyter` tool.

## Caveats

- Current tile dataset is ready for U-Net style single-input 7-channel models.
- ProCANet two-encoder export is ready under `dataset/tiles/procanet/`.
- Baseline U-Net and ProCANet implementations output binary segmentation logits.
- Training checkpoints are selected by validation IoU, not validation loss.
- ProCANet encoder design intentionally repeats SAR (`VV/VH`) in encoder 2 because
  SAR is the most reliable water/flood modality under cloud and rain conditions.
- S2 invalid regions are intentionally kept with HSV zeroed.
- `s2_valid_mask` should be used for audit, ablation, or sensitivity checks.
- Loss/evaluation use `label_valid_mask` intersected with `feature_valid_mask`
  to ignore pixels outside UNOSAT analysis coverage or feature coverage.
- `water_river_mask` is auxiliary by default; pass `--water-river`/`--water_river`
  to union it into the train/validation flood target for sensitivity experiments.

## Minimal PyTorch Loading Example

```python
from torch.utils.data import DataLoader

from training.datasets import FloodTileDataset
from training.losses import masked_bce_with_logits
from training.metrics import masked_iou

dataset = FloodTileDataset("train", architecture="unet")
loader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))

x = batch["features"]              # 2 x 7 x 512 x 512
y = batch["y"]                     # 2 x 1 x 512 x 512
label_valid_mask = batch["valid_mask"]                                 # 2 x 1 x 512 x 512
feature_valid_mask = batch["auxiliary_masks"]["feature_valid_mask"]     # 2 x 1 x 512 x 512
valid_mask = label_valid_mask & feature_valid_mask

# logits = model(x)
# loss = masked_bce_with_logits(logits, y, valid_mask)
# iou = masked_iou(logits, y, valid_mask)
```

Water/river-as-flood loader experiment:

```python
dataset = FloodTileDataset("train", architecture="unet", water_river_as_flood=True)
```

ProCANet loader:

```python
dataset = FloodTileDataset("train", architecture="procanet")
batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=True)))

x1 = batch["features"]["encoder1"]  # 2 x 7 x 512 x 512
x2 = batch["features"]["encoder2"]  # 2 x 2 x 512 x 512
```
