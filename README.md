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

`water_river_mask` bukan feature utama dan bukan label flood.
Gunakan hanya sebagai control mask/exclusion mask bila perlu.

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
.venv314/bin/python -m scripts.preprocess_features
```

Creates:

```text
dataset/features_preprocessed/
dataset/feature_preprocessing_summary.csv
```

### Tile generation

```bash
.venv314/bin/python -m scripts.make_tiles
```

Creates:

```text
dataset/tiles/7ch/
dataset/preprocessing_summary.csv
```

### ProCANet tile generation

```bash
.venv314/bin/python -m scripts.make_procanet_tiles
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
.venv314/bin/python -m scripts.verify_preprocessing
```

Expected output:

```text
feature_regions 11
tile_counts {'train': 983, 'val': 36, 'test': 102}
```

### Unit tests

```bash
.venv314/bin/python -m unittest tests.test_preprocessing_helpers tests.test_training_data tests.test_models tests.test_training_loop -v
```

Expected:

```text
Ran 20 tests
OK
```

### Training

Baseline U-Net:

```bash
.venv314/bin/python -m scripts.train_segmentation --architecture unet
```

ProCANet:

```bash
.venv314/bin/python -m scripts.train_segmentation --architecture procanet
```

Default training uses AdamW, `25` epochs, batch size `2`, learning rate `1e-4`,
weight decay `1e-4`, and auto-selects CUDA when available. Best checkpoints are
saved by validation IoU:

```text
runs/unet/best.pt
runs/procanet/best.pt
```

Per-epoch metrics are written to:

```text
runs/{architecture}/metrics.csv
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

GDAL Python bindings come from the system Python environment in this workspace.
The working runtime used for preprocessing is:

```text
.venv314/bin/python
```

This venv was created with system site packages so it can access system GDAL and
NumPy while also installing `whitebox`.

If recreating locally:

```bash
uv venv --python /usr/bin/python --system-site-packages .venv314
uv pip install --python .venv314/bin/python whitebox
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

## Caveats

- Current tile dataset is ready for U-Net style single-input 7-channel models.
- ProCANet two-encoder export is ready under `dataset/tiles/procanet/`.
- Baseline U-Net and ProCANet implementations output binary segmentation logits.
- Training checkpoints are selected by validation IoU, not validation loss.
- ProCANet encoder design intentionally repeats SAR (`VV/VH`) in encoder 2 because
  SAR is the most reliable water/flood modality under cloud and rain conditions.
- S2 invalid regions are intentionally kept with HSV zeroed.
- `s2_valid_mask` should be used for audit, ablation, or sensitivity checks.
- `label_valid_mask` must be used during loss/evaluation to ignore pixels outside
  UNOSAT analysis coverage.
- `water_river_mask` is auxiliary only; it is not flood target and not model input.

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
valid_mask = batch["valid_mask"]   # 2 x 1 x 512 x 512

# logits = model(x)
# loss = masked_bce_with_logits(logits, y, valid_mask)
# iou = masked_iou(logits, y, valid_mask)
```

ProCANet loader:

```python
dataset = FloodTileDataset("train", architecture="procanet")
batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=True)))

x1 = batch["features"]["encoder1"]  # 2 x 7 x 512 x 512
x2 = batch["features"]["encoder2"]  # 2 x 2 x 512 x 512
```
