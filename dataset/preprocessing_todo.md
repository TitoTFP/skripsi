# Preprocessing Todo

Pipeline ini memakai **UNOSAT sebagai label utama**, menggantikan pseudo-label otomatis pada `preprocess_steps.md`.

## Target Akhir

Input model segmentation:

```text
X = [VV, VH, Hue, Saturation, Value, Slope, HAND]
```

Output label:

```text
y = label_flood_binary
valid = label_valid_mask
water_river = label_water_river_mask
```

Interpretasi training:

```text
valid=1, y=1 -> flood
valid=1, y=0 -> non-flood
valid=0      -> ignore
water_river=1 -> optional exclusion/control mask, not main target
```

## Sudah Dilakukan

- [x] Sentinel-1 dan Sentinel-2 raw dari GEE tersedia di `dataset/satelit raw/`.
- [x] Tiap wilayah punya S1 dan S2 yang sudah align satu sama lain.
- [x] S1 berisi 2 band: `VV`, `VH`.
- [x] S2 berisi 6 band: `B2`, `B3`, `B4`, `B8`, `B11`, `B12`.
- [x] DEMNAS raw tersedia di `dataset/DEMNAS_Exports/`.
- [x] DEMNAS sudah di-warp ke grid Sentinel per wilayah.
- [x] DEMNAS aligned tersedia di `dataset/DEMNAS_warped_to_sentinel/`.
- [x] UNOSAT FileGDB tersedia di `dataset/unosat/FL20251126IDN.gdb`.
- [x] UNOSAT flood label sudah dirasterize ke grid Sentinel.
- [x] UNOSAT valid analysis extent sudah dirasterize ke grid Sentinel.
- [x] UNOSAT WaterExtent + River sudah digabung dan dirasterize sebagai mask tambahan.
- [x] Geometri UNOSAT invalid/kompleks diperbaiki dengan `MakeValid()` sebelum rasterisasi.
- [x] Rasterisasi UNOSAT dilakukan per-layer, lalu mask hasil raster digabung dengan OR/max.
- [x] `label_valid_mask.tif` dipotong dengan ROI wilayah administratif.
- [x] Label raster tersedia di `dataset/labels_unosat_rasterized/`.
- [x] Semua label raster sudah diverifikasi align dengan S1.
- [x] Semua label raster hanya berisi nilai `0/1`.

## Belum Dilakukan

- [x] Hitung fitur `Slope` dari DEMNAS aligned.
- [x] Hitung fitur `HAND` dari DEMNAS aligned.
- [x] Normalisasi `Slope` ke `[0,1]`.
- [x] Normalisasi `HAND` ke `[0,1]`.
- [x] Konversi Sentinel-1 `VV` dan `VH` ke skala final model.
- [x] Clip Sentinel-1 ke rentang target, misalnya `-30 dB` sampai `0 dB`.
- [x] Normalisasi Sentinel-1 ke `[0,1]`.
- [x] Buat pseudo-RGB Sentinel-2:
  - `R = B12`
  - `G = B8`
  - `B = B4`
- [x] Transform pseudo-RGB Sentinel-2 menjadi HSV.
- [x] Normalisasi `Hue`, `Saturation`, `Value` ke `[0,1]`.
- [x] Tangani wilayah dengan S2 valid sangat rendah/kosong:
  - Aceh Tamiang
  - Agam
  - Kota Langsa
  - Pasaman Barat
- [x] Stack feature 7 channel per wilayah.
- [x] Terapkan NoData/valid feature mask agar pixel rusak tidak masuk training.
- [x] Potong raster menjadi tile, misalnya `512 x 512` atau `256 x 256`.
- [x] Filter tile dengan terlalu banyak NoData.
- [x] Filter/atur jumlah tile background-only agar kelas tidak terlalu imbalance.
- [x] Split train/validation/test secara spasial atau per wilayah.
- [x] Simpan dataset siap training untuk U-Net.
- [x] Simpan dataset siap training untuk ProCANet dua encoder.
  - Encoder 1: `VV`, `VH`, `Hue`, `Saturation`, `Value`, `Slope`, `HAND`
  - Encoder 2: `VV`, `VH`
  - Output: `dataset/tiles/procanet/`
- [x] Buat ringkasan statistik final tile:
  - jumlah tile
  - jumlah flood pixel
  - jumlah non-flood pixel
  - rasio kelas
  - coverage per wilayah

## Output Saat Ini

```text
dataset/DEMNAS_warped_to_sentinel/
dataset/labels_unosat_rasterized/
dataset/features_preprocessed/
dataset/tiles/7ch/
dataset/tiles/procanet/
dataset/preprocessing_summary.csv
dataset/feature_preprocessing_summary.csv
dataset/preprocessing_verification_report.csv
```

Setiap folder wilayah di `dataset/labels_unosat_rasterized/` berisi:

```text
label_flood_binary.tif
label_valid_mask.tif
label_water_river_mask.tif
```

## Keputusan Label

- `label_flood_binary.tif` hanya memakai `FloodExtent_*` dari UNOSAT.
- `FloodExtent_*` dirasterize per-layer setelah `MakeValid()`, lalu digabung sebagai flood binary mask.
- `label_valid_mask.tif` adalah `AnalysisExtent_*` yang sudah dipotong ROI wilayah administratif dari `dataset/batas admin indo/`.
- `WaterExtent_*` dan `ST2_20251129_River_AcehProvince` tidak dimasukkan sebagai flood `1`.
- `WaterExtent_*` dan `River` dipakai sebagai `label_water_river_mask.tif`.
- Pixel luar `label_valid_mask.tif` wajib di-ignore saat training/evaluasi.

## Catatan Risiko

- Beberapa S2 wilayah kosong atau hampir kosong, sehingga HSV tidak reliable untuk wilayah tersebut.
- UNOSAT flood label banyak overlap dengan water/river mask; ini normal karena WaterExtent mencakup air terdeteksi.
- Jika target berubah menjadi "semua genangan/air", definisi label perlu direvisi. Untuk target "banjir", definisi sekarang tetap memakai `FloodExtent_*`.
