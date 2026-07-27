# Plan: Analisis Sensitivitas Modalitas dengan Modality Masking

## Tujuan

Menambahkan eksperimen evaluasi untuk mengetahui perubahan kinerja model ketika hanya satu kelompok modalitas input yang dipertahankan.

Eksperimen **tidak melakukan training ulang**. Checkpoint U-Net dan ProCANet yang sudah ada tetap digunakan. Seluruh model tetap menerima tensor dengan bentuk dan jumlah channel yang sama seperti saat training, sedangkan channel yang tidak digunakan ditetapkan bernilai nol pada tahap inference.

Skenario input yang dievaluasi:

- `all`: seluruh 7 channel dipertahankan
- `sentinel1`: hanya `VV` dan `VH` dipertahankan
- `sentinel2`: hanya `Hue`, `Saturation`, dan `Value` dipertahankan
- `demnas`: hanya `Slope` dan `HAND` dipertahankan

Urutan channel kanonis:

```text
0: VV
1: VH
2: Hue
3: Saturation
4: Value
5: Slope
6: HAND
```

## Interpretasi Metodologis

Eksperimen ini merupakan **modality masking**, **input occlusion analysis**, atau **analisis sensitivitas terhadap modalitas input**.

Eksperimen ini tidak boleh disebut sebagai pelatihan model unimodal atau performa optimal masing-masing modalitas karena:

- model dilatih menggunakan konfigurasi input penuh;
- bobot model tidak diubah;
- channel yang tidak digunakan hanya dimasking menjadi nol saat inference.

Kesimpulan yang diperbolehkan:

> Hasil menunjukkan sensitivitas model integrasi terhadap ketersediaan masing-masing kelompok modalitas input.

Kesimpulan yang tidak diperbolehkan:

> Hasil menunjukkan performa model yang dilatih hanya menggunakan Sentinel-1, Sentinel-2, atau DEMNAS.

## Prinsip Implementasi

1. Jangan mengubah pipeline training.
2. Jangan melatih ulang checkpoint.
3. Jangan mengubah jumlah input channel model.
4. Jangan mengubah arsitektur U-Net maupun ProCANet.
5. Jangan membuat dataset baru per modalitas.
6. Lakukan masking setelah tensor 7-channel selesai dimuat dan sebelum tensor masuk ke model.
7. Gunakan checkpoint, wilayah uji, tile, threshold, valid mask, mosaicking, dan metrik yang sama dengan evaluasi utama.
8. Skenario `all` harus menghasilkan output yang identik dengan inference lama.

## Perubahan Kode

### 1. Utilitas modality masking

Tambahkan fungsi terpusat, misalnya pada modul utilitas inference atau file baru seperti:

```text
evaluation/modality_masking.py
```

Fungsi minimal:

```python
apply_modality_mask(
    x: torch.Tensor,
    modality: str,
) -> torch.Tensor
```

Ketentuan:

- menerima tensor berbentuk `[B, 7, H, W]`;
- tidak memodifikasi tensor asli secara in-place;
- `all` mengembalikan seluruh channel tanpa perubahan nilai;
- `sentinel1` mempertahankan indeks `[0, 1]`;
- `sentinel2` mempertahankan indeks `[2, 3, 4]`;
- `demnas` mempertahankan indeks `[5, 6]`;
- seluruh channel lain diisi nol;
- menolak nama modalitas yang tidak valid;
- mempertahankan `dtype`, `device`, dan shape tensor.

Contoh hasil masking:

```text
all:
[VV, VH, H, S, V, Slope, HAND]

sentinel1:
[VV, VH, 0, 0, 0, 0, 0]

sentinel2:
[0, 0, H, S, V, 0, 0]

demnas:
[0, 0, 0, 0, 0, Slope, HAND]
```

### 2. Inference U-Net

Perbarui `scripts/infer_segmentation.py` atau entry point inference yang benar setelah memeriksa struktur repository.

Tambahkan opsi CLI:

```text
--input-scenario {all,sentinel1,sentinel2,demnas}
```

Default:

```text
all
```

Alur:

1. Muat tile 7-channel seperti biasa.
2. Terapkan preprocessing yang sudah ada.
3. Terapkan `apply_modality_mask(...)`.
4. Masukkan tensor hasil masking ke checkpoint U-Net yang sama.
5. Jangan membangun U-Net dengan jumlah channel berbeda.
6. Jangan membaca metadata `input_modality` dari checkpoint karena checkpoint lama merupakan model input penuh.
7. Simpan nama skenario pada metadata hasil inference.

### 3. Inference ProCANet

ProCANet tetap menggunakan struktur input dan arsitektur yang sama seperti checkpoint asli.

Jika implementasi ProCANet menerima tensor 7-channel lalu mengambil `VV` dan `VH` secara internal untuk encoder kedua:

- cukup lakukan masking pada tensor 7-channel sebelum pemanggilan model;
- pada skenario `sentinel2` dan `demnas`, channel `VV` dan `VH` sudah nol sehingga encoder kedua otomatis menerima input nol.

Jika pipeline inference membentuk input encoder pertama dan encoder kedua secara terpisah:

- `all`
  - encoder pertama: seluruh 7 channel asli;
  - encoder kedua: `VV`, `VH` asli.
- `sentinel1`
  - encoder pertama: `VV`, `VH`, channel lain nol;
  - encoder kedua: `VV`, `VH` asli.
- `sentinel2`
  - encoder pertama: hanya `H`, `S`, `V`;
  - encoder kedua: tensor nol dengan shape yang sama seperti input `VV`, `VH`.
- `demnas`
  - encoder pertama: hanya `Slope`, `HAND`;
  - encoder kedua: tensor nol dengan shape yang sama seperti input `VV`, `VH`.

Jangan mengubah jumlah channel encoder, mekanisme cross-attention, atau bobot model.

### 4. Batch evaluation seluruh skenario

Tambahkan mode atau script evaluasi batch, misalnya:

```text
scripts/evaluate_modality_masking.py
```

Script harus:

1. menerima path checkpoint U-Net dan ProCANet;
2. menjalankan empat skenario input untuk masing-masing model;
3. menggunakan wilayah uji independen Aceh Utara yang sama dengan evaluasi utama;
4. menggunakan threshold klasifikasi yang sama, yaitu nilai yang sudah dipakai pipeline utama;
5. menggunakan valid mask yang sama;
6. menjalankan mosaicking dengan prosedur yang sama;
7. menghitung metrik piksel dari hasil mosaik, bukan rata-rata metrik tile apabila evaluasi utama menggunakan agregasi mosaik;
8. tidak mengubah checkpoint;
9. tidak menjalankan optimizer, backward pass, atau scheduler.

Target kombinasi:

```text
U-Net:
- all
- sentinel1
- sentinel2
- demnas

ProCANet:
- all
- sentinel1
- sentinel2
- demnas
```

Total: 8 inference run.

## Metrik

Gunakan metrik yang sama seperti evaluasi utama:

- IoU
- Dice/F1
- Accuracy
- Precision
- Recall
- Specificity
- FPR
- FNR
- TP
- TN
- FP
- FN

Tambahkan perubahan performa terhadap skenario input penuh:

```text
delta_iou = iou_all - iou_scenario
delta_dice = dice_all - dice_scenario
```

Untuk skenario `all`:

```text
delta_iou = 0
delta_dice = 0
```

Jangan membandingkan metrik yang dihitung menggunakan mask atau populasi piksel berbeda.

## Evaluasi Sentinel-2

Evaluasi utama skenario `sentinel2` tetap menggunakan **valid mask yang sama dengan skenario lain** agar hasil dapat dibandingkan secara langsung.

Evaluasi tambahan `Sentinel-2 valid-only` boleh dibuat apabila `s2_valid_mask` tersedia, dengan ketentuan:

- hasil valid-only disimpan terpisah;
- tidak menggantikan hasil utama;
- diberi label jelas sebagai analisis tambahan;
- tidak dibandingkan langsung dengan skenario lain tanpa penjelasan bahwa populasi pikselnya berbeda.

Output tambahan opsional:

```text
metrics_s2_valid_only.json
metrics_s2_valid_only.csv
```

## Output

Ikuti kontrak output kanonis `bab4/`:

- hasil evaluasi mentah berada di `bab4/evaluation/`, bukan di `bab4/outputs/`;
- `bab4/outputs/` tetap datar dan hanya memiliki `tables/`, `figures/`, serta `narratives/`;
- artefak laporan didaftarkan melalui `bab4/artifacts.py`, dibuat oleh generator section, dan masuk `bab4_output_manifest.csv`;
- `Bab4Config.reset_output_dirs()` boleh membangun ulang artefak laporan tanpa menghapus hasil evaluasi mentah.

Struktur hasil evaluasi mentah:

```text
bab4/evaluation/modality_masking/
├── unet/
│   ├── all/eval_test/
│   ├── sentinel1/eval_test/
│   ├── sentinel2/eval_test/
│   └── demnas/eval_test/
├── procanet/
│   ├── all/eval_test/
│   ├── sentinel1/eval_test/
│   ├── sentinel2/eval_test/
│   └── demnas/eval_test/
├── modality_metrics.csv
├── modality_metrics.json
└── provenance.json
```

Setiap `eval_test/` mengikuti format inference lama dan minimal menyimpan:

- `metrics.csv` dan `metrics.json`;
- `predictions/<region>/*.npz`;
- `geotiff/` bila mosaik GeoTIFF diaktifkan;
- checkpoint, threshold, wilayah, dan nama skenario input dalam metadata.

Hasil `Sentinel-2 valid-only`, bila dibuat, disimpan sebagai file terpisah di folder skenario `sentinel2/eval_test/`.

Artefak laporan berada pada lokasi kanonis berikut:

```text
bab4/outputs/
├── tables/
│   ├── 4_7_modality_masking_metrics.csv
│   └── 4_7_modality_masking_s2_valid_only.csv
├── figures/
│   ├── 4_7_unet_modality_masking_panel.png
│   └── 4_7_procanet_modality_masking_panel.png
└── narratives/
    └── 4_7_modality_masking_interpretation.md
```

Tabel utama memiliki tepat delapan baris model-skenario dan kolom:

```text
model
input_scenario
iou
dice_f1
accuracy
precision
recall
specificity
fpr
fnr
tp
tn
fp
fn
delta_iou
delta_dice
checkpoint
threshold
```

Jangan membuat `bab4/outputs/modality_masking/` atau folder skenario di dalam `bab4/outputs/`.

## Visualisasi

Buat satu panel per model menggunakan tile atau area yang sama untuk seluruh skenario.

Panel minimal:

1. label referensi;
2. prediksi `all`;
3. prediksi `sentinel1`;
4. prediksi `sentinel2`;
5. prediksi `demnas`.

Bila ruang memungkinkan, tambahkan input representatif:

- Sentinel-1 VV;
- Sentinel-2 pseudo-RGB HSV;
- HAND atau Slope.

Gunakan tile yang sama untuk U-Net dan ProCANet agar pola perbedaannya dapat dibandingkan.

Tidak perlu memilih banyak kasus sebelum hasil numerik tersedia. Prioritaskan:

- satu kasus representatif;
- satu kasus dengan Sentinel-2 kosong atau hampir kosong;
- satu kasus badan air permanen atau false positive dominan, bila relevan.

## Validasi dan Tes

### Unit test modality masking

Tambahkan tes untuk memastikan:

- shape tensor tidak berubah;
- `all` identik dengan input;
- hanya indeks yang benar yang dipertahankan;
- channel lain tepat bernilai nol;
- tensor input asli tidak berubah;
- `dtype` dan `device` dipertahankan;
- nama skenario invalid menghasilkan error yang jelas.

### Regression test skenario `all`

Skenario `all` wajib dibandingkan dengan pipeline inference lama.

Kriteria:

- probabilitas/logit identik atau berbeda hanya dalam toleransi numerik yang sangat kecil;
- mask prediksi identik;
- confusion matrix identik;
- metrik identik.

Jika hasil `all` berubah secara material, hentikan eksperimen dan perbaiki pipeline sebelum menjalankan skenario lain.

### Tes ProCANet

Pastikan:

- encoder kedua menerima `VV`, `VH` asli pada `all` dan `sentinel1`;
- encoder kedua menerima tensor nol pada `sentinel2` dan `demnas`;
- tidak ada perubahan shape atau error cross-attention;
- checkpoint lama tetap dapat dimuat tanpa migrasi.

### Tes output

Pastikan:

- setiap kombinasi model-skenario menghasilkan folder terpisah;
- hasil lama tidak tertimpa;
- summary memiliki tepat 8 baris utama;
- delta dihitung terhadap skenario `all` dari model yang sama;
- konfigurasi checkpoint dan threshold tercatat.

## Dokumentasi

Perbarui `README.md` atau dokumentasi eksperimen dengan:

1. definisi modality masking;
2. perbedaan modality masking dan retraining unimodal;
3. urutan tujuh channel;
4. contoh command;
5. struktur output;
6. batas interpretasi hasil;
7. penjelasan bahwa nilai nol bukan representasi netral sempurna karena beberapa channel memiliki makna fisik pada nilai nol.

Contoh command akhir harus mengikuti CLI aktual repository, misalnya:

```bash
python scripts/evaluate_modality_masking.py \
  --unet-checkpoint <path-checkpoint-unet> \
  --procanet-checkpoint <path-checkpoint-procanet> \
  --test-region Aceh_Utara \
  --output-dir bab4/evaluation/modality_masking
```

Atau per skenario:

```bash
python -m scripts.infer_segmentation \
  --architecture unet \
  --checkpoint <path-checkpoint> \
  --region Aceh_Utara \
  --input-scenario sentinel1 \
  --output-dir bab4/evaluation/modality_masking/unet/sentinel1/eval_test
```

Sesuaikan command setelah nama argumen dan entry point aktual repository diverifikasi.

## Urutan Eksekusi

1. Periksa implementasi dataset, inference U-Net, inference ProCANet, mosaicking, valid mask, dan evaluasi metrik.
2. Identifikasi titik paling aman untuk menerapkan masking sesudah preprocessing dan sebelum forward pass.
3. Implementasikan fungsi modality masking terpusat.
4. Tambahkan CLI `--input-scenario`.
5. Pastikan skenario `all` kompatibel dengan pipeline lama.
6. Tambahkan batch evaluation empat skenario untuk dua model.
7. Tambahkan metrik dan agregasi summary.
8. Tambahkan unit test dan regression test.
9. Jalankan seluruh test terkait.
10. Jalankan smoke test pada sejumlah kecil tile.
11. Jalankan regression test penuh untuk skenario `all`.
12. Jalankan delapan inference pada Aceh Utara.
13. Verifikasi confusion matrix dan metrik.
14. Buat tabel serta visualisasi.
15. Tulis hasil faktual ke artefak BAB 4 setelah angka nyata tersedia.

## Kriteria Selesai

Eksperimen dianggap selesai apabila:

- tidak ada training baru yang dijalankan;
- checkpoint lama digunakan tanpa perubahan;
- model tetap menggunakan jumlah channel asli;
- tersedia hasil untuk 4 skenario pada U-Net dan 4 skenario pada ProCANet;
- skenario `all` mereproduksi hasil evaluasi utama;
- seluruh skenario menggunakan test set, mask, threshold, dan mosaicking yang sama;
- output tidak menimpa hasil lama;
- metrik dan delta tersimpan dalam CSV/JSON;
- visualisasi menggunakan lokasi yang sama;
- dokumentasi menjelaskan batas interpretasi;
- tidak ada angka atau kesimpulan yang dibuat sebelum inference selesai.

## Tidak Dikerjakan

- Training ulang U-Net atau ProCANet.
- Training model Sentinel-1-only, Sentinel-2-only, atau DEMNAS-only.
- Grid search atau hyperparameter tuning.
- Mengubah `in_channels` model.
- Mengubah arsitektur U-Net.
- Mengubah dual-encoder atau cross-attention ProCANet.
- Membuat checkpoint baru.
- Membuat salinan dataset per modalitas.
- Mengubah split train, validation, atau test.
- Mengganti threshold, valid mask, atau prosedur mosaicking.
- Menyimpulkan kontribusi kausal setiap modalitas.
- Menyatakan hasil sebagai performa model unimodal.
- Menulis angka atau kesimpulan sebelum hasil inference nyata tersedia.
