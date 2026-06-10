# Checklist Output BAB 4 — Wajib, Disarankan, Opsional

Dokumen ini adalah turunan praktis dari `TODO_BAB_4_Lengkap.md`. Fokusnya hanya pada **output yang harus dibuat** untuk BAB 4: tabel, gambar, grafik, diagram, dan narasi pembahasan pendukung. Gunakan ini sebagai daftar kerja ketika membuat file visual, tabel, dan materi pembahasan.

## Kode Prioritas

| Kode | Prioritas | Makna |
|---|---|---|
| W | Wajib | Harus ada agar BAB 4 kuat dan konsisten dengan alur penelitian. |
| SD | Sangat disarankan | Nilai ilmiahnya tinggi; sebaiknya dibuat jika data/script memungkinkan. |
| D | Disarankan | Membantu pembaca, tetapi bisa dilewati jika subbab sudah cukup kuat. |
| O | Opsional | Tambahan untuk mempercantik atau memperkuat argumen. |
| N | Narasi wajib | Tidak berupa tabel/gambar, tetapi harus ditulis sebagai interpretasi. |

---

# 1. Checklist Ringkas Output Utama

| Status | Prioritas | Output | Bentuk | Subbab | Tujuan |
|---|---|---|---|---|---|
| [x] | W | Statistik preprocessing VV/VH | Tabel | 4.1.1 | Membuktikan Sentinel-1 sudah dinormalisasi dan valid. |
| [x] | W | Persentase valid Sentinel-2 per wilayah | Tabel | 4.1.1 | Menunjukkan kualitas data optis dan kasus S2 kosong/hampir kosong. |
| [x] | W | Statistik Slope dan HAND | Tabel | 4.1.1 | Menunjukkan fitur topografi siap digunakan. |
| [x] | W | Contoh visual channel VV, VH, HSV/pseudo-RGB, Slope, HAND | Gambar panel | 4.1.1 | Memperlihatkan karakter visual tiap fitur input. |
| [x] | W | Verifikasi alignment raster multisensor | Tabel | 4.1.2 | Membuktikan grid, CRS, geotransform, dan dimensi layer sama. |
| [x] | W | Overlay OpenStreetMap terhadap layer stack | Gambar panel | 4.1.2 | Membuktikan secara visual bahwa stack tidak bergeser. |
| [x] | W | Statistik label UNOSAT per wilayah | Tabel | 4.2 | Menunjukkan sebaran piksel banjir, valid mask, dan water/river. |
| [ ] | W | Panel mask UNOSAT | Gambar panel | 4.2 | Menjelaskan perbedaan `label_flood_binary`, `label_valid_mask`, dan `water_river_mask`. |
| [x] | W | Jumlah tile positif/background per wilayah | Tabel | 4.2 | Menjelaskan distribusi dataset dan class imbalance. |
| [x] | W | Pembagian 5-fold spatial cross-validation | Tabel | 4.3 | Menjelaskan validasi berbasis wilayah dan final test Aceh_Utara. |
| [ ] | W | Jumlah tile train/val/test per fold | Tabel | 4.3 | Memastikan ukuran data tiap fold transparan. |
| [x] | W | Spesifikasi arsitektur U-Net dan ProCANet | Tabel | 4.4.1 | Membuktikan implementasi model sesuai repo. |
| [x] | W | Verifikasi forward pass model | Tabel | 4.4.1 | Membuktikan input-output shape model benar. |
| [x] | W | Hasil grid search hyperparameter | Tabel | 4.4.2 | Menunjukkan konfigurasi terbaik dipilih berdasarkan eksperimen. |
| [x] | W | Kurva training/validation loss, IoU, dan Dice | Grafik | 4.4.3 | Menunjukkan stabilitas proses pelatihan. |
| [x] | W | Metrik final U-Net vs ProCANet pada Aceh_Utara | Tabel | 4.5 | Menjawab perbandingan performa model. |
| [x] | W | Confusion matrix TP, TN, FP, FN | Tabel | 4.5 | Menjelaskan karakter kesalahan model. |
| [x] | W | Panel visual prediksi U-Net vs ProCANet | Gambar panel | 4.6 | Memperlihatkan hasil segmentasi secara spasial. |
| [x] | SD | Error map TP/FP/FN/TN | Gambar | 4.6 | Membuat pembahasan false positive dan false negative lebih kuat. |
| [x] | SD | Ringkasan U-Net vs ProCANet | Tabel | 4.7 | Merangkum trade-off arsitektur, metrik, FP, dan FN. |
| [x] | SD | Studi kasus Sentinel-2 kosong/hampir kosong | Tabel/grafik | 4.8 | Menguji ketahanan model pada kondisi data ekstrem. |
| [x] | O | Tabel mini ringkasan temuan BAB 4 | Tabel | 4.9 | Menutup BAB 4 dengan sintesis singkat. |

---

# 2. Checklist Detail per Subbab

## 4.1.1 Statistik Hasil Preprocessing

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Statistik Sentinel-1 VV/VH | Tabel | Wilayah, VV min/max/mean/std, VH min/max/mean/std, persentase piksel valid. | `dataset/features_preprocessed/<region>/vv_norm.tif`, `vh_norm.tif`, `feature_preprocessing_summary.csv` |
| [x] | W | Statistik validitas Sentinel-2 | Tabel | Wilayah, jumlah piksel valid S2, persentase `s2_valid_mask`, status kualitas: baik/rendah/hampir kosong/kosong. | `s2_valid_mask.tif`, `feature_preprocessing_summary.csv` |
| [x] | W | Statistik DEMNAS turunan | Tabel | Wilayah, Slope min/max/mean/std, HAND min/max/mean/std. | `slope_norm.tif`, `hand_norm.tif` |
| [x] | W | Contoh channel input multisensor | Gambar panel | VV, VH, pseudo-RGB/HSV, Slope, HAND pada wilayah representatif. | `vv_norm.tif`, `vh_norm.tif`, `hue.tif`, `saturation.tif`, `value.tif`, `slope_norm.tif`, `hand_norm.tif` |
| [x] | D | Contoh perbandingan S2 valid vs S2 kosong | Gambar panel | Satu wilayah dengan S2 valid dan satu wilayah S2 kosong/hampir kosong. | `s2_valid_mask.tif`, HSV/pseudo-RGB |
| [x] | N | Interpretasi karakter input | Narasi | Jelaskan fungsi SAR, optis, dan topografi; jelaskan kenapa fusi multisensor diperlukan. | Hasil tabel/gambar 4.1.1 |

Catatan output: tabel statistik tidak perlu terlalu besar. Jika jumlah wilayah banyak, gunakan ringkasan per wilayah dan lampirkan statistik detail di lampiran.

---

## 4.1.2 Verifikasi Kesesuaian Grid dan Stacking Multisensor

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Verifikasi alignment raster per wilayah | Tabel | Wilayah, raster referensi, ukuran raster, resolusi, CRS/proyeksi, geotransform match, status alignment. | `stack_7ch.tif`, semua layer fitur, GDAL metadata |
| [x] | D | Verifikasi layer dalam `stack_7ch.tif` | Tabel teknis | Band, nama layer, sumber, ukuran raster, CRS/proyeksi, geotransform, status. | `dataset/features_preprocessed/<region>/stack_7ch.tif` |
| [x] | W | Overlay OpenStreetMap dengan layer stack | Gambar panel | OSM + VV, OSM + VH, OSM + HSV/pseudo-RGB, OSM + Slope, OSM + HAND, OSM + label UNOSAT. | Layer fitur + OSM basemap |
| [x] | SD | Panel satu tile multi-layer dari koordinat yang sama | Gambar panel | VV, VH, pseudo-RGB/HSV, Slope, HAND, label UNOSAT, OSM outline. | `dataset/tiles/.../*.npz` atau crop raster |
| [x] | N | Dampak jika layer bergeser | Narasi | Jelaskan bahwa pergeseran satu layer akan membuat model belajar hubungan fitur-label yang salah. | Tabel alignment + overlay OSM |
| [x] | N | Kesimpulan kelayakan stack | Narasi | Nyatakan bahwa stack layak digunakan jika seluruh layer selaras. | Hasil verifikasi |

Format tabel yang disarankan:

```markdown
| Wilayah | Raster referensi | Ukuran raster | Resolusi | CRS/proyeksi | Geotransform | Status |
|---|---|---:|---:|---|---|---|
| Aceh_Utara | Sentinel-1 | H × W | 10 m | Sama | Sama | Selaras |
```

Format gambar yang disarankan:

```text
Gambar 4.x Verifikasi overlay OpenStreetMap terhadap stack multisensor pada wilayah Aceh_Utara:
(a) OSM+VV, (b) OSM+VH, (c) OSM+pseudo-RGB/HSV, (d) OSM+Slope, (e) OSM+HAND, (f) OSM+label UNOSAT.
```

---

## 4.2 Hasil Pembentukan Label UNOSAT dan Dataset Tile

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Statistik label UNOSAT per wilayah | Tabel | Wilayah, piksel valid, piksel banjir, persentase banjir terhadap area valid, piksel water/river. | `label_flood_binary.tif`, `label_valid_mask.tif`, `label_water_river_mask.tif` |
| [ ] | W | Panel mask UNOSAT | Gambar panel | `label_flood_binary`, `label_valid_mask`, `water_river_mask` pada wilayah representatif. | `dataset/labels_unosat_rasterized/<region>/` |
| [x] | W | Jumlah tile per wilayah | Tabel | Wilayah, total tile, tile positif, tile background, rasio positif-background. | `dataset/tiles/<split>/<region>/*.npz` |
| [x] | D | Bar chart persentase banjir per wilayah | Grafik batang | Wilayah vs persentase banjir terhadap area valid. | Tabel statistik label |
| [ ] | O | Contoh tile positif dan tile background | Gambar panel | Tile positif dan tile background-only, minimal VV/HSV/HAND + label. | `dataset/tiles/.../*.npz` |
| [ ] | N | Penjelasan fungsi mask | Narasi | Tegaskan `FloodExtent` menjadi label banjir; `AnalysisExtent` menjadi valid mask; `WaterExtent`/`River` auxiliary, bukan label utama. | Hasil rasterisasi UNOSAT |
| [x] | N | Penjelasan class imbalance | Narasi | Jelaskan ketimpangan banjir vs non-banjir dan kenapa IoU/Dice penting. | Statistik label + tile |

---

## 4.3 Pembagian Dataset Spasial dan Validasi Eksperimental

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Tabel 5-fold spatial cross-validation | Tabel | Fold, wilayah validasi, wilayah training, wilayah test final. | `scripts/preprocessing_utils.py`, konfigurasi split |
| [ ] | W | Jumlah tile train/val/test per fold | Tabel | Fold, jumlah tile train, jumlah tile validation, jumlah tile test, total tile. | `dataset/tiles/`, metadata fold |
| [ ] | D | Diagram skema spatial split | Diagram | Aceh_Utara sebagai final test, 10 wilayah lain sebagai 5-fold CV. | Daftar wilayah/fold |
| [ ] | N | Alasan menghindari random split tile | Narasi | Jelaskan spatial leakage dan kenapa spatial split lebih realistis. | Desain eksperimen |
| [ ] | N | Kesimpulan validasi eksperimental | Narasi | Tegaskan model diuji pada wilayah yang tidak pernah dilihat. | Tabel fold + split |

Format tabel fold:

```markdown
| Fold | Wilayah validasi | Wilayah training | Wilayah test final |
|---:|---|---|---|
| 0 | Pidie, Pidie_Jaya | 8 wilayah CV lainnya | Aceh_Utara |
| 1 | Aceh_Besar, Banda_Aceh | 8 wilayah CV lainnya | Aceh_Utara |
| 2 | Aceh_Tamiang, Aceh_Timur | 8 wilayah CV lainnya | Aceh_Utara |
| 3 | Bireuen, Langsa | 8 wilayah CV lainnya | Aceh_Utara |
| 4 | Agam, Pasaman_Barat | 8 wilayah CV lainnya | Aceh_Utara |
```

---

## 4.4.1 Hasil Implementasi Arsitektur Model

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Spesifikasi implementasi U-Net dan ProCANet | Tabel | Jenis model, input, output, base channels, depth, blok konvolusi, mekanisme fusi, decoder. | `training/models/unet.py`, `procanet.py`, `blocks.py` |
| [x] | W | Verifikasi forward pass | Tabel | Model/komponen, input shape, output shape, status. | `tests/test_models.py` |
| [ ] | O | Diagram U-Net aktual | Diagram | Input 7ch → encoder → bottleneck → decoder → logit 1ch. | Implementasi U-Net |
| [ ] | SD | Diagram ProCANet aktual | Diagram | Encoder 1 7ch + Encoder 2 2ch → PCAB skip/bottleneck → decoder → logit 1ch. | Implementasi ProCANet |
| [ ] | N | Penjelasan output logit | Narasi | Output model adalah logit 1-channel; mask biner diperoleh dari sigmoid + threshold 0,5 saat inferensi. | Implementasi + evaluasi |
| [x] | N | Penjelasan perbedaan model | Narasi | U-Net = fusi langsung 7-channel; ProCANet = fusi selektif dual encoder + attention. | Tabel arsitektur |

Format tabel arsitektur:

```markdown
| Komponen | U-Net | ProCANet |
|---|---|---|
| Jenis model | Single encoder-decoder | Dual encoder-decoder dengan Progressive Cross-Attention |
| Input utama | 7 channel: VV, VH, Hue, Saturation, Value, Slope, HAND | Encoder 1: 7 channel penuh; Encoder 2: VV, VH |
| Output akhir | 1 channel logit | 1 channel logit |
```

---

## 4.4.2 Hasil Tuning Hyperparameter

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Grid search U-Net | Tabel | Varian, learning rate, weight decay, mean val IoU, std val IoU, mean val Dice, mean val loss, status terbaik. | `runs/unet/.../metrics.csv`, rekap grid search |
| [x] | W | Grid search ProCANet | Tabel | Format sama dengan U-Net. | `runs/procanet/.../metrics.csv`, rekap grid search |
| [x] | D | Tabel gabungan grid search | Tabel | U-Net dan ProCANet dalam satu tabel, baris terbaik diberi penanda. | Rekap tuning |
| [x] | N | Interpretasi learning rate | Narasi | Jelaskan efek learning rate terhadap stabilitas dan performa. | Tabel grid search |
| [ ] | N | Interpretasi weight decay | Narasi | Jelaskan efek regularisasi terhadap overfitting/underfitting. | Tabel grid search |
| [ ] | N | Keterbatasan ruang grid search | Narasi | Tegaskan hanya enam kombinasi lr/wd yang diuji. | Desain tuning |

---

## 4.4.3 Stabilitas Pelatihan Model

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Kurva training vs validation loss | Grafik | Epoch pada sumbu x; train loss dan val loss untuk U-Net dan ProCANet. | `runs/<model>/fold_*/metrics.csv`, `runs/final/<model>/metrics.csv` |
| [x] | W | Kurva training vs validation IoU | Grafik | Epoch pada sumbu x; train IoU dan val IoU. | Metrics CSV |
| [x] | W | Kurva training vs validation Dice/F1 | Grafik | Epoch pada sumbu x; train Dice dan val Dice. | Metrics CSV |
| [x] | D | Kurva learning rate | Grafik | Epoch vs learning rate untuk melihat efek scheduler. | Metrics CSV |
| [ ] | N | Penjelasan checkpoint terbaik | Narasi | Checkpoint terbaik dipilih berdasarkan validation IoU. | Training metrics |
| [ ] | N | Penjelasan loss | Narasi | Loss adalah masked BCE + Dice Loss; hanya piksel valid yang dihitung. | Training implementation |
| [ ] | N | Interpretasi overfitting/stabilitas | Narasi | Bahas jarak train-val, stagnasi IoU, penurunan learning rate, dan early stopping. | Grafik training |

Catatan visual: grafik boleh berupa satu gambar multi-panel agar tidak terlalu banyak figure.

---

## 4.5 Evaluasi Akhir Model pada Wilayah Uji Aceh_Utara

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Metrik final U-Net vs ProCANet | Tabel | Model, loss, IoU, Dice/F1, accuracy, interpretasi singkat. | `runs/final/.../metrics.csv`, output inferensi |
| [x] | W | Confusion matrix piksel | Tabel | Model, TP, TN, FP, FN. Tambahkan precision dan recall jika memungkinkan. | Output inferensi/evaluasi |
| [ ] | O | Bar chart FP/FN | Grafik batang | FP dan FN U-Net vs ProCANet. | Confusion matrix |
| [ ] | N | Interpretasi IoU dan Dice | Narasi | Jelaskan model mana yang tumpang tindihnya lebih baik terhadap UNOSAT. | Tabel metrik |
| [ ] | N | Interpretasi FP/FN | Narasi | Jelaskan model agresif vs konservatif. | Confusion matrix |
| [ ] | N | Posisi akurasi | Narasi | Akurasi hanya metrik pelengkap karena kelas non-banjir dominan. | Tabel metrik |

---

## 4.6 Analisis Visual dan Spasial Hasil Segmentasi

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | W | Panel visual prediksi U-Net vs ProCANet | Gambar panel | VV, VH/HSV, Slope/HAND, label UNOSAT, prediksi U-Net, prediksi ProCANet. | Tile Aceh_Utara + output prediksi |
| [x] | SD | Error map model | Gambar | TP, FP, FN, TN untuk U-Net dan/atau ProCANet. | Prediksi + label + valid mask |
| [ ] | O | Tabel aspek visual model | Tabel | Aspek visual, U-Net, ProCANet: cakupan banjir, FP, FN, batas prediksi, konsistensi label. | Panel prediksi + error map |
| [ ] | N | Interpretasi area benar | Narasi | Jelaskan bagian yang berhasil dideteksi oleh kedua model. | Panel visual |
| [x] | N | Interpretasi false positive | Narasi | Hubungkan FP dengan backscatter rendah, radar shadow, tanah basah, atau badan air permanen. | Error map/panel |
| [x] | N | Interpretasi false negative | Narasi | Hubungkan FN dengan banjir tipis, label tidak presisi, atau sinyal sensor lemah. | Error map/panel |
| [ ] | N | Pengait visual-kuantitatif | Narasi | Tegaskan visual harus dibaca bersama metrik, bukan sebagai bukti tunggal. | 4.5 + 4.6 |

---

## 4.7 Pembahasan Efektivitas U-Net vs ProCANet

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | SD | Ringkasan perbandingan U-Net vs ProCANet | Tabel | Aspek, U-Net, ProCANet, interpretasi: strategi fusi, IoU, Dice, FP, FN, karakter prediksi, kelebihan, keterbatasan. | Tabel metrik, confusion matrix, visual 4.6 |
| [ ] | O | Diagram arsitektur tambahan | Diagram | Hanya jika diagram belum ditampilkan di 4.4.1. | Implementasi model |
| [ ] | O | Bar chart trade-off FP/FN | Grafik | Perbandingan FP dan FN untuk menjelaskan agresif/konservatif. | Confusion matrix |
| [ ] | O | Tabel komparasi literatur | Tabel | Penelitian, model, data, metrik, perbedaan konteks dengan penelitian ini. | Bab 2 dan hasil penelitian |
| [ ] | N | Pembahasan U-Net | Narasi | Jelaskan U-Net sebagai baseline fusi langsung yang sederhana/stabil. | Hasil evaluasi |
| [ ] | N | Pembahasan ProCANet | Narasi | Jelaskan ProCANet sebagai dual encoder + attention yang lebih selektif. | Hasil evaluasi |
| [ ] | N | Pembahasan trade-off | Narasi | Jangan memaksa ProCANet menang mutlak; bahas aspek unggul/kalah secara jujur. | Metrik + visual |
| [ ] | N | Komparasi dengan literatur | Narasi | Jelaskan apakah hasil mendukung, berbeda, atau memperluas studi sebelumnya. | Literatur Bab 2 |

---

## 4.8 Ketahanan Model pada Kondisi Data Ekstrem

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | SD | Wilayah Sentinel-2 kosong/hampir kosong | Tabel | Wilayah ekstrem, persentase S2 valid, IoU/Dice validasi jika tersedia, model yang lebih stabil, catatan interpretasi. | `s2_valid_mask`, metrics per fold/wilayah |
| [ ] | O | Contoh tile HSV=0 | Gambar | Tile dengan HSV kosong/hampir kosong, ditampilkan bersama VV/VH dan prediksi. | Tile wilayah ekstrem |
| [ ] | O | Studi kasus topografi sulit/radar shadow | Gambar | VV + Slope/HAND + prediksi + error map pada area curam. | Fitur topografi + prediksi |
| [ ] | O | Studi kasus badan air permanen | Gambar | `water_river_mask` dibandingkan dengan prediksi model. | Water/river mask + prediksi |
| [x] | N | Peran SAR saat S2 tidak valid | Narasi | Jelaskan VV/VH sebagai sumber informasi utama ketika HSV tidak informatif. | Studi kasus ekstrem |
| [ ] | N | Peran Slope/HAND | Narasi | Jelaskan apakah topografi membantu mengurangi FP pada lereng/radar shadow. | Studi kasus topografi |
| [ ] | N | Keterbatasan UNOSAT | Narasi | UNOSAT adalah proxy label, tidak pixel-perfect, ada potensi beda waktu akuisisi. | Hasil + desain label |
| [ ] | N | Keterbatasan generalisasi | Narasi | Generalisasi masih terbatas pada wilayah studi Sumatra. | Pembahasan akhir |

---

## 4.9 Ringkasan Temuan Bab 4

| Status | Prioritas | Output | Bentuk | Isi minimal | Sumber data/berkas |
|---|---|---|---|---|---|
| [x] | O | Tabel mini temuan utama | Tabel | Aspek, temuan utama, implikasi. | Ringkasan 4.1–4.8 |
| [x] | N | Jawaban tujuan penelitian pertama | Narasi | Penerapan U-Net dan ProCANet dengan input fusi multisensor berhasil dilakukan. | Hasil implementasi + training |
| [x] | N | Jawaban tujuan penelitian kedua | Narasi | Perbandingan performa U-Net dan ProCANet berdasarkan metrik dan karakter kesalahan. | Evaluasi akhir |
| [x] | N | Ringkasan keterbatasan | Narasi | Sentinel-2 kosong, UNOSAT proxy label, generalisasi terbatas. | 4.8 |
| [x] | N | Transisi ke BAB 5 | Narasi | Arahkan pembaca menuju kesimpulan dan saran. | Sintesis BAB 4 |

---

# 3. Daftar Output Berdasarkan Jenis

## 3.1 Tabel

| Status | Prioritas | Nama tabel | Subbab |
|---|---|---|---|
| [x] | W | Statistik preprocessing VV/VH per wilayah | 4.1.1 |
| [x] | W | Persentase valid Sentinel-2 per wilayah | 4.1.1 |
| [x] | W | Statistik Slope dan HAND per wilayah | 4.1.1 |
| [x] | W | Verifikasi alignment raster multisensor | 4.1.2 |
| [x] | D | Verifikasi layer pada `stack_7ch.tif` | 4.1.2 |
| [x] | W | Statistik label UNOSAT per wilayah | 4.2 |
| [x] | W | Jumlah tile positif/background per wilayah | 4.2 |
| [x] | W | Pembagian 5-fold spatial cross-validation | 4.3 |
| [ ] | W | Jumlah tile train/val/test per fold | 4.3 |
| [x] | W | Spesifikasi arsitektur U-Net dan ProCANet | 4.4.1 |
| [x] | W | Verifikasi forward pass model | 4.4.1 |
| [x] | W | Hasil grid search U-Net | 4.4.2 |
| [x] | W | Hasil grid search ProCANet | 4.4.2 |
| [x] | W | Metrik final U-Net vs ProCANet | 4.5 |
| [x] | W | Confusion matrix piksel | 4.5 |
| [x] | SD | Ringkasan perbandingan U-Net vs ProCANet | 4.7 |
| [x] | SD | Studi kasus Sentinel-2 kosong/hampir kosong | 4.8 |
| [x] | O | Ringkasan temuan utama BAB 4 | 4.9 |

## 3.2 Gambar, Diagram, dan Grafik

| Status | Prioritas | Nama output | Jenis | Subbab |
|---|---|---|---|---|
| [x] | W | Visual channel VV, VH, HSV/pseudo-RGB, Slope, HAND | Gambar panel | 4.1.1 |
| [x] | W | Overlay OpenStreetMap terhadap layer stack | Gambar panel | 4.1.2 |
| [x] | SD | Tile multi-layer dari lokasi yang sama | Gambar panel | 4.1.2 |
| [ ] | W | Panel mask UNOSAT | Gambar panel | 4.2 |
| [ ] | O | Contoh tile positif dan background | Gambar panel | 4.2 |
| [ ] | D | Diagram spatial split | Diagram | 4.3 |
| [ ] | O | Diagram U-Net aktual | Diagram | 4.4.1 |
| [ ] | SD | Diagram ProCANet aktual | Diagram | 4.4.1 |
| [x] | W | Training/validation loss | Grafik kurva | 4.4.3 |
| [x] | W | Training/validation IoU | Grafik kurva | 4.4.3 |
| [x] | W | Training/validation Dice/F1 | Grafik kurva | 4.4.3 |
| [x] | D | Learning rate curve | Grafik kurva | 4.4.3 |
| [ ] | O | Bar chart FP/FN | Grafik batang | 4.5 atau 4.7 |
| [x] | W | Panel prediksi U-Net vs ProCANet | Gambar panel | 4.6 |
| [x] | SD | Error map TP/FP/FN/TN | Gambar | 4.6 |
| [ ] | O | Contoh tile HSV=0 | Gambar panel | 4.8 |
| [ ] | O | Studi kasus topografi/radar shadow | Gambar panel | 4.8 |
| [ ] | O | Studi kasus badan air permanen | Gambar panel | 4.8 |

## 3.3 Narasi Wajib

| Status | Subbab | Narasi yang harus ada |
|---|---|---|
| [x] | 4.1.1 | Interpretasi karakter input SAR, optis, dan topografi. |
| [x] | 4.1.1 | Alasan fusi multisensor diperlukan. |
| [ ] | 4.1.2 | Penjelasan kenapa alignment/stacking penting untuk akurasi model. |
| [x] | 4.1.2 | Kesimpulan bahwa stack layak digunakan jika semua layer selaras. |
| [ ] | 4.2 | Perbedaan fungsi `label_flood_binary`, `label_valid_mask`, dan `water_river_mask`. |
| [x] | 4.2 | Penjelasan class imbalance dan penggunaan IoU/Dice. |
| [x] | 4.3 | Alasan spatial split lebih kuat daripada random split tile. |
| [x] | 4.4.1 | Perbedaan U-Net sebagai fusi langsung dan ProCANet sebagai fusi attention. |
| [ ] | 4.4.2 | Interpretasi efek learning rate dan weight decay. |
| [ ] | 4.4.3 | Interpretasi stabilitas training, early stopping, dan scheduler. |
| [ ] | 4.5 | Interpretasi metrik final, akurasi sebagai pelengkap, dan trade-off FP/FN. |
| [x] | 4.6 | Interpretasi visual TP/FP/FN, bukan hanya deskripsi gambar. |
| [ ] | 4.7 | Pembahasan jujur apakah ProCANet unggul atau tidak, beserta alasan. |
| [x] | 4.8 | Pembahasan keterbatasan UNOSAT, Sentinel-2 kosong, dan generalisasi model. |
| [x] | 4.9 | Sintesis yang menjawab dua tujuan penelitian. |

---

# 4. Urutan Pengerjaan yang Disarankan

| Urutan | Output yang dibuat | Alasan |
|---:|---|---|
| 1 | Tabel statistik preprocessing dan valid mask | Menjadi dasar 4.1. |
| 2 | Tabel alignment + overlay OSM | Membuktikan stack valid sebelum model dibahas. |
| 3 | Tabel label UNOSAT dan tile | Menjadi dasar pembahasan dataset. |
| 4 | Tabel spatial split dan fold | Menjadi dasar validasi eksperimen. |
| 5 | Tabel arsitektur + forward pass | Menunjukkan model berhasil dibangun. |
| 6 | Tabel grid search + kurva training | Menunjukkan proses training dan tuning. |
| 7 | Tabel metrik final + confusion matrix | Menjadi hasil kuantitatif utama. |
| 8 | Panel prediksi + error map | Menjadi pembahasan visual/spasial. |
| 9 | Tabel ringkasan U-Net vs ProCANet + studi kasus ekstrem | Menjadi pembahasan analitis. |
| 10 | Ringkasan temuan | Menutup BAB 4. |

---

# 5. Output yang Bisa Dimasukkan ke Lampiran

Beberapa output terlalu detail untuk BAB 4 utama. Gunakan lampiran jika diperlukan.

| Status | Output | Alasan dimasukkan ke lampiran |
|---|---|---|
| [ ] | Statistik lengkap semua channel per wilayah dan per band | Terlalu detail untuk narasi utama. |
| [ ] | Semua overlay OSM untuk seluruh wilayah | BAB 4 cukup menampilkan wilayah representatif. |
| [ ] | Semua tile contoh positif/background | BAB 4 cukup menampilkan beberapa contoh. |
| [ ] | Training curve seluruh fold secara terpisah | BAB 4 cukup menampilkan ringkasan atau fold representatif. |
| [ ] | Semua prediksi tile pada Aceh_Utara | BAB 4 cukup menampilkan tile representatif. |
| [ ] | Semua error map per tile | Terlalu banyak untuk bagian utama. |

---

# 6. Kriteria Output Siap Masuk Laporan

Sebelum memasukkan output ke BAB 4, cek hal berikut.

| Status | Kriteria |
|---|---|
| [ ] | Setiap tabel/gambar sudah memiliki nomor dan judul. |
| [ ] | Setiap tabel/gambar dirujuk dalam teks sebelum muncul. |
| [ ] | Setiap tabel/gambar ditafsirkan setelah muncul. |
| [ ] | Tabel tidak terlalu lebar; jika terlalu lebar, pindahkan detail ke lampiran. |
| [ ] | Gambar memiliki resolusi cukup dan label panel jelas: (a), (b), (c), dan seterusnya. |
| [ ] | Warna/legend error map jelas membedakan TP, FP, FN, dan TN. |
| [ ] | Satuan/metrik pada tabel jelas: persen, piksel, IoU, Dice, accuracy. |
| [ ] | Tidak ada output yang hanya ditempel tanpa narasi interpretasi. |
| [ ] | Narasi tidak mengulang metode Bab 3 secara panjang. |
| [x] | Pembahasan tetap jujur terhadap keterbatasan data dan label. |
