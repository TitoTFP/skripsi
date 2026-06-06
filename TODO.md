# TODO Skripsi

Sumber: `5003221164_Proposal TA Final.docx`, terutama Bab 3 Metode Penelitian.
Status dicek terhadap isi repo pada 2026-06-06.

Topik penelitian: integrasi Sentinel-1, Sentinel-2, dan DEMNAS untuk deteksi area banjir di Sumatra dengan baseline U-Net dan model ProCANet.

Catatan metodologi terbaru: rancangan pseudo-label rule-based pada proposal awal sudah outdated. Implementasi sekarang memakai UNOSAT sebagai proxy label utama.

## Sudah Dilakukan

- [x] Studi literatur dan rancangan proposal tersedia.
  - Proposal sudah memuat latar belakang, tinjauan pustaka, metode, variabel, struktur data, metrik evaluasi, dan jadwal.

- [x] Pengumpulan data citra utama tersedia di repo.
  - Sentinel-1 dan Sentinel-2 raw ada di `dataset/satelit raw/`.
  - DEMNAS raw ada di `dataset/DEMNAS_Exports/`.
  - UNOSAT FileGDB ada di `dataset/unosat/FL20251126IDN.gdb`.
  - Batas administrasi wilayah ada di `dataset/batas admin indo/`.

- [x] Area studi Sumatra sudah direpresentasikan per wilayah.
  - 11 wilayah tersedia: Aceh Besar, Aceh Tamiang, Aceh Timur, Aceh Utara, Agam, Banda Aceh, Bireuen, Langsa, Pasaman Barat, Pidie, dan Pidie Jaya.

- [x] Alignment data dasar sudah dilakukan.
  - Sentinel-1 dan Sentinel-2 sudah align per wilayah.
  - Script cropping DEMNAS (`scripts/export_demnas.py`) dan warping (`scripts/warp_demnas.py`) sudah diimplementasi beserta unit test-nya (`tests/test_export_demnas.py`, `tests/test_warp_demnas.py`).
  - DEMNAS sudah di-warp ke grid Sentinel di `dataset/DEMNAS_warped_to_sentinel/`.

- [x] Preprocessing Sentinel-1 sudah dilakukan.
  - Channel `VV` dan `VH` sudah dinormalisasi menjadi `vv_norm.tif` dan `vh_norm.tif`.

- [x] Preprocessing Sentinel-2 sudah dilakukan.
  - Pseudo-RGB dari band Sentinel-2 sudah dikonversi ke HSV.
  - Output `hue.tif`, `saturation.tif`, `value.tif`, dan `s2_valid_mask.tif` tersedia.

- [x] Preprocessing DEMNAS sudah dilakukan.
  - `slope_norm.tif` tersedia.
  - `hand_norm.tif` tersedia.
  - `feature_valid_mask.tif` tersedia.

- [x] Struktur channel input 7-band sudah dibuat.
  - Stack tersedia di `dataset/features_preprocessed/<region>/stack_7ch.tif`.
  - Urutan channel repo saat ini: `VV`, `VH`, `Hue`, `Saturation`, `Value`, `Slope`, `HAND`.

- [x] Proxy label target banjir dari UNOSAT sudah dibuat.
  - Label UNOSAT raster tersedia di `dataset/labels_unosat_rasterized/`.
  - `label_flood_binary.tif` untuk flood.
  - `label_valid_mask.tif` untuk area valid analisis.
  - `label_water_river_mask.tif` untuk water/river auxiliary mask.
  - Rasterisasi UNOSAT sekarang memperbaiki geometri invalid dengan `MakeValid()`, rasterize per-layer, lalu merge raster mask dengan OR/max.
  - `label_valid_mask.tif` dipotong dengan ROI wilayah administratif dari `dataset/batas admin indo/`.
  - UNOSAT dipakai sebagai proxy label utama menggantikan pseudo-label rule-based proposal awal.

- [x] Dataset tile siap training U-Net sudah dibuat.
  - Output canonical ada di `dataset/tiles/7ch/by_region/<region>/`.
  - Ukuran tile: `512 x 512`, stride overlap `256`.
  - Total tile saat dicek: `4423`.
  - Spatial CV: `Aceh_Utara` final test, 10 wilayah lain untuk 5-fold train/validation.

- [x] Dataset siap ProCANet dua encoder sudah dibuat.
  - Output canonical ada di `dataset/tiles/procanet/by_region/<region>/`.
  - Encoder 1: `VV`, `VH`, `Hue`, `Saturation`, `Value`, `Slope`, `HAND`.
  - Encoder 2: `VV`, `VH`.
  - Desain ini mengikuti pola paper ProCANet: encoder utama membawa konteks lengkap, encoder kedua mengulang modalitas air paling informatif. Untuk kasus ini dipilih SAR karena lebih stabil saat awan/hujan dibanding HSV.
  - Total tile sama dengan dataset 7-channel: `4423`.

- [x] Ringkasan dan verifikasi preprocessing sudah tersedia.
  - `dataset/preprocessing_summary.csv`
  - `dataset/feature_preprocessing_summary.csv`
  - `dataset/preprocessing_verification_report.csv`
  - `dataset/preprocessing_todo.md`
  - `README.md`

- [x] Implementasi dataset loader training.
  - Loader U-Net membaca `.npz` tile dari `dataset/tiles/7ch/`.
  - Loader ProCANet membaca `.npz` tile dari `dataset/tiles/procanet/`.
  - Loader mendukung mode fold via `fold=0..4` untuk spatial CV region-level.
  - Loader perlu mengembalikan feature tensor, `y`, `valid_mask`, dan metadata.
  - Loss dan metrik wajib mengabaikan piksel di luar effective mask `valid_mask & feature_valid_mask`.
  - Implemented in `training.datasets`, `training.losses`, and `training.metrics`.

- [x] Implementasi augmentasi data dinamis.
  - Rotasi 90/180/270 derajat.
  - Flip horizontal dan vertikal.
  - Gaussian noise ringan dan channel dropout kecil hanya pada feature tensor.
  - Augmentasi hanya untuk split train.
  - Implemented in `training.augmentations`.

- [x] Implementasi baseline U-Net.
  - Model menerima tensor 7-channel.
  - Output segmentasi biner flood/non-flood.
  - Simpan checkpoint terbaik berdasarkan validation loss atau validation IoU/F1.
  - Checkpoint terbaik dipilih dengan validation IoU.

- [x] Implementasi ProCANet.
  - Dua encoder sesuai proposal.
  - Progressive cross-attention block pada beberapa level resolusi.
  - Decoder menghasilkan mask segmentasi biner.

- [x] Implementasi loss function.
  - Minimal Dice Loss sesuai proposal.
  - Pastikan masking `label_valid_mask & feature_valid_mask` diterapkan dalam loss.
  - Opsional: BCE + Dice untuk stabilitas training.

- [x] Implementasi konfigurasi training.
  - Optimizer AdamW.
  - Learning rate, batch size, epoch, weight decay, early stopping.
  - Logging train/validation loss dan metrik.
  - Training CLI tersedia di `scripts.train_segmentation`.
  - Spatial CV tersedia lewat `--fold 0..4`; default output menjadi `runs/{architecture}/fold_{k}` saat fold dipakai.
  - Preset hyperparameter tuning grid search (`--tuning-preset grid`) sudah diimplementasikan untuk menjalankan sequential grid search pada 6 hyperparameter combinations.
  - Config tersimpan di `runs/{architecture}/fold_{k}/config.json` untuk mode fold.
  - Metrics tersimpan di `runs/{architecture}/fold_{k}/metrics.csv` for mode fold.

- [x] Implementasi opsi eksperimen label water/river sebagai flood.
  - Default tetap flood-only dari `label_flood_binary`.
  - Jika argumen `--water-river` atau `--water_river` dipakai, target training/validation menjadi union `label_flood_binary | label_water_river_mask`.
  - `water_river_mask` tetap disimpan sebagai auxiliary mask untuk audit.
  - Contoh command:
    ```bash
    uv run python -m scripts.train_segmentation --architecture unet --water-river --amp --gradient-accumulation-steps 2
    ```

- [x] Implementasi kontrol optimasi training tambahan.
  - Learning rate scheduler: `--lr-scheduler reduce-on-plateau` atau `--lr-scheduler none`.
  - Default scheduler: `ReduceLROnPlateau` dengan mode `max` pada validation IoU.
  - Argumen scheduler: `--lr-factor` dan `--lr-patience`.
  - Gradient accumulation tersedia lewat `--gradient-accumulation-steps`.
  - Automatic mixed precision tersedia lewat `--amp`; efektif hanya saat device CUDA.
  - Metrics CSV menambahkan kolom `lr`.

- [x] Training baseline U-Net (Legacy Run).
  - Training sudah dijalankan pada split train.
  - Validasi sudah dijalankan pada split validation.
  - Output ada di `runs/baseline_unet/`.
  - Checkpoint terbaik: `runs/baseline_unet/best.pt`.
  - Log metrik: `runs/baseline_unet/metrics.csv`.
  - Config training: `runs/baseline_unet/config.json`.
  - Config aktual: `epochs=50`, `batch_size=8`, `lr=1e-4`, `weight_decay=1e-4`, `device=cuda`.
  - Best validation IoU: `0.6062394985`.
  - Best validation Dice/F1 pada epoch checkpoint: `0.7548556726`.
  - Early stopping berhenti pada epoch `10`.

- [x] Training ProCANet (Legacy Run).
  - Training sudah dijalankan dengan input dua encoder.
  - Validasi sudah dijalankan pada split validation.
  - Output ada di `runs/procanet/`.
  - Checkpoint terbaik: `runs/procanet/best.pt`.
  - Log metrik: `runs/procanet/metrics.csv`.
  - Config training: `runs/procanet/config.json`.
  - Config aktual: `epochs=50`, `batch_size=4`, `lr=1e-4`, `weight_decay=1e-4`, `device=cuda`.
  - Best validation IoU: `0.6224360219`.
  - Best validation Dice/F1 pada epoch checkpoint: `0.7672857524`.
  - Early stopping berhenti pada epoch `17`.

- [x] Tuning Hyperparameter dengan 5-Fold Cross-Validation & Grid Search.
  - Tuning diselesaikan secara lengkap pada 10 wilayah pelatihan/validasi.
  - Analisis hasil tuning didokumentasikan di `notebooks/hyperparameter_tuning_analysis.ipynb`.
  - Hasil terbaik (mean ± std):
    - **U-Net**: `grid_lr_5e-5_wd_1e-4` (Mean Val IoU: `0.6423`, Mean Val Dice: `0.7711`).
    - **ProCANet**: `grid_lr_1e-4_wd_1e-4` (Mean Val IoU: `0.6531`, Mean Val Dice: `0.7785`).

- [x] Training Model Final (Train + Validation Combined).
  - Menggunakan script `scripts/train_final.py` untuk melatih model gabungan pada 10 wilayah dengan parameter terbaik.
  - Output checkpoint tersimpan sebagai `runs/final/{architecture}/final.pt`.
  - Parameter U-Net: `lr=5e-5`, `epochs=21` (rata-rata CV), `batch_size=8`.
  - Parameter ProCANet: `lr=1e-4`, `epochs=18` (rata-rata CV), `batch_size=8`.

- [x] Implementasi Suite Unit Testing.
  - Mengembangkan 60 unit test di folder `tests/` untuk memvalidasi fungsionalitas pipeline preprocessing, dataset loaders, arsitektur model, masking loss, dan loop pelatihan.
  - Seluruh pengujian berjalan sukses (`OK`).

- [x] Visualisasi kurva training awal.
  - Grafik tersimpan di `runs/training_curves.png`.
  - Kurva ini merangkum hasil training baseline U-Net dan ProCANet yang sudah ada di `runs/`.

- [x] Evaluasi model pada split test.
  - Inferensi U-Net dan ProCANet pada test set wilayah **Aceh Utara** menggunakan checkpoint final model (`final.pt`).
  - Hasil Evaluasi Akhir:
    - **U-Net**: IoU `85.10%` (`0.85099`), Dice/F1 `91.95%` (`0.91950`), Akurasi `97.07%` (`0.97073`) (TP: `16,863,330`, TN: `81,075,762`, FP: `1,227,212`, FN: `1,725,554`).
    - **ProCANet**: IoU `83.83%` (`0.83830`), Dice/F1 `91.20%` (`0.91204`), Akurasi `96.88%` (`0.96875`) (TP: `16,344,540`, TN: `81,394,556`, FP: `908,418`, FN: `2,244,344`).

- [x] Analisis komparatif hasil.
  - Bandingkan U-Net vs ProCANet.
  - Jelaskan apakah cross-attention membantu fusi SAR, optis, dan topografi.
  - Analisis false positive/false negative pada wilayah S2 kosong atau hampir kosong.
  - Hasil ditulis di `runs/final/comparative_analysis.md`.

- [x] Visualisasi hasil segmentasi.
  - Simpan contoh overlay prediksi vs label.
  - Tampilkan perbandingan input Sentinel-1, HSV Sentinel-2, DEMNAS, label, prediksi U-Net, dan prediksi ProCANet.
  - Visualisasi plot disimpan di `runs/final/evaluation_plots.png` menggunakan `scripts.visualize_results`.

- [x] Dokumentasi eksperimen.
  - Buat tabel konfigurasi training.
  - Buat tabel hasil evaluasi.
  - Catat hardware, runtime, seed, dan versi dependency.
  - Hasil dicatat di `runs/final/comparative_analysis.md`.

## Belum Dilakukan

- [ ] Revisi narasi label pada proposal/laporan.
  - Proposal awal masih menulis alur "pseudo ground truth" berbasis NDWI, threshold SAR, slope, rule-based fusion, morfologi, dan hand-correction.
  - Narasi final perlu diganti: UNOSAT dipakai sebagai proxy label utama untuk flood segmentation.
  - Jelaskan `FloodExtent_*` sebagai flood positif, `AnalysisExtent_*` sebagai valid mask, dan `WaterExtent_*`/river sebagai auxiliary mask.

- [ ] Revisi Bab 3 (Metodologi) proposal/laporan agar sama dengan implementasi akhir.
  - Sesuaikan sumber label: UNOSAT sebagai proxy label utama.
  - Sesuaikan struktur split aktual (spatial cross-validation).
  - Sesuaikan urutan channel aktual (7-band input).
  - Sesuaikan penanganan wilayah Sentinel-2 kosong/hampir kosong.

- [ ] Penulisan Bab 4 (Hasil dan Analisis/Pembahasan) Laporan TA.
  - Dokumentasikan hasil komparasi performa model final U-Net vs ProCANet pada data uji Aceh Utara.
  - Ulas karakteristik spasial prediksi: ProCANet memiliki false positive (FP) lebih rendah berkat modul cross-attention, tetapi sensitivitas deteksi (FN) sedikit meningkat.
  - Analisis pengaruh noise Sentinel-2 (misal wilayah dengan HSV=0).

- [ ] Penulisan Bab 5 (Kesimpulan dan Saran) Laporan TA.
  - Rangkum performa fusi multi-sensor SAR, optis, dan topografi dalam mendeteksi area banjir.
  - Berikan rekomendasi/saran untuk penelitian selanjutnya (misalnya augmentasi berbasis SAR khusus atau peningkatan kualitas input optis).

- [ ] Pembersihan dan pengarsipan data repositori.
  - Pindahkan atau cadangkan file zip besar (`procanet_old.zip`, `procanet_runs.zip`) di direktori `runs/` untuk menghemat ruang penyimpanan.

## Catatan Penting

- `WaterExtent_*` dan river mask tidak otomatis dianggap flood positif secara default. Repo tetap memisahkan flood label dari water/river auxiliary mask, tetapi eksperimen union label bisa dijalankan eksplisit dengan `--water-river`.
- Training dan evaluasi harus memakai `label_valid_mask & feature_valid_mask`; piksel di luar area valid UNOSAT atau feature coverage tidak boleh dihitung sebagai benar/salah.
- Wilayah dengan Sentinel-2 kosong/hampir kosong tetap masuk dataset. Analisis hasil perlu mencatat risiko model belajar artefak `HSV=0`.
- Proposal awal memakai istilah "pseudo ground truth" berbasis rule-based fusion, tetapi narasi final sekarang harus memakai UNOSAT sebagai proxy label.
