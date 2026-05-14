# TODO Skripsi

Sumber: `5003221164_Proposal TA Final.docx`, terutama Bab 3 Metode Penelitian.
Status dicek terhadap isi repo pada 2026-05-14.

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
  - UNOSAT dipakai sebagai proxy label utama menggantikan pseudo-label rule-based proposal awal.

- [x] Dataset tile siap training U-Net sudah dibuat.
  - Output ada di `dataset/tiles/7ch/{train,val,test}/`.
  - Ukuran tile: `512 x 512`.
  - Total tile saat dicek: `1121`.
  - Split README saat ini: train `983`, validation `36`, test `102`.

- [x] Dataset siap ProCANet dua encoder sudah dibuat.
  - Output ada di `dataset/tiles/procanet/{train,val,test}/`.
  - Encoder 1: `VV`, `VH`, `Hue`, `Saturation`, `Value`, `Slope`, `HAND`.
  - Encoder 2: `VV`, `VH`.
  - Desain ini mengikuti pola paper ProCANet: encoder utama membawa konteks lengkap, encoder kedua mengulang modalitas air paling informatif. Untuk kasus ini dipilih SAR karena lebih stabil saat awan/hujan dibanding HSV.
  - Total tile sama dengan dataset 7-channel: train `983`, validation `36`, test `102`.

- [x] Ringkasan dan verifikasi preprocessing sudah tersedia.
  - `dataset/preprocessing_summary.csv`
  - `dataset/feature_preprocessing_summary.csv`
  - `dataset/preprocessing_verification_report.csv`
  - `dataset/preprocessing_todo.md`
  - `README.md`

## Belum Dilakukan

- [ ] Revisi narasi label pada proposal/laporan.
  - Proposal awal masih menulis alur "pseudo ground truth" berbasis NDWI, threshold SAR, slope, rule-based fusion, morfologi, dan hand-correction.
  - Narasi final perlu diganti: UNOSAT dipakai sebagai proxy label utama untuk flood segmentation.
  - Jelaskan `FloodExtent_*` sebagai flood positif, `AnalysisExtent_*` sebagai valid mask, dan `WaterExtent_*`/river sebagai auxiliary mask.

- [ ] Implementasi dataset loader training.
  - Loader U-Net perlu membaca `.npz` tile dari `dataset/tiles/7ch/`.
  - Loader ProCANet perlu membaca `.npz` tile dari `dataset/tiles/procanet/`.
  - Loader perlu mengembalikan feature tensor, `y`, `valid_mask`, dan metadata.
  - Loss dan metrik wajib mengabaikan piksel dengan `valid_mask=0`.

- [ ] Implementasi augmentasi data dinamis.
  - Rotasi 90/180/270 derajat.
  - Flip horizontal dan vertikal.
  - Augmentasi hanya untuk split train.

- [ ] Implementasi baseline U-Net.
  - Model menerima tensor 7-channel.
  - Output segmentasi biner flood/non-flood.
  - Simpan checkpoint terbaik berdasarkan validation loss atau validation IoU/F1.

- [ ] Implementasi ProCANet.
  - Dua encoder sesuai proposal.
  - Progressive cross-attention block pada beberapa level resolusi.
  - Decoder menghasilkan mask segmentasi biner.

- [ ] Implementasi loss function.
  - Minimal Dice Loss sesuai proposal.
  - Pastikan masking `label_valid_mask` diterapkan dalam loss.
  - Opsional: BCE + Dice untuk stabilitas training.

- [ ] Implementasi konfigurasi training.
  - Optimizer AdamW.
  - Learning rate, batch size, epoch, weight decay, early stopping.
  - Logging train/validation loss dan metrik.

- [ ] Training baseline U-Net.
  - Jalankan training pada split train.
  - Validasi pada split validation.
  - Simpan bobot terbaik dan grafik/log training.

- [ ] Training ProCANet.
  - Jalankan training dengan input dua encoder.
  - Validasi pada split validation.
  - Simpan bobot terbaik dan grafik/log training.

- [ ] Evaluasi model pada split test.
  - Inferensi U-Net dan ProCANet pada test set.
  - Hitung confusion matrix piksel: TP, FP, TN, FN.
  - Hitung IoU.
  - Hitung F1-score/Dice.
  - Hitung akurasi bila tetap dipakai sesuai proposal.

- [ ] Analisis komparatif hasil.
  - Bandingkan U-Net vs ProCANet.
  - Jelaskan apakah cross-attention membantu fusi SAR, optis, dan topografi.
  - Analisis false positive/false negative pada wilayah S2 kosong atau hampir kosong.

- [ ] Visualisasi hasil segmentasi.
  - Simpan contoh overlay prediksi vs label.
  - Tampilkan perbandingan input Sentinel-1, HSV Sentinel-2, DEMNAS, label, prediksi U-Net, dan prediksi ProCANet.

- [ ] Dokumentasi eksperimen.
  - Buat tabel konfigurasi training.
  - Buat tabel hasil evaluasi.
  - Catat hardware, runtime, seed, dan versi dependency.

- [ ] Revisi Bab 3 proposal/laporan agar sama dengan implementasi akhir.
  - Sesuaikan sumber label: UNOSAT sebagai proxy label utama.
  - Sesuaikan struktur split aktual.
  - Sesuaikan urutan channel aktual.
  - Sesuaikan penanganan wilayah Sentinel-2 kosong/hampir kosong.

## Catatan Penting

- `WaterExtent_*` dan river mask tidak boleh otomatis dianggap flood positif jika targetnya "banjir". Repo saat ini sudah memisahkan flood label dari water/river auxiliary mask.
- Training dan evaluasi harus memakai `label_valid_mask`; piksel di luar area valid tidak boleh dihitung sebagai benar/salah.
- Wilayah dengan Sentinel-2 kosong/hampir kosong tetap masuk dataset. Analisis hasil perlu mencatat risiko model belajar artefak `HSV=0`.
- Proposal awal memakai istilah "pseudo ground truth" berbasis rule-based fusion, tetapi narasi final sekarang harus memakai UNOSAT sebagai proxy label.
