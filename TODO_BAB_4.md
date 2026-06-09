# Checklist Penulisan BAB 4: Hasil dan Pembahasan

Dokumen ini berisi panduan dan checklist terperinci untuk menulis **BAB 4: Hasil dan Pembahasan** pada Laporan Tugas Akhir Anda. Struktur ini disusun berdasarkan standar Departemen Statistika FSAD ITS dan disesuaikan dengan hasil eksperimen yang telah dilakukan di repositori.

---

## 4.1 Deskripsi Data dan Hasil Preprocessing

Bagian ini menjelaskan karakteristik citra satelit dan data elevasi setelah melalui tahap preprocessing.

- [x] **4.1.1 Hasil Preprocessing Citra Sentinel-1 (SAR)**
  - [x] Jelaskan proses normalisasi nilai backscatter (VV dan VH) dari skala desibel (dB) yang di-clip ke rentang `[-30, 0] dB` menjadi `[0, 1]`.
  - [x] Tampilkan contoh visualisasi citra Sentinel-1 (polarisasi VV dan VH) sebelum dan sesudah normalisasi (rujukan data: [dataset/features_preprocessed/](file:///home/nozomi/Productive/skripsi/dataset/features_preprocessed/)).

- [x] **4.1.2 Hasil Preprocessing Citra Sentinel-2 (Optis)**
  - [x] Jelaskan transformasi ruang warna dari pseudo-RGB (`R=B12`, `G=B8`, `B=B4`) menjadi HSV (`Hue`, `Saturation`, `Value`) untuk mempertajam kontras spektral air.
  - [x] Bahas penanganan tutupan awan menggunakan mask validitas Sentinel-2 (`s2_valid_mask.tif`).
  - [x] **Bahas Risiko/Keterbatasan Data**: Sebutkan bahwa terdapat 4 wilayah dengan data Sentinel-2 kosong atau hampir kosong akibat tutupan awan permanen:
    - **Aceh Tamiang**: Validitas S2 hanya `0.0004%`
    - **Agam**: Validitas S2 `0.0000%` (kosong)
    - **Langsa**: Validitas S2 `0.0000%` (kosong)
    - **Pasaman Barat**: Validitas S2 `0.0016%`
    *Jelaskan bahwa data ini sengaja dipertahankan agar model belajar mengenali kondisi no-data (HSV=0) dan menguji ketahanan fusi multi-sensor.*


- [x] **4.1.3 Hasil Preprocessing DEMNAS (Topografi)**
  - [x] Jelaskan ekstraksi fitur kemiringan lereng (Slope) yang dinormalisasi `/45` derajat dan Height Above Nearest Drainage (HAND) yang dinormalisasi `/50` meter.
  - [x] Jelaskan urgensi koreksi hidrologis menggunakan algoritma *Least-Cost Breaching* pada DEMNAS sebelum ekstraksi HAND untuk menjaga kontinuitas aliran digital.


- [x] **4.1.4 Tabel Ringkasan Statistik Preprocessing Citra**
  - [x] Sajikan tabel resume statistik preprocessing per wilayah (gunakan data dari [dataset/feature_preprocessing_summary.csv](file:///home/nozomi/Productive/skripsi/dataset/feature_preprocessing_summary.csv) dan [dataset/preprocessing_summary.csv](file:///home/nozomi/Productive/skripsi/dataset/preprocessing_summary.csv)).


---

## 4.2 Pembuatan Label Target (UNOSAT) dan Tiling

Bagian ini membahas hasil rasterisasi label acuan banjir (UNOSAT) dan pembagian dataset spasial.

- [x] **4.2.1 Hasil Rasterisasi Label UNOSAT**
  - [x] Jelaskan pembentukan `label_flood_binary.tif` dari poligon `FloodExtent_*` UNOSAT.
  - [x] Jelaskan pembentukan `label_valid_mask.tif` yang membatasi evaluasi hanya pada area analisis UNOSAT yang beririsan dengan ROI wilayah administratif.
  - [x] Jelaskan pemisahan auxiliary mask `label_water_river_mask.tif` agar badan air permanen tidak bias sebagai area banjir baru.

- [x] **4.2.2 Hasil Pemotongan Citra (Tiling) dan Pembagian Dataset Spasial**
  - [x] Jelaskan pemotongan raster menjadi tile berukuran $512 \times 512$ piksel dengan stride overlap $256$.
  - [x] Bahas kebijakan **Spatial Cross-Validation**:
    - **Aceh Utara** dikunci sebagai data uji independen (*final test holdout*) sebanyak **493 tile** (332 positif, 161 background).
    - **10 wilayah lainnya** dibagi menjadi **5-fold spatial cross-validation** sebanyak **3930 tile** untuk pelatihan dan validasi internal.
  - [x] Tampilkan tabel sebaran jumlah tile dan piksel banjir per wilayah (ambil data dari tabel ringkasan di [README.md](file:///home/nozomi/Productive/skripsi/README.md#L376-L389)).

---

## 4.3 Pelatihan dan Tuning Hyperparameter (5-Fold Spatial CV)

Bagian ini menjabarkan proses pencarian parameter optimal menggunakan Grid Search 5-fold cross-validation.

- [x] **4.3.1 Desain Grid Search**
  - [x] Jelaskan 6 kombinasi hyperparameter yang diuji untuk masing-masing model (kombinasi $lr \in \{1\times10^{-4}, 5\times10^{-5}, 1\times10^{-5}\}$ dan $wd \in \{1\times10^{-4}, 1\times10^{-5}\}$).

- [ ] **4.3.2 Hasil Evaluasi Parameter Optimal**
  - [ ] Tampilkan tabel perbandingan nilai rata-rata *Mean Validation IoU* dan *Mean Validation Dice* untuk seluruh variasi (sumber analisis: [notebooks/hyperparameter_tuning_analysis.ipynb](file:///home/nozomi/Productive/skripsi/notebooks/hyperparameter_tuning_analysis.ipynb)).
  - [ ] Dokumentasikan varian terbaik yang dipilih:
    - **U-Net**: Variasi `grid_lr_5e-5_wd_1e-4` ($lr=5\times10^{-5}, wd=1\times10^{-4}$) dengan Mean Val IoU **0.6423** dan Mean Val Dice **0.7711**.
    - **ProCANet**: Variasi `grid_lr_1e-4_wd_1e-4` ($lr=1\times10^{-4}, wd=1\times10^{-4}$) dengan Mean Val IoU **0.6531** dan Mean Val Dice **0.7785**.
  - [ ] Tampilkan grafik kurva pelatihan (training vs validation loss/metrics) untuk merangkum stabilitas pelatihan (rujukan gambar: [runs/training_curves.png](file:///home/nozomi/Productive/skripsi/runs/training_curves.png)).

---

## 4.4 Pembentukan Model Final

Bagian ini memaparkan spesifikasi model final sebelum diuji ke data holdout.

- [ ] **4.4.1 Parameter Pelatihan Model Final**
  - [ ] Jelaskan bahwa model final dilatih menggunakan seluruh **10 wilayah training + validation gabungan** dengan hyperparameter optimal masing-masing.
  - [ ] Sebutkan durasi epoch optimal berdasarkan rata-rata epoch terbaik saat CV:
    - **U-Net**: 21 epoch ($lr = 5\times10^{-5}$, $wd = 1\times10^{-4}$).
    - **ProCANet**: 18 epoch ($lr = 1\times10^{-4}$, $wd = 1\times10^{-4}$).

---

## 4.5 Evaluasi Performa Model pada Data Uji (Aceh Utara)

Bagian ini menyajikan hasil pengujian independen pada wilayah Aceh Utara menggunakan model final (`final.pt`).

- [ ] **4.5.1 Tabel Confusion Matrix dan Metrik Performa Akhir**
  - [ ] Sajikan tabel perbandingan performa evaluasi akhir (salin tabel dari [runs/final/comparative_analysis.md](file:///home/nozomi/Productive/skripsi/runs/final/comparative_analysis.md#L30-L43)):
    
    | Metrik | Baseline U-Net | ProCANet |
    | :--- | :---: | :---: |
    | **Loss (BCE + Dice)** | **0.6050** | 0.6285 |
    | **IoU** | **85.10%** | 83.83% |
    | **Dice / F1-Score** | **91.95%** | 91.20% |
    | **Akurasi** | **97.07%** | 96.88% |
    | **True Positive (TP)** | **16.863.330** | 16.344.540 |
    | **True Negative (TN)** | 81.075.762 | **81.394.556** |
    | **False Positive (FP)** | 1.227.212 | **908.418** |
    | **False Negative (FN)** | **1.725.554** | 2.244.344 |

- [ ] **4.5.2 Perbandingan Kinerja Kuantitatif**
  - [ ] Bahas bahwa baseline **U-Net** memiliki performa IoU sedikit lebih tinggi (**+1.27%**) dan F1-Score lebih tinggi (**+0.75%**) dibandingkan ProCANet pada data uji Aceh Utara.
  - [ ] Bahas nilai TP dan FN: U-Net mendeteksi lebih banyak piksel banjir (TP lebih tinggi, FN lebih rendah), menunjukkan sensitivitas deteksi yang lebih baik pada wilayah Aceh Utara.

---

## 4.6 Analisis Karakteristik Prediksi Spasial dan Pembahasan (Discussion)

Bagian ini menganalisis implikasi fisis hasil segmentasi secara spasial.

- [ ] **4.6.1 Efektivitas Modul Progressive Cross-Attention (PCAB) ProCANet**
  - [ ] Bahas nilai FP: **ProCANet secara signifikan mengurangi False Positive** dibandingkan U-Net (908K vs 1.23M piksel). Ini membuktikan mekanisme *cross-attention* berhasil menyaring fitur yang saling melengkapi antar-sensor sehingga model lebih konservatif dan presisi tinggi (tidak mudah mengklasifikasikan daratan sebagai banjir).
  - [ ] Bahas kelemahan ProCANet: Karena atensi yang sangat ketat, model cenderung menghasilkan False Negative (FN) yang lebih tinggi (melewatkan area banjir tipis).

- [ ] **4.6.2 Analisis Visual Hasil Segmentasi**
  - [ ] Gunakan gambar visualisasi [runs/final/evaluation_plots.png](file:///home/nozomi/Productive/skripsi/runs/final/evaluation_plots.png) untuk membandingkan secara spasial:
    - Input citra asli (S1 VV/VH, S2 HSV, DEMNAS).
    - Poligon acuan (Ground Truth UNOSAT).
    - Hasil prediksi U-Net vs ProCANet.

- [ ] **4.6.3 Studi Kasus Kondisi Ekstrem (Pembahasan Khusus)**
  - [ ] **Tutupan Awan Tebal / Data Sentinel-2 Kosong**: Analisis bagaimana model memprediksi daerah dengan nilai `HSV = 0`. Jelaskan bahwa model berhasil mengandalkan fitur Sentinel-1 SAR dan DEMNAS untuk melakukan deteksi berkat kombinasi fusi multi-sensor.
  - [ ] **Bayangan Radar (Radar Shadow) di Lereng Curam**: Jelaskan bagaimana integrasi fitur Slope dan HAND secara efektif menekan False Positive pada lereng curam atau pegunungan yang biasanya menghasilkan pantulan radar rendah (gelap) mirip air.
  - [ ] **Vegetasi Basah dan Badan Air Permanen**: Bahas pengaruh `label_water_river_mask.tif` dalam mencegah bias deteksi pada daerah aliran sungai permanen.
