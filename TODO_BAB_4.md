# Checklist Penulisan BAB 4: Hasil dan Pembahasan (Revisi Struktur)

Dokumen ini berisi panduan dan checklist terperinci untuk menulis **BAB 4: Hasil dan Pembahasan** pada Laporan Tugas Akhir Anda. Struktur ini telah direstrukturisasi agar patuh pada panduan penulisan di mana penjelasan metodologi teknis dipindahkan ke Bab 3, menyisakan hasil numerik, visual, dan penafsiran ilmiah yang analitis di Bab 4.

---

## 4.1 Karakteristik Data Masukan dan Keterbatasannya
Subbab ini menyajikan hasil akhir preprocessing data masukan dan membahas keterbatasan data akibat awan tropis secara jujur.

- [x] **4.1.1 Tabel Ringkasan Statistik Preprocessing Citra**
  - [x] Sajikan tabel resume statistik preprocessing per wilayah (menggunakan data dari [dataset/preprocessing_summary.csv](file:///home/nozomi/Productive/skripsi/dataset/preprocessing_summary.csv)).
  - [x] Jelaskan kualitas data Sentinel-2 bebas awan secara geografis (wilayah bebas awan vs wilayah awan tropis permanen).
  - [x] Bahas konsekuensi fisis dari wilayah dengan data Sentinel-2 yang kosong atau hampir kosong (Aceh Tamiang: 0,0008%, Agam: 0,0000%, Langsa: 0,0000%, Pasaman Barat: 0,0035%). Jelaskan pentingnya mempertahankan data ini untuk menguji ketahanan fusi sensor model.
  - [x] Bahas konsistensi cakupan validitas label target UNOSAT (mencakup >99.8% wilayah studi).

---

## 4.2 Hasil Pembagian Dataset Spasial
Subbab ini membahas hasil pemotongan citra (*tiling*) serta pembagian data untuk uji coba model secara geografis untuk menghindari kebocoran data.

- [x] **4.2.1 Sebaran Tile Hasil Pemotongan Spasial**
  - [x] Sajikan tabel sebaran jumlah tile (positif vs latar belakang/background) dan akumulasi piksel hasil pemotongan.
  - [x] Jelaskan kebijakan **Spatial Cross-Validation**: Aceh Utara dikunci sebagai data uji independen (*final test holdout* - 493 tile: 332 positif, 161 background), dan 10 wilayah lainnya dibagi menjadi 5-fold cross-validation (3930 tile).
  - [x] Bahas secara fisik mengapa akumulasi piksel pada tingkat tile bertambah dibandingkan piksel riil akibat stride overlap 256 piksel (50% overlap).
  - [x] Bahas penerapan **reduksi ubin latar belakang** secara detail (saringan validitas area studi $\ge 70\%$, penyeimbangan rasio 1:1 antara positif dan latar belakang, serta pemilihan deterministik berbasis koordinat untuk memastikan reproduksibilitas).

---

## 4.3 Hasil Pelatihan dan Evaluasi Model
Subbab ini memaparkan hasil numerik optimasi model, baik selama proses tuning hyperparameter maupun saat pengujian akhir.

- [ ] **4.3.1 Hasil Tuning Hyperparameter (Grid Search)**
  - [x] Sajikan tabel evaluasi 6 kombinasi parameter ($lr \in \{1\times10^{-4}, 5\times10^{-5}, 1\times10^{-5}\}$ dan $wd \in \{1\times10^{-4}, 1\times10^{-5}\}$) untuk U-Net dan ProCANet.
  - [x] Bahas dinamika pengaruh *Learning Rate* dan *Weight Decay* ($L_2$ regularization) terhadap skor IoU dan loss validasi.
  - [x] Identifikasi konfigurasi parameter optimal: U-Net (`grid_lr_5e-5_wd_1e-4` dengan Mean Val IoU 0,6423) dan ProCANet (`grid_lr_1e-4_wd_1e-4` dengan Mean Val IoU 0,6531).
  - [x] Hubungkan temuan parameter optimal dengan literatur luar (misalnya Nemni dkk., 2020) dan kemukakan keterbatasan ruang Grid Search secara jujur.
  - [ ] Tampilkan grafik kurva pelatihan (training vs validation loss/metrics) untuk merangkum stabilitas pelatihan (rujukan gambar: [runs/training_curves.png](file:///home/nozomi/Productive/skripsi/runs/training_curves.png)).

- [ ] **4.3.2 Performa Model pada Wilayah Pengujian (Aceh Utara)**
  - [ ] Sajikan tabel evaluasi akhir perbandingan performa U-Net vs ProCANet pada data uji Aceh Utara menggunakan model final (`final.pt`).
  - [ ] Tampilkan confusion matrix lengkap (nilai TP, TN, FP, FN) serta metrik akurasi, IoU, Dice/F1-Score.
  - [ ] Bahas perbandingan kuantitatif secara detail (misalnya U-Net memiliki IoU sedikit lebih tinggi +1.27% dan mendeteksi lebih banyak piksel banjir/sensitivitas tinggi).

---

## 4.4 Pembahasan Kinerja Segmentasi Spasial
Subbab ini merupakan bagian utama pembahasan yang menafsirkan implikasi fisis hasil prediksi secara spasial dan membandingkannya dengan teori/penelitian terdahulu.

- [ ] **4.4.1 Perbandingan Efektivitas U-Net vs ProCANet**
  - [ ] Bahas implikasi modul *Progressive Cross-Attention* (PCAB) pada ProCANet yang berhasil menekan nilai *False Positive* secara signifikan dibandingkan U-Net (908K vs 1.23M piksel), membuktikan efektivitas fusi fitur antar-sensor yang selektif.
  - [ ] Bahas kelemahan ProCANet yang menghasilkan *False Negative* lebih tinggi (kecenderungan melewatkan banjir tipis akibat penyaringan atensi yang terlalu ketat).
  - [ ] Tampilkan analisis visual perbandingan spasial 2D menggunakan gambar visualisasi hasil prediksi (citra masukan, ground truth, prediksi U-Net vs ProCANet).

- [ ] **4.4.2 Ketahanan Model Terhadap Kondisi Ekstrem**
  - [ ] **Studi Kasus Tutupan Awan Tebal / S2 Kosong**: Analisis bagaimana model memprediksi daerah bernilai $HSV=0$. Jelaskan bahwa model berhasil mengandalkan fitur radar Sentinel-1 dan topografi DEMNAS secara efektif tanpa ketergantungan pada data optis.
  - [ ] **Studi Kasus Bayangan Radar (Radar Shadow)**: Bahas peran penting fitur kemiringan lereng (*Slope*) dan HAND dalam mengeliminasi *False Positive* pada area berlereng curam/pegunungan yang biasanya memicu pantulan radar rendah mirip genangan air.
  - [ ] **Studi Kasus Badan Air Permanen**: Bahas peran masker pembantu (`label_water_river_mask.tif`) dalam mencegah terjadinya bias deteksi pada daerah aliran sungai permanen.
  - [ ] **Komparasi Literatur**: Sisipkan perbandingan hasil temuan segmentasi spasial Anda dengan literatur sejenis terdahulu untuk memenuhi syarat wajib pembahasan.
