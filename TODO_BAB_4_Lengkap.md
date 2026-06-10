# TODO BAB 4 — Hasil dan Pembahasan

Dokumen ini berisi rancangan lengkap struktur **BAB 4: Hasil dan Pembahasan** untuk laporan tugas akhir segmentasi banjir berbasis fusi Sentinel-1, Sentinel-2 HSV, DEMNAS, U-Net, dan ProCANet. Format disusun dengan prinsip: **isi yang dibahas** diletakkan berdampingan dengan **output visual/tabel yang perlu dibuat**, sehingga setiap subbab langsung memiliki target penulisan dan bukti pendukung.

Prinsip umum penulisan BAB 4:

1. BAB 4 tidak mengulang tahapan metode secara panjang. Tahapan teknis cukup dirujuk secara singkat, lalu fokus pada **hasil**, **pola**, **interpretasi**, dan **implikasi**.
2. Setiap tabel/gambar harus disebut dalam teks sebelum muncul, lalu ditafsirkan setelahnya. Jangan hanya menempel tabel/gambar tanpa pembahasan.
3. Data numerik yang banyak sebaiknya diringkas dalam tabel. Data sederhana cukup dijelaskan dalam narasi.
4. Pembahasan harus menjawab tujuan penelitian: penerapan model U-Net dan ProCANet, serta perbandingan kinerja keduanya.
5. Pembahasan harus jujur terhadap keterbatasan: kualitas Sentinel-2, ketidakpastian label UNOSAT sebagai proxy reference, perbedaan waktu akuisisi, dan potensi spatial bias.

---

## Struktur BAB 4 yang Disarankan

| Subbab | Fokus utama | Output utama |
|---|---|---|
| 4.1 Karakteristik Data Masukan Multisensor | Kualitas data hasil preprocessing SAR, optis, dan topografi | Tabel statistik preprocessing + gambar channel |
| 4.1.1 Statistik Hasil Preprocessing | Ringkasan VV, VH, HSV, Slope, HAND, dan valid mask | Tabel statistik per wilayah + contoh visual |
| 4.1.2 Verifikasi Kesesuaian Grid dan Stacking Multisensor | Bukti stack layer tidak bergeser | Tabel alignment + overlay OpenStreetMap |
| 4.2 Hasil Pembentukan Label UNOSAT dan Dataset Tile | Label target, valid mask, water/river mask, tiling, dan class imbalance | Tabel label + tabel tile + contoh mask |
| 4.3 Pembagian Dataset Spasial dan Validasi Eksperimental | Spatial CV dan final test Aceh Utara | Tabel fold + tabel split |
| 4.4 Hasil Implementasi Model dan Stabilitas Pelatihan | Model yang dibangun dan proses training | Tabel arsitektur + training curve |
| 4.4.1 Hasil Implementasi Arsitektur Model | U-Net dan ProCANet aktual dari repo | Tabel spesifikasi + diagram model |
| 4.4.2 Hasil Tuning Hyperparameter | Grid search learning rate dan weight decay | Tabel grid search |
| 4.4.3 Stabilitas Pelatihan | Kurva loss, IoU, Dice, dan learning rate | Grafik training curve |
| 4.5 Evaluasi Akhir pada Wilayah Uji Aceh Utara | Performa final U-Net vs ProCANet | Tabel metrik + confusion matrix |
| 4.6 Analisis Visual dan Spasial Hasil Segmentasi | Pola prediksi, FP, FN, dan error spasial | Panel prediksi + error map |
| 4.7 Pembahasan Efektivitas U-Net vs ProCANet | Interpretasi arsitektur dan trade-off | Tabel ringkasan perbandingan + pembahasan literatur |
| 4.8 Ketahanan Model pada Kondisi Data Ekstrem | S2 kosong, radar shadow, badan air permanen | Studi kasus visual + tabel ekstrem |
| 4.9 Ringkasan Temuan Bab 4 | Sintesis hasil utama sebelum BAB 5 | Narasi ringkas, tabel mini opsional |

---

# 4.1 Karakteristik Data Masukan Multisensor

Tujuan subbab ini adalah menunjukkan kondisi data setelah preprocessing dan menjelaskan tantangan utama pada data. Subbab ini harus membuat pembaca paham bahwa segmentasi banjir tidak hanya bergantung pada model, tetapi juga pada kualitas dan keselarasan data multisensor.

## 4.1.1 Statistik Hasil Preprocessing

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Sajikan ringkasan hasil preprocessing Sentinel-1 VV/VH: rentang nilai sebelum clipping, clipping -30 dB sampai 0 dB, hasil normalisasi [0,1], serta pola umum VV dan VH. | **Tabel wajib**: statistik VV dan VH per wilayah, berisi min, max, mean, std, persentase piksel valid. **Gambar disarankan**: contoh VV dan VH ternormalisasi pada wilayah representatif, misalnya Aceh_Utara. |
| Jelaskan karakteristik Sentinel-1 untuk deteksi banjir: air cenderung gelap karena backscatter rendah, tetapi radar shadow dan permukaan halus non-air juga dapat tampak gelap. | **Narasi + gambar contoh**. Tidak perlu tabel tambahan. Jika memungkinkan, beri anotasi pada area gelap yang berpotensi ambigu. |
| Sajikan ringkasan hasil preprocessing Sentinel-2: pseudo-RGB SWIR2-NIR-Red, transformasi ke HSV, serta validitas piksel setelah cloud masking. | **Tabel wajib**: persentase `s2_valid_mask` per wilayah. **Gambar disarankan**: pseudo-RGB dan HSV untuk satu wilayah dengan Sentinel-2 valid dan satu wilayah dengan Sentinel-2 kosong/hampir kosong. |
| Bahas wilayah dengan Sentinel-2 kosong atau hampir kosong, seperti Aceh_Tamiang, Agam, Langsa, dan Pasaman_Barat. Jelaskan bahwa kondisi ini menjadi skenario uji ketahanan model terhadap kondisi tropis, bukan hanya kelemahan data. | **Tabel wajib**: wilayah, jumlah piksel valid Sentinel-2, persentase valid, status kualitas data optis. Status bisa berupa: baik, rendah, hampir kosong, kosong. **Narasi analitis** untuk implikasinya terhadap fusi SAR-topografi. |
| Sajikan hasil preprocessing DEMNAS menjadi Slope dan HAND. Jelaskan bahwa Slope membantu mengurangi prediksi banjir pada lereng curam, sedangkan HAND memberi konteks ketinggian relatif terhadap drainase. | **Tabel wajib**: statistik Slope dan HAND per wilayah, berisi min, max, mean, std. **Gambar disarankan**: Slope dan HAND pada wilayah representatif. |
| Hubungkan kualitas data masukan dengan kebutuhan fusi multisensor. Sentinel-1 kuat pada kondisi awan, Sentinel-2 kaya spektral saat valid, dan DEMNAS memberi batasan fisik/hidrologis. | **Narasi sintesis saja**. Letakkan di paragraf akhir 4.1.1. |

Checklist 4.1.1:

- [ ] Membuat tabel statistik Sentinel-1 VV/VH.
- [ ] Membuat tabel persentase valid Sentinel-2 per wilayah.
- [ ] Membuat tabel statistik Slope dan HAND.
- [ ] Membuat visual contoh channel VV, VH, HSV/pseudo-RGB, Slope, dan HAND.
- [ ] Menjelaskan wilayah dengan Sentinel-2 kosong/hampir kosong.
- [ ] Menutup subbab dengan argumen bahwa fusi multisensor diperlukan karena setiap sensor memiliki keterbatasan.

Data/berkas yang bisa dipakai:

- `dataset/feature_preprocessing_summary.csv`
- `dataset/features_preprocessed/<region>/vv_norm.tif`
- `dataset/features_preprocessed/<region>/vh_norm.tif`
- `dataset/features_preprocessed/<region>/hue.tif`
- `dataset/features_preprocessed/<region>/saturation.tif`
- `dataset/features_preprocessed/<region>/value.tif`
- `dataset/features_preprocessed/<region>/slope_norm.tif`
- `dataset/features_preprocessed/<region>/hand_norm.tif`
- `dataset/features_preprocessed/<region>/s2_valid_mask.tif`
- `dataset/features_preprocessed/<region>/feature_valid_mask.tif`

Contoh kalimat pembuka:

> Tahap preprocessing menghasilkan tujuh saluran fitur utama, yaitu VV, VH, Hue, Saturation, Value, Slope, dan HAND. Evaluasi awal terhadap saluran-saluran tersebut diperlukan untuk memastikan bahwa setiap fitur memiliki rentang nilai yang seragam, status validitas yang jelas, serta karakteristik spasial yang relevan terhadap segmentasi banjir.

---

## 4.1.2 Verifikasi Kesesuaian Grid dan Stacking Multisensor

Tujuan subbab ini adalah membuktikan bahwa proses stacking multisensor tidak menyebabkan pergeseran spasial antar-layer. Ini penting karena model hanya menerima tensor piksel. Jika VV, VH, HSV, Slope, HAND, dan label tidak berada pada grid yang sama, model akan mempelajari hubungan spasial yang salah.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Jelaskan bahwa seluruh layer fitur diselaraskan ke grid referensi Sentinel-1. Artinya setiap piksel pada VV, VH, HSV, Slope, dan HAND merepresentasikan lokasi geografis yang sama. | **Tabel wajib**: hasil verifikasi grid raster multisensor. Kolom: wilayah, raster referensi, ukuran raster, resolusi piksel, CRS/proyeksi, geotransform match, status alignment. |
| Buktikan bahwa setiap layer pada `stack_7ch.tif` tidak mengalami pergeseran. Pemeriksaan dilakukan dengan memastikan kesamaan dimensi raster, geotransform, dan proyeksi antar-layer. | **Tabel teknis disarankan**: layer, ukuran raster, resolusi, CRS, geotransform, projection, status. Bisa dibuat untuk satu wilayah representatif atau semua wilayah. |
| Sajikan overlay OpenStreetMap dengan layer-layer hasil preprocessing. OSM digunakan sebagai validasi visual bahwa sungai, garis pantai, jalan, dan permukiman tidak bergeser terhadap pola citra. | **Gambar wajib**: panel overlay OSM + VV, OSM + VH, OSM + HSV/pseudo-RGB, OSM + Slope, OSM + HAND, OSM + label UNOSAT. Pilih wilayah representatif seperti Aceh_Utara. |
| Tampilkan contoh satu tile yang sama dari seluruh channel. Jika stacking benar, batas sungai, jalan, dataran rendah, dan pola topografi harus berada pada posisi yang konsisten di seluruh panel. | **Gambar panel disarankan**: VV, VH, pseudo-RGB/HSV, Slope, HAND, label UNOSAT, OSM outline. |
| Jelaskan konsekuensi jika stacking bergeser: model dapat menghubungkan label banjir dengan sinyal sensor yang salah lokasi, sehingga prediksi menjadi tidak akurat. | **Narasi analitis saja**. Letakkan setelah tabel dan gambar verifikasi. |
| Tarik kesimpulan bahwa stack multisensor layak digunakan sebagai input model karena tidak ditemukan pergeseran spasial antar-layer. | **Narasi sintesis saja** di akhir subbab. |

Checklist 4.1.2:

- [ ] Membuat tabel verifikasi alignment raster per wilayah.
- [ ] Memastikan seluruh layer memiliki dimensi, CRS, dan geotransform yang sama dengan Sentinel-1.
- [ ] Membuat visual overlay OpenStreetMap terhadap VV, VH, HSV, Slope, HAND, dan label UNOSAT.
- [ ] Membuat satu panel contoh tile multi-layer dari lokasi yang sama.
- [ ] Menjelaskan bahwa OSM adalah bukti visual pendukung, sedangkan bukti utama alignment adalah kesamaan grid/geotransform/proyeksi.
- [ ] Menyimpulkan bahwa `stack_7ch.tif` aman digunakan sebagai input model.

Contoh format tabel:

**Tabel 4.x Hasil Verifikasi Grid Raster Multisensor**

| Wilayah | Raster referensi | Ukuran raster | Resolusi | CRS/proyeksi | Geotransform | Status |
|---|---|---:|---:|---|---|---|
| Aceh_Utara | Sentinel-1 | H × W | 10 m | Sama | Sama | Selaras |
| Pidie | Sentinel-1 | H × W | 10 m | Sama | Sama | Selaras |
| Pidie_Jaya | Sentinel-1 | H × W | 10 m | Sama | Sama | Selaras |
| ... | ... | ... | ... | ... | ... | ... |

**Tabel 4.x Hasil Verifikasi Layer pada `stack_7ch.tif`**

| Band | Nama layer | Sumber | Ukuran raster | Geotransform | CRS/proyeksi | Status |
|---:|---|---|---:|---|---|---|
| 1 | VV | Sentinel-1 | H × W | Sama | Sama | Selaras |
| 2 | VH | Sentinel-1 | H × W | Sama | Sama | Selaras |
| 3 | Hue | Sentinel-2 | H × W | Sama | Sama | Selaras |
| 4 | Saturation | Sentinel-2 | H × W | Sama | Sama | Selaras |
| 5 | Value | Sentinel-2 | H × W | Sama | Sama | Selaras |
| 6 | Slope | DEMNAS | H × W | Sama | Sama | Selaras |
| 7 | HAND | DEMNAS | H × W | Sama | Sama | Selaras |

Contoh format gambar:

**Gambar 4.x Verifikasi overlay OpenStreetMap terhadap stack multisensor pada wilayah Aceh Utara**

Panel:

- (a) OSM + VV
- (b) OSM + VH
- (c) OSM + pseudo-RGB/HSV
- (d) OSM + Slope
- (e) OSM + HAND
- (f) OSM + label UNOSAT

Contoh narasi:

> Berdasarkan Tabel 4.x, seluruh raster hasil preprocessing memiliki ukuran, sistem koordinat, geotransform, dan resolusi yang sama dengan raster referensi Sentinel-1. Hal ini menunjukkan bahwa proses stacking tidak mengubah posisi spasial antar-layer. Secara visual, Gambar 4.x juga memperlihatkan bahwa fitur sungai, garis pantai, dan jaringan jalan pada OpenStreetMap berada pada posisi yang konsisten terhadap pola gelap Sentinel-1, struktur HSV Sentinel-2, serta pola topografi Slope dan HAND. Dengan demikian, setiap piksel pada tensor 7-channel dapat dianggap merepresentasikan lokasi geografis yang sama.

---

# 4.2 Hasil Pembentukan Label UNOSAT dan Dataset Tile

Tujuan subbab ini adalah menjelaskan kualitas label target dan distribusi dataset yang digunakan untuk pelatihan dan evaluasi. Subbab ini harus meyakinkan pembaca bahwa label tidak dibuat secara sembarangan dan bahwa class imbalance ditangani secara eksplisit.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Jelaskan hasil rasterisasi UNOSAT menjadi `label_flood_binary`, `label_valid_mask`, dan `label_water_river_mask`. Tegaskan bahwa label banjir utama berasal dari `FloodExtent`, sedangkan `WaterExtent` dan `River` hanya auxiliary mask. | **Gambar wajib**: panel label banjir, valid mask, dan water/river mask pada satu wilayah. **Narasi wajib** untuk menjelaskan perbedaan fungsi ketiga mask. |
| Sajikan sebaran piksel banjir UNOSAT per wilayah. Bahas wilayah dengan cakupan banjir terbesar dan terkecil. | **Tabel wajib**: wilayah, jumlah piksel valid, jumlah piksel banjir, persentase banjir terhadap area valid, jumlah piksel water/river. |
| Bahas class imbalance antara piksel banjir dan non-banjir. Jelaskan mengapa IoU dan Dice/F1 lebih informatif daripada akurasi saja. | **Narasi + tabel label**. Tidak perlu visual baru, kecuali ingin menambahkan bar chart persentase banjir per wilayah. |
| Sajikan hasil tiling 512 × 512 dengan stride 256. Jelaskan bahwa overlap 50% membuat jumlah sampel tile meningkat dan area yang sama dapat muncul di lebih dari satu tile. | **Tabel wajib**: wilayah, total tile, tile positif, tile background, rasio positif-background. |
| Jelaskan strategi pemilihan tile: tile positif selalu dipertahankan, sedangkan background-only disampling deterministik per wilayah untuk mengurangi dominasi kelas non-banjir. | **Narasi + tabel tile**. Tidak perlu gambar baru. |
| Tunjukkan contoh tile positif dan tile background. | **Gambar opsional tetapi menarik**: contoh tile positif berisi banjir dan contoh tile background-only. Panel minimal berisi VV/HSV/HAND + label. |

Checklist 4.2:

- [ ] Membuat tabel statistik label UNOSAT per wilayah.
- [ ] Membuat visual `label_flood_binary`, `label_valid_mask`, dan `label_water_river_mask`.
- [ ] Membuat tabel jumlah tile per wilayah.
- [ ] Menjelaskan class imbalance di level piksel dan level tile.
- [ ] Menjelaskan bahwa `water_river_mask` bukan label banjir utama.
- [ ] Menjelaskan kenapa valid mask penting untuk loss dan evaluasi.

Data/berkas yang bisa dipakai:

- `dataset/labels_unosat_rasterized/<region>/label_flood_binary.tif`
- `dataset/labels_unosat_rasterized/<region>/label_valid_mask.tif`
- `dataset/labels_unosat_rasterized/<region>/label_water_river_mask.tif`
- `dataset/tiles/<split>/<region>/*.npz`
- `dataset/feature_preprocessing_summary.csv`

Contoh narasi:

> Label target pada penelitian ini diperoleh dari hasil rasterisasi data UNOSAT. Layer `FloodExtent` digunakan sebagai sumber label positif banjir, sedangkan `AnalysisExtent` digunakan untuk membatasi area valid analisis. Sementara itu, layer `WaterExtent` dan `River` tidak digabungkan sebagai label banjir utama, melainkan disimpan sebagai auxiliary mask untuk audit terhadap area air permanen.

---

# 4.3 Pembagian Dataset Spasial dan Validasi Eksperimental

Tujuan subbab ini adalah meyakinkan pembaca bahwa evaluasi tidak mengalami spatial leakage. Pembagian dataset harus dibahas sebagai hasil desain eksperimen, bukan sekadar prosedur teknis.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Jelaskan bahwa Aceh_Utara dikunci sebagai final test region dan tidak digunakan dalam training maupun validation. | **Narasi + tabel split**. |
| Sajikan 5-fold spatial cross-validation untuk 10 wilayah selain Aceh_Utara. | **Tabel wajib**: fold, wilayah validasi, wilayah training. |
| Jelaskan alasan penggunaan spatial split, bukan random split tile. Tekankan risiko spatial leakage jika tile dari wilayah berdekatan masuk ke training dan test sekaligus. | **Narasi analitis saja**. Tidak perlu visual. |
| Tunjukkan sebaran jumlah tile pada train, validation, dan test. | **Tabel wajib**: jumlah tile train/val/test per fold. Jika terlalu panjang, cukup ringkasan total per fold. |
| Jelaskan bahwa skema ini membuat evaluasi lebih realistis karena model diuji pada wilayah yang tidak pernah dilihat sebelumnya. | **Narasi sintesis saja** di akhir subbab. |

Checklist 4.3:

- [ ] Menyebut Aceh_Utara sebagai final test region.
- [ ] Membuat tabel 5-fold spatial cross-validation.
- [ ] Membuat tabel jumlah tile train/val/test per fold.
- [ ] Menjelaskan alasan ilmiah menghindari random split tile.
- [ ] Menjelaskan spatial leakage secara singkat.

Contoh format tabel:

**Tabel 4.x Pembagian 5-Fold Spatial Cross-Validation**

| Fold | Wilayah validasi | Wilayah training | Wilayah test final |
|---:|---|---|---|
| 0 | Pidie, Pidie_Jaya | 8 wilayah CV lainnya | Aceh_Utara |
| 1 | Aceh_Besar, Banda_Aceh | 8 wilayah CV lainnya | Aceh_Utara |
| 2 | Aceh_Tamiang, Aceh_Timur | 8 wilayah CV lainnya | Aceh_Utara |
| 3 | Bireuen, Langsa | 8 wilayah CV lainnya | Aceh_Utara |
| 4 | Agam, Pasaman_Barat | 8 wilayah CV lainnya | Aceh_Utara |

Contoh narasi:

> Pembagian data dilakukan berbasis wilayah untuk menghindari spatial leakage. Pada citra satelit, tile yang berdekatan sering memiliki pola spektral dan tekstur yang sangat mirip. Jika tile dari wilayah yang sama dibagi secara acak ke data latih dan data uji, nilai evaluasi dapat menjadi terlalu optimistis karena model sebenarnya diuji pada pola spasial yang sudah mirip dengan data latih.

---

# 4.4 Hasil Implementasi Model dan Stabilitas Pelatihan

Tujuan subbab ini adalah menunjukkan bahwa model yang dipakai dalam eksperimen benar-benar berhasil dibangun sesuai rancangan dan dilatih secara stabil. Subbab ini dapat dibagi menjadi tiga bagian: implementasi arsitektur, tuning hyperparameter, dan stabilitas pelatihan.

## 4.4.1 Hasil Implementasi Arsitektur Model

Bagian ini tidak perlu mengulang teori arsitektur dari BAB 2. Fokusnya adalah implementasi aktual yang dibuat pada folder `training/models/`.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Jelaskan bahwa dua model berhasil diimplementasikan, yaitu U-Net sebagai baseline dan ProCANet sebagai model utama. | **Narasi pembuka singkat**. Tidak perlu gambar. |
| Sajikan konfigurasi U-Net aktual: input 7 channel, output 1 channel logit, encoder depth 4, base channel 32, bottleneck, decoder, dan skip connection. | **Tabel wajib**: spesifikasi implementasi arsitektur model. Kolom: komponen, U-Net, ProCANet. |
| Sajikan konfigurasi ProCANet aktual: Encoder 1 menerima 7 channel penuh, Encoder 2 menerima VV/VH, PCAB pada skip features dan bottleneck, output 1 channel logit. | **Tabel wajib** yang sama. Pastikan tidak ada lagi keterangan Encoder 1 = 5 channel. |
| Tampilkan diagram implementasi U-Net berdasarkan repo, bukan diagram teori umum dari BAB 2. | **Gambar opsional tapi bagus**: Input 7ch → Encoder → Bottleneck → Decoder → Logit 1ch. |
| Tampilkan diagram implementasi ProCANet berdasarkan repo. | **Gambar sangat disarankan**: Encoder 1 7ch dan Encoder 2 2ch → dual encoder → PCAB skip + bottleneck → decoder → logit 1ch. |
| Verifikasi bahwa output model memiliki ukuran spasial yang sama dengan input. U-Net menerima input Batch × 7 × H × W dan menghasilkan Batch × 1 × H × W. ProCANet menerima Encoder 1 Batch × 7 × H × W dan Encoder 2 Batch × 2 × H × W, lalu menghasilkan Batch × 1 × H × W. | **Tabel wajib**: hasil verifikasi forward pass model. |
| Jelaskan bahwa output berupa logit 1-channel, bukan langsung mask final. Mask biner diperoleh setelah sigmoid dan threshold 0,5 pada tahap inferensi/evaluasi. | **Narasi teknis singkat**. |
| Tarik kesimpulan bahwa kedua arsitektur sudah siap digunakan untuk pelatihan dan evaluasi karena menerima format input sesuai dataset tile dan menghasilkan output segmentasi biner dengan resolusi spasial yang sama. | **Narasi sintesis saja**. |

Checklist 4.4.1:

- [ ] Membuat tabel spesifikasi U-Net dan ProCANet.
- [ ] Membuat diagram U-Net aktual.
- [ ] Membuat diagram ProCANet aktual.
- [ ] Membuat tabel verifikasi forward pass.
- [ ] Menjelaskan output logit dan threshold inferensi.
- [ ] Menjelaskan perbedaan U-Net sebagai fusi langsung dan ProCANet sebagai fusi selektif berbasis attention.

Berkas yang bisa dirujuk:

- `training/models/unet.py`
- `training/models/procanet.py`
- `training/models/blocks.py`
- `training/models/__init__.py`
- `tests/test_models.py`

Contoh format tabel:

**Tabel 4.x Spesifikasi Implementasi Arsitektur U-Net dan ProCANet**

| Komponen | U-Net | ProCANet |
|---|---|---|
| Jenis model | Single encoder-decoder | Dual encoder-decoder dengan Progressive Cross-Attention |
| Input utama | 7 channel: VV, VH, Hue, Saturation, Value, Slope, HAND | Encoder 1: 7 channel penuh; Encoder 2: VV, VH |
| Jumlah output | 1 channel logit | 1 channel logit |
| Base channels | 32 | 32 |
| Kedalaman encoder | 4 level | 4 level pada masing-masing encoder |
| Blok konvolusi | Conv2d 3×3 + GroupNorm + ReLU, dua kali | Conv2d 3×3 + GroupNorm + ReLU, dua kali |
| Mekanisme fusi | Concatenation melalui skip connection | Progressive Cross-Attention pada skip features dan bottleneck |
| Decoder | ConvTranspose2d + skip concatenation + ConvBlock | Decoder dari fitur hasil attention |
| Output akhir | Logit segmentasi banjir | Logit segmentasi banjir |

**Tabel 4.x Hasil Verifikasi Forward Pass Model**

| Model/komponen | Input | Output | Status |
|---|---|---|---|
| U-Net | Batch × 7 × H × W | Batch × 1 × H × W | Sesuai |
| ProCANet Encoder 1 | Batch × 7 × H × W | Fitur encoder multi-level | Sesuai |
| ProCANet Encoder 2 | Batch × 2 × H × W | Fitur encoder multi-level | Sesuai |
| PCAB | Dua fitur Batch × C × H × W | Batch × C × H × W | Shape terjaga |
| ProCANet final | Encoder 1 + Encoder 2 | Batch × 1 × H × W | Sesuai |

Contoh narasi:

> Hasil implementasi menunjukkan bahwa kedua arsitektur telah menyesuaikan format input dataset. U-Net memproses seluruh informasi multisensor melalui satu tensor 7-channel, sedangkan ProCANet memproses informasi melalui dua jalur encoder. Encoder pertama membawa konteks multisensor lengkap, sementara encoder kedua mengulang modalitas SAR VV/VH sebagai sumber informasi yang lebih stabil pada kondisi tutupan awan. Perbedaan ini membuat U-Net bertindak sebagai baseline fusi langsung, sedangkan ProCANet menjadi model fusi selektif berbasis attention.

---

## 4.4.2 Hasil Tuning Hyperparameter

Tujuan subbab ini adalah menunjukkan bahwa konfigurasi model dipilih berdasarkan eksperimen, bukan asumsi.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Sajikan hasil grid search learning rate dan weight decay untuk U-Net. | **Tabel wajib**: varian tuning, learning rate, weight decay, mean validation IoU, mean Dice, mean loss, std IoU. |
| Sajikan hasil grid search learning rate dan weight decay untuk ProCANet. | **Tabel wajib** serupa dengan U-Net. Boleh digabung dengan tabel U-Net jika tidak terlalu panjang. |
| Identifikasi konfigurasi terbaik untuk masing-masing model berdasarkan mean validation IoU. | **Narasi + highlight pada tabel**. Beri bold pada baris terbaik. |
| Bahas pengaruh learning rate. Learning rate terlalu kecil dapat membuat pembelajaran lambat, sedangkan learning rate terlalu besar dapat membuat validasi tidak stabil. | **Narasi analitis + grafik training curve** bila tersedia. |
| Bahas pengaruh weight decay sebagai regularisasi. Jelaskan apakah weight decay lebih tinggi membantu menekan overfitting atau justru menurunkan kemampuan belajar. | **Narasi + tabel grid search**. Tidak perlu visual baru. |
| Jelaskan bahwa konfigurasi terbaik dipilih berdasarkan rata-rata validation IoU di seluruh fold. | **Narasi saja**. |

Checklist 4.4.2:

- [ ] Membuat tabel grid search U-Net.
- [ ] Membuat tabel grid search ProCANet.
- [ ] Menandai konfigurasi terbaik masing-masing model.
- [ ] Membahas efek learning rate.
- [ ] Membahas efek weight decay.
- [ ] Menjelaskan keterbatasan ruang grid search.

Contoh format tabel:

**Tabel 4.x Hasil Grid Search Hyperparameter**

| Model | Varian | Learning rate | Weight decay | Mean Val IoU | Std Val IoU | Mean Val Dice | Mean Val Loss | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| U-Net | grid_lr_1e-4_wd_1e-4 | 1e-4 | 1e-4 | ... | ... | ... | ... |  |
| U-Net | grid_lr_5e-5_wd_1e-4 | 5e-5 | 1e-4 | ... | ... | ... | ... | Terbaik U-Net |
| ProCANet | grid_lr_1e-4_wd_1e-4 | 1e-4 | 1e-4 | ... | ... | ... | ... | Terbaik ProCANet |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 4.4.3 Stabilitas Pelatihan Model

Tujuan subbab ini adalah menunjukkan apakah model belajar secara stabil selama training.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Sajikan kurva training dan validation loss untuk U-Net dan ProCANet. | **Grafik wajib**: training loss vs validation loss. Bisa dibuat satu gambar multi-panel. |
| Sajikan kurva training dan validation IoU untuk U-Net dan ProCANet. | **Grafik wajib**: training IoU vs validation IoU. |
| Sajikan kurva training dan validation Dice/F1 untuk U-Net dan ProCANet. | **Grafik disarankan**: training Dice vs validation Dice. |
| Jelaskan checkpoint terbaik disimpan berdasarkan validation IoU, bukan validation loss. | **Narasi saja**. |
| Jelaskan bahwa loss yang dipakai adalah masked BCE + Dice Loss sehingga hanya piksel valid yang memengaruhi pelatihan. | **Narasi saja**. |
| Bahas tanda-tanda overfitting atau stabilitas. Misalnya jarak train-val yang terlalu jauh, validation IoU stagnan, atau learning rate turun oleh scheduler. | **Narasi analitis berdasarkan grafik**. |

Checklist 4.4.3:

- [ ] Membuat grafik loss training vs validation.
- [ ] Membuat grafik IoU training vs validation.
- [ ] Membuat grafik Dice training vs validation.
- [ ] Menjelaskan best checkpoint berbasis validation IoU.
- [ ] Menjelaskan early stopping berbasis validation IoU.
- [ ] Menjelaskan scheduler ReduceLROnPlateau.
- [ ] Menafsirkan grafik, bukan hanya menampilkan grafik.

Berkas yang bisa dipakai:

- `runs/<model>/fold_*/metrics.csv`
- `runs/final/<model>/metrics.csv`
- `runs/training_curves.png` jika sudah dibuat

Contoh narasi:

> Kurva pelatihan pada Gambar 4.x menunjukkan bahwa nilai training loss menurun seiring bertambahnya epoch, sedangkan validation IoU digunakan sebagai indikator utama pemilihan checkpoint terbaik. Pemilihan validation IoU lebih relevan dibanding validation loss karena tujuan akhir penelitian adalah memperoleh segmentasi spasial yang memiliki tumpang tindih tinggi terhadap label referensi.

---

# 4.5 Evaluasi Akhir Model pada Wilayah Uji Aceh Utara

Tujuan subbab ini adalah menyajikan hasil kuantitatif utama dan menjawab model mana yang lebih baik pada wilayah uji independen.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Sajikan hasil evaluasi akhir U-Net dan ProCANet pada Aceh_Utara menggunakan model final. | **Tabel wajib**: model, loss, IoU, Dice/F1, accuracy. |
| Sajikan confusion matrix piksel untuk kedua model. | **Tabel wajib**: model, TP, TN, FP, FN. Bisa ditambah precision dan recall jika ingin interpretasi lebih kuat. |
| Bandingkan IoU dan Dice/F1. Jelaskan model mana yang memiliki tumpang tindih spasial lebih baik terhadap label UNOSAT. | **Narasi + tabel metrik**. Tidak perlu grafik jika tabel sudah jelas. |
| Bandingkan FP dan FN. Jelaskan apakah model cenderung agresif mendeteksi banjir atau konservatif. | **Bar chart opsional**: FP dan FN U-Net vs ProCANet. Ini menarik karena langsung memperlihatkan trade-off model. |
| Jelaskan posisi akurasi sebagai metrik pelengkap karena kelas non-banjir biasanya dominan. | **Narasi saja**. |
| Tarik kesimpulan sementara: model mana unggul secara angka, dan model mana lebih baik dari sisi karakter kesalahan. | **Narasi sintesis saja** di akhir 4.5. |

Checklist 4.5:

- [ ] Membuat tabel metrik final U-Net vs ProCANet.
- [ ] Membuat tabel confusion matrix piksel.
- [ ] Menambahkan precision dan recall jika memungkinkan.
- [ ] Membahas IoU dan Dice sebagai metrik utama.
- [ ] Membahas akurasi sebagai metrik pelengkap.
- [ ] Membahas FP dan FN sebagai karakter kesalahan model.

Contoh format tabel:

**Tabel 4.x Performa Akhir Model pada Wilayah Uji Aceh_Utara**

| Model | Loss | IoU | Dice/F1 | Accuracy | Interpretasi singkat |
|---|---:|---:|---:|---:|---|
| U-Net | ... | ... | ... | ... | ... |
| ProCANet | ... | ... | ... | ... | ... |

**Tabel 4.x Confusion Matrix Piksel pada Wilayah Uji Aceh_Utara**

| Model | TP | TN | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| U-Net | ... | ... | ... | ... | ... | ... |
| ProCANet | ... | ... | ... | ... | ... | ... |

Contoh narasi:

> Perbandingan confusion matrix menunjukkan karakter kesalahan yang berbeda antara kedua model. Model dengan FP lebih tinggi cenderung lebih agresif dalam mendeteksi area banjir, sedangkan model dengan FN lebih tinggi cenderung lebih konservatif dan berisiko melewatkan area banjir yang tipis atau memiliki respons sensor kurang tegas.

---

# 4.6 Analisis Visual dan Spasial Hasil Segmentasi

Tujuan subbab ini adalah memperlihatkan bagaimana prediksi model bekerja di peta, bukan hanya dalam angka.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Tampilkan satu tile representatif dari Aceh_Utara yang berisi banjir jelas. | **Gambar wajib**: panel 2 × 3 atau 2 × 4 berisi VV, VH/HSV, Slope/HAND, label UNOSAT, prediksi U-Net, prediksi ProCANet. |
| Jelaskan area yang berhasil ditangkap oleh kedua model. Fokus pada bentuk spasial genangan yang sesuai dengan label. | **Narasi interpretatif berdasarkan gambar**. Jangan mengulang isi gambar secara datar. |
| Jelaskan area false positive: wilayah non-banjir yang diprediksi sebagai banjir. Hubungkan dengan kemungkinan backscatter rendah, bayangan radar, tanah basah, atau badan air permanen. | **Gambar error map sangat disarankan**: TP, FP, FN, TN. |
| Jelaskan area false negative: wilayah banjir UNOSAT yang tidak tertangkap model. Hubungkan dengan banjir tipis, label tidak presisi, atau fitur visual yang kurang kontras. | **Gambar error map** atau narasi berdasarkan panel prediksi. |
| Bandingkan pola spasial U-Net dan ProCANet. U-Net mungkin lebih agresif, sedangkan ProCANet mungkin lebih selektif. | **Tabel ringkas opsional**: aspek visual, U-Net, ProCANet. Misalnya cakupan banjir, FP, FN, kehalusan batas, konsistensi dengan label. |
| Jelaskan bahwa hasil visual harus dibaca bersama metrik kuantitatif, bukan sebagai bukti tunggal. | **Narasi saja**. |

Checklist 4.6:

- [ ] Membuat panel visual input-label-prediksi.
- [ ] Membuat error map TP/FP/FN/TN jika memungkinkan.
- [ ] Menjelaskan area yang benar diprediksi kedua model.
- [ ] Menjelaskan area FP.
- [ ] Menjelaskan area FN.
- [ ] Membandingkan kecenderungan spasial U-Net dan ProCANet.

Contoh format gambar:

**Gambar 4.x Perbandingan visual hasil segmentasi pada tile Aceh_Utara**

Panel:

- (a) Sentinel-1 VV
- (b) Sentinel-2 pseudo-RGB/HSV
- (c) DEMNAS Slope/HAND
- (d) Label UNOSAT
- (e) Prediksi U-Net
- (f) Prediksi ProCANet

**Gambar 4.x Error map prediksi model**

Warna/kelas yang disarankan:

- TP: banjir benar terdeteksi
- FP: non-banjir salah diprediksi banjir
- FN: banjir terlewat
- TN: non-banjir benar

Contoh narasi:

> Secara visual, kedua model mampu menangkap pola genangan utama pada area dataran rendah yang berdekatan dengan jaringan aliran. Namun, perbedaan terlihat pada area transisi di tepi genangan. U-Net cenderung menghasilkan cakupan prediksi yang lebih luas, sedangkan ProCANet tampak lebih selektif dalam mempertahankan area yang memiliki dukungan fitur lebih konsisten.

---

# 4.7 Pembahasan Efektivitas U-Net vs ProCANet

Tujuan subbab ini adalah menjawab kontribusi utama penelitian: apakah Progressive Cross-Attention memberi manfaat dibanding baseline U-Net.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Bahas U-Net sebagai baseline fusi langsung 7-channel. Jelaskan kelebihannya: sederhana, stabil, dan mampu memanfaatkan semua channel sekaligus. | **Narasi analitis saja**. |
| Bahas ProCANet sebagai arsitektur dual-encoder dengan Encoder 1 multisensor lengkap dan Encoder 2 SAR VV/VH. Jelaskan potensi attention dalam menyeleksi fitur lintas sensor. | **Diagram arsitektur opsional** jika belum ditampilkan di 4.4.1. Jika sudah ada, cukup narasi. |
| Bandingkan performa berdasarkan metrik akhir dan confusion matrix. | **Tabel ringkasan wajib**: aspek, U-Net, ProCANet, interpretasi. |
| Bahas trade-off FP dan FN. Jika ProCANet menekan FP tetapi menaikkan FN, jelaskan bahwa attention dapat membuat model lebih selektif tetapi berisiko melewatkan banjir tipis. | **Bar chart FP/FN opsional** atau pakai tabel confusion matrix dari 4.5. |
| Jelaskan apakah hasil penelitian mendukung atau berbeda dari paper ProCANet asli. Jangan hanya menyatakan “sesuai penelitian sebelumnya”; jelaskan persamaan dan perbedaannya. | **Narasi pembahasan + rujukan literatur**. Tidak perlu tabel kecuali ingin membuat tabel komparasi literatur. |
| Jelaskan kemungkinan alasan U-Net bisa tetap kompetitif: jumlah data, noise label, konfigurasi input 7-channel, atau domain data yang berbeda dari ProCANet asli. | **Narasi analitis saja**. |
| Simpulkan secara jujur: ProCANet tidak harus dipaksakan menang mutlak; yang penting adalah memahami kapan dan mengapa model itu unggul atau kalah. | **Narasi sintesis saja**. |

Checklist 4.7:

- [ ] Membandingkan U-Net dan ProCANet berdasarkan IoU, Dice, accuracy, FP, FN.
- [ ] Menjelaskan kelebihan U-Net sebagai baseline fusi langsung.
- [ ] Menjelaskan kelebihan ProCANet sebagai fusi attention.
- [ ] Menjelaskan kelemahan masing-masing model.
- [ ] Menghubungkan hasil dengan literatur ProCANet/Feliren et al.
- [ ] Menghindari klaim “ProCANet pasti lebih baik” jika angka tidak mendukung.

Contoh format tabel:

**Tabel 4.x Ringkasan Perbandingan U-Net dan ProCANet**

| Aspek | U-Net | ProCANet | Interpretasi |
|---|---|---|---|
| Strategi fusi | Fusi langsung 7-channel | Dual encoder + cross-attention | ProCANet lebih selektif terhadap fitur |
| IoU | ... | ... | ... |
| Dice/F1 | ... | ... | ... |
| False Positive | ... | ... | ... |
| False Negative | ... | ... | ... |
| Karakter prediksi | Agresif/luas | Selektif/ketat | ... |
| Kelebihan utama | ... | ... | ... |
| Keterbatasan utama | ... | ... | ... |

Contoh narasi:

> Hasil evaluasi menunjukkan bahwa keunggulan arsitektur tidak hanya dapat dibaca dari satu metrik tunggal. U-Net dapat menghasilkan cakupan prediksi yang lebih luas karena seluruh channel digabung sejak awal, sedangkan ProCANet memiliki mekanisme seleksi fitur melalui cross-attention. Mekanisme ini berpotensi menekan prediksi berlebih, tetapi pada kondisi tertentu juga dapat membuat model terlalu konservatif terhadap area banjir yang tipis atau memiliki sinyal sensor yang lemah.

---

# 4.8 Ketahanan Model pada Kondisi Data Ekstrem

Tujuan subbab ini adalah membuat pembahasan lebih kuat secara ilmiah karena membahas reliabilitas model pada kondisi sulit, bukan hanya skor akhir.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Studi kasus Sentinel-2 kosong atau hampir kosong. Jelaskan apakah model masih mampu memprediksi banjir dengan mengandalkan SAR dan DEMNAS. | **Tabel wajib**: wilayah ekstrem, persentase S2 valid, IoU/Dice validasi jika tersedia, catatan performa. **Gambar opsional**: contoh tile HSV=0. |
| Bahas peran SAR pada kondisi awan. Jelaskan bahwa VV/VH menjadi sumber informasi utama saat HSV tidak informatif. | **Narasi + contoh visual VV/VH**. |
| Studi kasus topografi sulit: lereng curam atau radar shadow. Jelaskan apakah Slope dan HAND membantu mengurangi false positive pada area yang secara radar tampak gelap. | **Gambar opsional**: VV + Slope/HAND + prediksi + error map pada area curam. |
| Studi kasus badan air permanen atau sungai. Jelaskan potensi kebingungan antara banjir sementara dan air permanen. | **Gambar opsional**: `water_river_mask` dibandingkan dengan prediksi. |
| Bahas keterbatasan UNOSAT sebagai proxy label. Label mungkin tidak pixel-perfect, ada perbedaan waktu akuisisi citra, dan batas poligon bisa tidak setajam batas genangan aktual. | **Narasi wajib**. Tidak perlu visual. |
| Bahas keterbatasan model: generalisasi masih diuji pada wilayah Sumatra tertentu, bukan seluruh Indonesia; hasil dapat berubah pada wilayah urban padat, hutan lebat, atau banjir sangat dangkal. | **Narasi wajib** di akhir subbab. |

Checklist 4.8:

- [ ] Membuat tabel wilayah dengan S2 valid rendah/kosong.
- [ ] Membuat studi kasus visual wilayah ekstrem.
- [ ] Membahas peran SAR ketika HSV tidak valid.
- [ ] Membahas peran Slope/HAND pada radar shadow atau lereng curam.
- [ ] Membahas potensi bias badan air permanen.
- [ ] Membahas keterbatasan UNOSAT sebagai proxy label.
- [ ] Membahas keterbatasan generalisasi model.

Contoh format tabel:

**Tabel 4.x Studi Kasus Kondisi Data Ekstrem**

| Wilayah | Kondisi ekstrem | Persentase S2 valid | Model yang lebih stabil | Catatan interpretasi |
|---|---|---:|---|---|
| Aceh_Tamiang | Sentinel-2 hampir kosong | ... | ... | Model bergantung pada SAR dan topografi |
| Agam | Sentinel-2 kosong | ... | ... | Topografi kompleks, potensi radar shadow |
| Langsa | Sentinel-2 kosong | ... | ... | ... |
| Pasaman_Barat | Sentinel-2 hampir kosong | ... | ... | ... |

Contoh narasi:

> Kondisi Sentinel-2 yang kosong atau hampir kosong menjadi ujian penting bagi pendekatan fusi multisensor. Pada situasi ini, komponen HSV tidak memberikan informasi optis yang memadai, sehingga model harus mengandalkan Sentinel-1 VV/VH serta fitur topografi Slope dan HAND. Apabila model tetap mampu menghasilkan prediksi yang konsisten, hal ini menunjukkan bahwa arsitektur tidak sepenuhnya bergantung pada sensor optis.

---

# 4.9 Ringkasan Temuan Bab 4

Tujuan subbab ini adalah menutup BAB 4 dengan sintesis yang langsung mengarah ke kesimpulan BAB 5.

| Isi yang perlu dibahas | Output visual/tabel yang disarankan |
|---|---|
| Ringkas kualitas data input: SAR stabil, Sentinel-2 tidak selalu valid, DEMNAS memberi konteks topografi. | **Narasi saja**. |
| Ringkas verifikasi stacking: layer VV, VH, HSV, Slope, HAND, dan label berada pada grid yang sama sehingga layak dijadikan input tensor. | **Narasi saja**. |
| Ringkas dataset dan label: UNOSAT dipakai sebagai proxy label, valid mask membatasi area evaluasi, dan tile dibuat dengan overlap. | **Narasi saja**. |
| Ringkas hasil implementasi model: U-Net dan ProCANet berhasil dibangun sesuai konfigurasi input. | **Narasi saja**. |
| Ringkas hasil pelatihan: konfigurasi terbaik dipilih berdasarkan validation IoU dari spatial cross-validation. | **Narasi saja**. |
| Ringkas evaluasi akhir: model mana yang unggul pada IoU/Dice dan bagaimana trade-off FP/FN. | **Tabel mini opsional**: model terbaik per aspek. |
| Ringkas implikasi: fusi Sentinel-1, Sentinel-2 HSV, dan DEMNAS relevan untuk segmentasi banjir di wilayah tropis, tetapi kualitas label dan kondisi data tetap membatasi interpretasi hasil. | **Narasi penutup saja**. |

Checklist 4.9:

- [ ] Menjawab tujuan penelitian pertama: penerapan U-Net dan ProCANet dengan input fusi multisensor.
- [ ] Menjawab tujuan penelitian kedua: perbandingan performa U-Net dan ProCANet.
- [ ] Menyebut temuan paling penting dari preprocessing, alignment, dataset, training, dan evaluasi.
- [ ] Menyebut keterbatasan utama secara ringkas.
- [ ] Memberi transisi yang jelas menuju BAB 5.

Contoh tabel mini opsional:

**Tabel 4.x Ringkasan Temuan Utama BAB 4**

| Aspek | Temuan utama | Implikasi |
|---|---|---|
| Data input | ... | ... |
| Alignment stack | ... | ... |
| Dataset tile | ... | ... |
| Implementasi model | ... | ... |
| Tuning | ... | ... |
| Evaluasi akhir | ... | ... |
| Ketahanan ekstrem | ... | ... |

Contoh narasi penutup:

> Secara keseluruhan, hasil BAB 4 menunjukkan bahwa pendekatan fusi Sentinel-1, Sentinel-2 HSV, dan DEMNAS dapat diterapkan untuk segmentasi banjir berbasis deep learning. Verifikasi stacking menunjukkan bahwa seluruh layer berada pada grid spasial yang konsisten, sehingga input tensor layak digunakan dalam pelatihan. Evaluasi akhir pada wilayah uji Aceh_Utara memperlihatkan adanya perbedaan karakter antara U-Net dan ProCANet, terutama pada trade-off antara cakupan prediksi, false positive, dan false negative. Temuan ini menjadi dasar untuk menarik kesimpulan pada BAB 5.

---

# Daftar Output Minimal yang Wajib Dibuat

Bagian ini adalah checklist praktis supaya BAB 4 tidak terasa kosong.

| Prioritas | Output | Subbab | Status | Artefak |
|---|---|---:|:---:|---|
| Wajib | Tabel statistik preprocessing VV, VH, HSV, Slope, HAND | 4.1.1 | [x] | `outputs/bab4/tables/4_1_1_preprocessing_stats.csv` |
| Wajib | Tabel persentase `s2_valid_mask` per wilayah | 4.1.1 | [x] | `outputs/bab4/tables/4_1_1_s2_valid_mask_by_region.csv` |
| Wajib | Gambar contoh channel VV, VH, HSV/pseudo-RGB, Slope, HAND | 4.1.1 | [x] | `outputs/bab4/figures/4_1_1_channel_example_aceh_utara.png` |
| Wajib | Tabel verifikasi alignment raster | 4.1.2 | [x] | `outputs/bab4/tables/4_1_2_alignment_verification.csv` |
| Wajib | Gambar overlay OSM dengan layer stack | 4.1.2 | [x] | `outputs/bab4/figures/4_1_2_osm_overlay_stack_aceh_utara.png` |
| Wajib | Tabel statistik label UNOSAT per wilayah | 4.2 | [x] | `outputs/bab4/tables/4_2_label_mask_tile_stats.csv` |
| Wajib | Tabel jumlah tile positif/background per wilayah | 4.2 | [x] | `outputs/bab4/tables/4_2_tile_distribution_by_split_region.csv` |
| Wajib | Tabel 5-fold spatial cross-validation / split spasial | 4.3 | [x] | `outputs/bab4/tables/4_3_spatial_split_summary.csv` |
| Wajib | Tabel arsitektur U-Net dan ProCANet | 4.4.1 | [x] | `outputs/bab4/tables/4_4_1_model_architecture_specs.csv` |
| Wajib | Verifikasi/kontrak forward pass model | 4.4.1 | [x] | Dicakup di `outputs/bab4/tables/4_4_1_model_architecture_specs.csv` |
| Wajib | Tabel/heatmap grid search hyperparameter | 4.4.2 | [x] | `outputs/bab4/tables/4_4_2_hyperparameter_tuning_summary.csv` |
| Wajib | Grafik training/validation loss, IoU, dan Dice | 4.4.3 | [x] | `outputs/bab4/figures/4_4_3_training_curves.png` |
| Wajib | Tabel metrik final U-Net vs ProCANet | 4.5 | [x] | `outputs/bab4/tables/4_5_final_metrics.csv` |
| Wajib | Tabel confusion matrix TP, TN, FP, FN | 4.5 | [x] | `outputs/bab4/tables/4_5_confusion_matrix_pixels.csv` |
| Wajib | Panel visual prediksi U-Net vs ProCANet | 4.6 | [x] | `outputs/bab4/figures/4_6_segmentation_panel_aceh_utara.png` |
| Sangat disarankan | Error map TP/FP/FN/TN | 4.6 | [x] | `outputs/bab4/figures/4_6_error_map_aceh_utara.png` |
| Sangat disarankan | Tabel ringkasan U-Net vs ProCANet | 4.7 | [x] | `outputs/bab4/tables/4_7_unet_vs_procanet_effectiveness_summary.csv` |
| Sangat disarankan | Tabel studi kasus Sentinel-2 kosong/hampir kosong | 4.8 | [x] | `outputs/bab4/tables/4_8_difficult_data_case_studies.csv` |
| Opsional | Tabel mini ringkasan temuan BAB 4 | 4.9 | [x] | `outputs/bab4/tables/4_9_bab4_findings_summary.csv` |

---

# Catatan Penulisan Supaya BAB 4 Terlihat Matang

1. Jangan menulis “model A lebih baik” tanpa menjelaskan **mengapa** dan **dalam aspek apa**.
2. Jangan memakai akurasi sebagai metrik utama. Akurasi hanya pelengkap karena kelas non-banjir biasanya dominan.
3. Jika ProCANet tidak unggul mutlak, tulis sebagai trade-off, bukan kegagalan.
4. Verifikasi alignment perlu ditekankan karena ini adalah fondasi validitas seluruh model.
5. Visual hasil segmentasi harus ditafsirkan: area mana yang benar, area mana yang FP, area mana yang FN, dan apa kemungkinan penyebabnya.
6. Pembahasan literatur tidak perlu mengulang teori panjang. Cukup gunakan penelitian terdahulu untuk menjelaskan apakah hasilmu memperkuat, berbeda, atau memperluas temuan sebelumnya.
7. Keterbatasan harus ditulis eksplisit: UNOSAT adalah proxy label, bukan label lapangan pixel-perfect; Sentinel-2 dapat kosong; waktu akuisisi citra dan label mungkin tidak identik; generalisasi masih terbatas pada wilayah studi.

