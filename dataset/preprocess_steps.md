Untuk penelitianmu **Sentinel-1 + Sentinel-2 HSV + DEMNAS → deep learning segmentation**, preprocessing yang diperlukan bisa dibagi menjadi **3 level**: koreksi sensor, penyamaan spasial, lalu penyiapan tensor untuk model. Di proposalmu sendiri, input akhirnya dirancang menjadi **7 channel**: VV, VH, Hue, Saturation, Value, Slope, dan HAND .

## 1. Preprocessing Sentinel-1 SAR

Untuk Sentinel-1, fitur yang dipakai adalah **VV dan VH**. Preprocessing utamanya:

1. **Apply orbit correction**
   Agar posisi citra sesuai lintasan satelit yang sudah dikoreksi.

2. **Thermal noise removal**
   Mengurangi noise instrumen. Ini penting karena area air pada SAR punya backscatter rendah, jadi noise kecil bisa mengganggu deteksi banjir.

3. **Radiometric calibration**
   Mengubah nilai piksel mentah menjadi nilai fisik backscatter, biasanya dalam bentuk `sigma0`.

4. **Terrain correction / orthorectification**
   Mengoreksi distorsi geometri akibat topografi dan sudut pandang radar.

5. **Konversi ke dB**
   Nilai SAR biasanya lebih stabil untuk model setelah dikonversi ke log scale:

   ```text
   sigma0_dB = 10 * log10(sigma0)
   ```

6. **Clipping nilai backscatter**
   Misalnya:

   ```text
   VV/VH: -30 dB sampai 0 dB
   ```

   Ini mengurangi outlier ekstrem.

7. **Normalisasi ke 0–1**
   Agar range SAR sebanding dengan channel lain.

Dalam metode proposalmu, bagian Sentinel-1 disebut memakai produk `COPERNICUS/S1_GRD` mode IW dengan polarisasi VV dan VH, lalu dilakukan clipping backscatter `-30 dB s.d. 0 dB` dan Min-Max normalization ke `[0,1]` .

---

## 2. Preprocessing Sentinel-2 Optical

Untuk Sentinel-2, targetmu bukan raw RGB biasa, tapi **HSV hasil transformasi dari pseudo-RGB**.

Tahapannya:

1. **Pakai Sentinel-2 Surface Reflectance**
   Gunakan `COPERNICUS/S2_SR_HARMONIZED`, bukan TOA, karena sudah berupa surface reflectance. Di proposalmu juga memakai Sentinel-2 Level-2A `S2_SR_HARMONIZED` .

2. **Cloud masking**
   Ini wajib karena awan dan bayangan awan bisa salah terbaca sebagai objek terang/gelap. Dalam proposalmu, cloud masking Sentinel-2 disebut sebagai tahapan penting untuk menghindari false positive akibat awan dan bayangan awan .

3. **Pilih band untuk pseudo-RGB**
   Untuk transformasi HSV, susun dulu pseudo-RGB dari:

   ```text
   R = B12 / SWIR2
   G = B8  / NIR
   B = B4  / Red
   ```

   Proposalmu menjelaskan konstruksi ini: channel merah diisi Band 12/SWIR, hijau Band 8/NIR, dan biru Band 4/Red sebelum ditransformasi ke HSV .

4. **Resampling band 20 m ke 10 m**
   Karena B12/SWIR2 resolusinya 20 m, sedangkan B8 dan B4 10 m, maka B12 perlu di-resample ke 10 m.

5. **Transformasi RGB → HSV**
   Setelah pseudo-RGB terbentuk, ubah menjadi:

   ```text
   H = Hue
   S = Saturation
   V = Value
   ```

6. **Normalisasi HSV**
   Biasanya:

   ```text
   H, S, V → [0, 1]
   ```

---

## 3. Preprocessing DEMNAS

DEMNAS tidak langsung dipakai sebagai elevasi mentah saja. Lebih baik diekstrak menjadi fitur topografi.

Tahapannya:

1. **Download DEMNAS sesuai AOI**
   Dari BIG / Ina-Geoportal.

2. **Hydrological correction**
   Misalnya sink filling atau lebih bagus **breaching** agar aliran air tidak terputus oleh artefak DEM.

3. **Resampling ke 10 m**
   Karena DEMNAS sekitar 8 m, sedangkan target input modelmu 10 m, perlu disamakan resolusinya.

4. **Ekstraksi Slope**
   Slope membantu model tahu apakah suatu area mungkin tergenang. Area curam biasanya kecil kemungkinan menjadi genangan.

5. **Ekstraksi HAND**
   HAND atau *Height Above Nearest Drainage* menunjukkan tinggi relatif piksel terhadap jaringan drainase terdekat. Ini sangat berguna untuk membedakan area dataran banjir dari area tinggi.

6. **Normalisasi Slope dan HAND**
   Misalnya:

   ```text
   Slope → clip 0–30 atau 0–45 derajat → normalize 0–1
   HAND  → clip 0–50 m atau 0–100 m → normalize 0–1
   ```

Dalam proposalmu, DEMNAS direncanakan diekstrak menjadi **Slope dan HAND** sebagai batasan hidrologis untuk mengurangi false positive pada bayangan radar atau daratan tinggi .

---

## 4. Penyamaan CRS, resolusi, dan grid

Ini tahap yang sangat penting sebelum stacking.

Semua layer harus punya:

```text
CRS sama
extent sama
resolusi sama
jumlah baris-kolom sama
alignment pixel sama
```

Untuk penelitianmu, targetnya adalah **10 meter per piksel**. Proposalmu menyebut seluruh data Sentinel-1, Sentinel-2, dan DEMNAS diseragamkan ke resolusi 10 meter agar kompatibel dengan ProCANet dan U-Net .

Output tahap ini:

```text
VV      : H x W
VH      : H x W
Hue     : H x W
Sat     : H x W
Value   : H x W
Slope   : H x W
HAND    : H x W
Mask    : H x W
```

---

## 5. Stacking menjadi feature tensor

Setelah semua channel bersih dan sejajar, gabungkan menjadi tensor:

```text
X = [VV, VH, H, S, V, Slope, HAND]
```

Dimensi:

```text
H x W x 7
```

Untuk PyTorch biasanya diubah menjadi:

```text
7 x H x W
```

Proposalmu menyebut channel input model terdiri dari 7 saluran: **VV, VH, Hue, Saturation, Value, Slope, HAND** .

---

## 6. Pembuatan label / pseudo ground truth

Karena tidak ada ground truth resmi per piksel, preprocessing label juga perlu.

Tahap yang disarankan:

1. **Buat kandidat air dari Sentinel-2**
   Menggunakan NDWI + Otsu thresholding.

2. **Buat kandidat air dari Sentinel-1**
   Menggunakan threshold backscatter VV/VH.

3. **Buat filter topografi dari DEMNAS**
   Misalnya area dengan slope terlalu tinggi dianggap bukan genangan.

4. **Rule-based fusion**
   Contoh logika:

   ```text
   flood = (NDWI_water AND SAR_water)
        OR (NDWI_water AND low_slope)
        OR (SAR_water AND low_slope)
   ```

5. **Morphological cleaning**
   Opening untuk menghapus noise kecil, closing untuk menutup lubang kecil.

6. **Manual correction sebagian**
   Terutama untuk data uji, agar evaluasi tidak sepenuhnya bergantung pada pseudo-label otomatis.

Proposalmu memang merancang pseudo ground truth dengan NDWI, Otsu thresholding, threshold SAR, filter slope, rule-based fusion, operasi morfologi, lalu stratified sampling dan koreksi visual/manual pada sebagian tiles .

---

## 7. Tiling / patch extraction

Citra utuh terlalu besar untuk langsung masuk model, jadi perlu dipotong menjadi patch.

Umumnya:

```text
512 x 512 x 7
```

atau kalau GPU terbatas:

```text
256 x 256 x 7
```

Untuk mask:

```text
512 x 512 x 1
```

Proposalmu menyebut data input dipotong menjadi tiles agar sesuai dengan model ProCANet dan U-Net serta untuk memperbanyak data latih .

---

## 8. Filtering tile kosong

Jangan semua tile langsung dipakai. Banyak tile bisa berisi hanya laut, awan, no-data, atau non-banjir semua.

Sebaiknya lakukan filtering:

```text
Buang tile dengan no-data terlalu banyak
Buang tile yang 100% awan
Buang tile yang 100% background jika terlalu dominan
Pertahankan sebagian background-only tile untuk negative sample
```

Untuk kasus banjir, ini penting karena kelas banjir biasanya minoritas.

---

## 9. Split train/validation/test secara spasial

Jangan split acak per tile secara sembarangan kalau tile saling overlap atau berasal dari AOI yang sama. Bisa terjadi spatial leakage.

Lebih aman:

```text
Train      : beberapa AOI/wilayah
Validation : AOI berbeda atau blok spasial berbeda
Test       : AOI berbeda / tiles hand-corrected
```

Dalam proposalmu, data uji direncanakan dari 10–20% tiles hasil stratified sampling yang telah dikoreksi visual, sedangkan sisanya untuk train dan validation .

---

## 10. Augmentasi data

Untuk segmentation remote sensing, augmentasi aman:

```text
Horizontal flip
Vertical flip
Rotasi 90/180/270 derajat
Random crop
```

Augmentasi yang perlu hati-hati:

```text
Brightness/contrast → hanya untuk Sentinel-2, jangan sembarangan untuk SAR/DEM
Gaussian noise → hati-hati karena SAR sudah punya karakter speckle
Elastic transform → bisa merusak geometri topografi
```

Proposalmu menyebut augmentasi dinamis berupa rotasi acak 90°, 180°, 270° serta flip horizontal dan vertikal .

---

## Pipeline ringkasnya

```text
1. Ambil AOI dan tanggal banjir
2. Ambil Sentinel-1 VV/VH
3. Preprocess S1: orbit, noise removal, calibration, terrain correction, dB, clipping, normalization
4. Ambil Sentinel-2 SR
5. Preprocess S2: cloud mask, pilih B12-B8-B4, resample, transform HSV, normalization
6. Ambil DEMNAS
7. Preprocess DEMNAS: hydrological correction, resample, slope, HAND, normalization
8. Samakan CRS, resolusi, extent, dan grid
9. Stack menjadi 7 channel
10. Buat pseudo ground truth
11. Morphological cleaning
12. Tiling 256/512 px
13. Filter tile kosong/no-data
14. Split train/val/test secara spasial/stratified
15. Augmentasi hanya pada train
16. Simpan ke format siap training
```

## Output akhir yang siap untuk deep learning

Untuk U-Net:

```text
X_train: N x 7 x 512 x 512
y_train: N x 1 x 512 x 512
```

Untuk ProCANet versi dua encoder:

```text
Encoder 1: HSV + Slope + HAND
Shape    : N x 5 x 512 x 512

Encoder 2: VV + VH
Shape    : N x 2 x 512 x 512

Mask:
Shape    : N x 1 x 512 x 512
```

Jadi, preprocessing minimum yang wajib adalah:

```text
SAR correction + dB + clipping + normalization
S2 cloud masking + HSV transformation + normalization
DEMNAS hydrological correction + slope + HAND + normalization
Resampling/alignment semua layer ke 10 m
Stacking 7 channel
Pseudo-label generation
Tiling
Train/val/test split
Augmentation
```

