
    Preprocessing menghasilkan tujuh channel input: VV, VH, Hue, Saturation, Value, Slope, dan HAND.
    Tabel 4.1 sampai Tabel 4.3 dibuat ulang dari raster dan ringkasan dataset, sehingga angka pada folder
    `bab4/outputs` dapat dilacak ke `dataset/features_preprocessed`, `dataset/feature_preprocessing_summary.csv`,
    dan `dataset/preprocessing_summary.csv`. Wilayah dengan Sentinel-2 kosong atau hampir kosong tetap dipertahankan
    sebagai kasus ketahanan model karena kanal SAR dan topografi masih tersedia.
