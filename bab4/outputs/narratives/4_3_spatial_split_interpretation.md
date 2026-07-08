
    Pembagian eksperimen dibuat ulang dari definisi fold di `scripts.preprocessing_utils`.
    Aceh_Utara dikunci sebagai final test region, sedangkan sepuluh wilayah lain membentuk
    5-fold spatial cross-validation. Skema ini menghindari spatial leakage yang dapat terjadi
    jika tile dari wilayah yang sama dibagi acak ke train dan test.
