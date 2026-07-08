
    Label BAB 4 dibuat ulang dari ringkasan tile dan raster label UNOSAT. `label_flood_binary`
    menjadi target utama, `label_valid_mask` membatasi area loss/evaluasi, sedangkan
    `label_water_river_mask` tetap menjadi auxiliary mask. Distribusi tile positif dan background
    menunjukkan class imbalance sehingga IoU/Dice lebih informatif daripada akurasi tunggal.
