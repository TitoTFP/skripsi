
    Kasus data ekstrem dipilih ulang dari tile `.npz` asli, meliputi Sentinel-2 kosong/hampir kosong, topografi/radar shadow, badan air permanen. Statistik pada
    Tabel 4.16 dan 4.17 memperlihatkan bahwa kualitas Sentinel-2, dominasi badan air, dan
    konfigurasi topografi dapat mempengaruhi interpretasi visual. Generator tidak menggunakan
    gambar lama; setiap panel dibentuk langsung dari channel input, label, dan mask tile.
