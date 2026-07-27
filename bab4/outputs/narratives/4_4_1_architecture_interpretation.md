
    Spesifikasi arsitektur dibuat ulang dari source model di `training/models`.
    U-Net menggunakan satu encoder 7-channel, bottleneck, dan decoder dengan skip connection.
    ProCANet mempertahankan encoder utama 7-channel dan encoder auxiliary 2-channel,
    lalu menggabungkan representasi melalui progressive cross-attention sebelum decoder.
