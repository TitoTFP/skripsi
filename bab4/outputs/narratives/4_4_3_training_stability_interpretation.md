
    Kurva stabilitas training dibuat dari lima `metrics.csv` spatial cross-validation pada
    konfigurasi hyperparameter dengan mean best validation IoU tertinggi untuk masing-masing
    model. U-Net memakai lr=5e-5 dan wd=1e-4; ProCANet memakai lr=1e-4 dan wd=1e-4. Pemilihan dan kurva ini tidak memakai metrik evaluasi Aceh
    Utara; wilayah tersebut tetap menjadi data uji independen pada tahap evaluasi akhir.

    Pada setiap epoch, loss, IoU, Dice, dan learning rate diringkas sebagai mean serta pita
    plus/minus satu sample standard deviation dari fold yang masih aktif. Early stopping
    menyebabkan jumlah fold aktif dapat berkurang; garis putus-putus pada sumbu kanan panel
    learning rate menunjukkan nilai n tersebut. Tidak ada nilai yang diteruskan setelah suatu
    fold berhenti. Mean dan pita standard deviation dihentikan saat n kurang dari dua, sehingga
    bagian akhir tidak diklaim sebagai agregasi lima fold.
