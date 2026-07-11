
    Kurva stabilitas training dibuat dari `metrics.csv` milik checkpoint terbaik spatial CV
    yang tercatat pada metadata evaluasi `cv_best_checkpoint_eval`. U-Net menggunakan fold 0
    dengan lr=5e-5 dan wd=1e-4, sedangkan ProCANet menggunakan fold 0 dengan lr=1e-4 dan
    wd=1e-4. Generator tidak menjalankan training ulang; ia hanya membaca loss, IoU, Dice,
    dan learning rate per epoch dari run checkpoint yang dipakai untuk evaluasi Aceh Utara.
