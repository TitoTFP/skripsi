
    Kurva stabilitas training dibuat dari `runs/final/*/metrics.csv`, yaitu log final model
    yang dipakai untuk evaluasi BAB 4. Generator tidak menjalankan training ulang; ia hanya
    membaca loss, IoU, dan Dice per epoch dari artefak final yang sudah tersedia.
