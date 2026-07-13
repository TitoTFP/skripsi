Evaluasi wilayah uji Aceh_Utara menggunakan mosaik piksel unik. Probabilitas dari
seluruh tile yang bertumpang tindih terlebih dahulu dirata-ratakan pada GeoTIFF, kemudian
diberi ambang 0.5. TP, TN, FP, dan FN dihitung hanya satu kali untuk setiap
piksel dalam `effective_valid_mask`. IoU, Dice/F1, accuracy, precision, recall, dan specificity
seluruhnya diturunkan dari confusion matrix mosaik yang sama. Kolom
`loss_tile_rata_rata_batch` tetap berasal dari `metrics.csv` dan dipertahankan sebagai loss
rata-rata per batch tile, sehingga unitnya berbeda dari metrik klasifikasi piksel unik.
ProCANet unggul pada IoU, Dice, accuracy, precision, dan specificity, sedangkan U-Net unggul
tipis pada recall dan memiliki FN lebih rendah. Model dengan IoU tertinggi pada wilayah uji adalah ProCANet dengan IoU 0.853908.
