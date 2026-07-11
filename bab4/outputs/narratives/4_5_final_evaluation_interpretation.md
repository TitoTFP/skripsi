
    Evaluasi wilayah uji dihitung ulang dari `cv_best_checkpoint_eval/*/eval_test/metrics.csv`
    untuk region Aceh_Utara. Checkpoint ini dipilih dari hasil spatial cross-validation.
    Precision dan recall diturunkan kembali dari TP, FP, dan FN sehingga tabel metrik dan
    confusion matrix memiliki sumber numerik yang sama. ProCANet unggul pada loss, IoU, Dice,
    akurasi, precision, dan specificity, sedangkan U-Net unggul tipis pada recall dan memiliki
    FN lebih rendah. Model dengan IoU tertinggi pada wilayah uji adalah ProCANet dengan IoU 0.853774.
