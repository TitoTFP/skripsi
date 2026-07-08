
    Ringkasan tuning dihitung ulang dari setiap `metrics.csv` pada fold spatial CV.
    Nilai yang dipakai adalah epoch dengan validation IoU tertinggi per fold, lalu dirata-ratakan
    per kombinasi learning rate dan weight decay. Kombinasi terbaik pada rekap generator adalah ProCANet lr=1e-4 wd=1e-4 dengan mean best validation IoU 0.6531.
