#!/bin/bash
# ==============================================================================
# RunPod GPU Instance Setup Script for Skripsi Flood Segmentation
# ==============================================================================
# Cara Penggunaan:
# 1. Jalankan GPU Pod di RunPod (disarankan pakai Template PyTorch).
# 2. Upload file `kaggle.json` Anda ke folder `/workspace` via Jupyter Lab.
# 3. Jalankan script ini dari terminal Jupyter:
#    bash scripts/setup_runpod.sh <username-kaggle>/<nama-dataset-kaggle>
# ==============================================================================

set -e

# Ambil argumen dataset slug jika ada
KAGGLE_DATASET=$1

echo "========================================="
echo "  Setting Up RunPod GPU Environment      "
echo "========================================="

# 1. Install uv (Astral) untuk instalasi package super cepat
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "[1/5] uv already installed."
fi

# 2. Install dependencies (Preserve pre-installed PyTorch & CUDA di system)
echo "[2/5] Installing dependencies via uv..."
# Menggunakan --system untuk menginstal ke system python RunPod (menjaga PyTorch CUDA tetap utuh)
uv pip install --system kaggle whitebox earthengine-api tqdm numpy

# 3. Setup Kaggle API credentials
echo "[3/5] Setting up Kaggle API credentials..."
mkdir -p ~/.kaggle

# Coba cari kaggle.json di beberapa lokasi umum
if [ -f "/workspace/kaggle.json" ]; then
    cp /workspace/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✔ Berhasil: kaggle.json disalin dari /workspace/"
elif [ -f "./kaggle.json" ]; then
    cp ./kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✔ Berhasil: kaggle.json disalin dari folder saat ini"
elif [ -f ~/.kaggle/kaggle.json ]; then
    chmod 600 ~/.kaggle/kaggle.json
    echo "✔ Berhasil: kaggle.json sudah ada di ~/.kaggle/"
else
    echo "❌ Peringatan: kaggle.json TIDAK ditemukan!"
    echo "Silakan upload file 'kaggle.json' dari akun Kaggle Anda ke folder /workspace/ via Jupyter Lab,"
    echo "lalu jalankan command berikut secara manual:"
    echo "  cp /workspace/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
fi

# 4. Download & Extract Dataset dari Kaggle
if [ -n "$KAGGLE_DATASET" ]; then
    echo "[4/5] Downloading dataset: $KAGGLE_DATASET..."
    
    # Buat direktori tujuan
    mkdir -p dataset/tiles
    
    # Download file zip dataset ke temporary folder
    echo "Memulai download dari Kaggle..."
    kaggle datasets download -d "$KAGGLE_DATASET" -p dataset/
    
    # Cari nama file zip hasil download
    ZIP_FILE=$(find dataset/ -maxdepth 1 -name "*.zip" | head -n 1)
    
    if [ -n "$ZIP_FILE" ]; then
        echo "Meng-ekstrak dataset: $ZIP_FILE..."
        
        # Ekstrak dataset. Kita perlu memastikan format folder ter-ekstrak dengan benar.
        # Biasanya dataset di-upload sebagai folder 'tiles' atau langsung berisi '7ch' / 'procanet'.
        # Kita ekstrak dulu ke folder temporer untuk mendeteksi strukturnya.
        mkdir -p dataset/temp_extract
        unzip -q "$ZIP_FILE" -d dataset/temp_extract
        
        # Deteksi struktur folder hasil ekstrak
        if [ -d "dataset/temp_extract/7ch" ] || [ -d "dataset/temp_extract/procanet" ]; then
            echo "Struktur: Berisi folder 7ch / procanet langsung."
            mv dataset/temp_extract/* dataset/tiles/
        elif [ -d "dataset/temp_extract/tiles/7ch" ] || [ -d "dataset/temp_extract/tiles/procanet" ]; then
            echo "Struktur: Berisi folder tiles/7ch / tiles/procanet."
            mv dataset/temp_extract/tiles/* dataset/tiles/
        else
            echo "Struktur tidak biasa. Memindahkan semua isi langsung ke dataset/tiles/."
            mv dataset/temp_extract/* dataset/tiles/
        fi
        
        # Bersihkan file sampah
        rm -rf dataset/temp_extract
        rm "$ZIP_FILE"
        echo "✔ Dataset berhasil di-ekstrak ke dataset/tiles/"
    else
        echo "❌ Gagal mengunduh file zip dataset. Periksa koneksi internet atau hak akses dataset Kaggle Anda."
    fi
else
    echo "[4/5] Skip download dataset karena argumen dataset slug kosong."
    echo "Untuk mendownload secara manual nanti, jalankan perintah:"
    echo "  kaggle datasets download -d <username>/<dataset-slug> -p dataset/"
    echo "  unzip dataset/*.zip -d dataset/tiles/ && rm dataset/*.zip"
fi

# 5. Verifikasi Dataset & Lingkungan
echo "[5/5] Memverifikasi setup..."
echo "--- Python & CUDA Info ---"
python -c "import torch; print('Python version:', torch.__sys__.version.split()[0]); print('PyTorch version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
echo "--------------------------"

echo ""
echo "=== Setup Selesai! ==="
echo "Untuk mulai melatih model baseline U-Net (Fold 0), jalankan perintah:"
echo "  uv run python -m scripts.train_segmentation --architecture unet --fold 0 --epochs 50 --batch-size 8 --amp"
echo ""
echo "Untuk melatih semua fold (5-fold Cross Validation) dengan U-Net baseline:"
echo "  uv run python -m scripts.train_segmentation --architecture unet --fold all --epochs 50 --batch-size 8 --amp"
echo "========================================="
