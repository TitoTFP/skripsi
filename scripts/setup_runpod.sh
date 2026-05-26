#!/bin/bash
# ==============================================================================
# All-In-One RunPod Setup & Training Script (Skripsi Flood Segmentation)
# ==============================================================================
# Cara Penggunaan:
# 1. Jalankan GPU Pod di RunPod (disarankan pakai Template PyTorch).
# 2. Upload file `kaggle.json` Anda ke folder `/workspace` via Jupyter Lab.
# 3. Buka script ini, isi `RUNPOD_API_KEY` Anda di bagian konfigurasi di bawah.
# 4. Jalankan script ini di background agar bisa ditinggal tidur:
#    nohup bash scripts/setup_runpod.sh > runpod_job.log 2>&1 &
# 5. Anda bisa keluar dari browser. Pantau log kapan saja dengan:
#    tail -f runpod_job.log
# ==============================================================================

# ==============================================================================
# CONFIGURATION (Silakan sesuaikan variabel di bawah ini)
# ==============================================================================
KAGGLE_DATASET="titofauzanputra/skripsi-tiles"   # Slug dataset Kaggle Anda
RUNPOD_API_KEY=""                                # Masukkan API Key RunPod Anda untuk auto-stop

# Parameter Training
ARCHITECTURE="procanet"                          # "unet" atau "procanet"
EPOCHS=100                                       # Jumlah epoch training
BATCH_SIZE=16                                    # Batch size (16 cocok untuk VRAM 24GB)
NUM_WORKERS=4                                    # Jumlah worker CPU untuk dataloader
TUNING_PRESET="grid"                             # "none" atau "grid" (grid untuk 6 kombinasi lr/wd)
EARLY_STOPPING_PATIENCE=10                       # Kesabaran early stopping
OUTPUT_DIR="./runs/procanet/"                    # Folder output hasil training
# ==============================================================================

set -e

echo "========================================="
echo "  All-In-One RunPod Pipeline Started"
echo "  Time: $(date)"
echo "========================================="

# Resolusi Pod ID (RunPod biasanya menyediakan env $RUNPOD_POD_ID)
POD_ID=${RUNPOD_POD_ID:-$(hostname)}

# Fungsi untuk mematikan Pod RunPod secara otomatis
stop_runpod_pod() {
    echo "========================================="
    echo "  Initiating Auto-Shutdown Procedure..."
    echo "========================================="
    
    if [ -z "$RUNPOD_API_KEY" ]; then
        echo "❌ [Auto-Shutdown] RUNPOD_API_KEY kosong! Pod tidak bisa dimatikan secara otomatis."
        echo "Silakan matikan Pod Anda secara manual lewat dashboard RunPod."
        return
    fi
    
    if [ -z "$POD_ID" ]; then
        echo "❌ [Auto-Shutdown] Gagal mendeteksi Pod ID. Pod tidak bisa dimatikan secara otomatis."
        return
    fi
    
    echo "[Auto-Shutdown] Mengirim request stop untuk Pod ID: $POD_ID..."
    # Kirim query GraphQL ke API RunPod untuk mematikan Pod
    RESPONSE=$(curl -s -X POST \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"mutation { stopPod(input: { podId: \\\"$POD_ID\\\" }) { id desiredStatus } }\"}" \
      "https://api.runpod.io/v2/user?apikey=$RUNPOD_API_KEY")
      
    echo "[Auto-Shutdown] Response dari RunPod API: $RESPONSE"
}

# Trap untuk memastikan Auto-Shutdown tetap berjalan meskipun training sukses maupun error/crash
cleanup_and_shutdown() {
    # Beri jeda 5 detik agar penulisan log selesai
    sleep 5
    stop_runpod_pod
}
trap cleanup_and_shutdown EXIT


# ------------------------------------------------------------------------------
# 1. Install uv (Astral) untuk instalasi package super cepat
# ------------------------------------------------------------------------------
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "[1/5] uv already installed."
fi

# ------------------------------------------------------------------------------
# 2. Install dependencies (Menjaga PyTorch CUDA bawaan RunPod tetap utuh)
# ------------------------------------------------------------------------------
echo "[2/5] Installing dependencies via uv..."
uv pip install --system kaggle whitebox earthengine-api tqdm numpy

# ------------------------------------------------------------------------------
# 3. Setup Kredensial Kaggle API
# ------------------------------------------------------------------------------
echo "[3/5] Setting up Kaggle API credentials..."
mkdir -p ~/.kaggle

if [ -f "/workspace/kaggle.json" ]; then
    cp /workspace/kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✔ Kredensial Kaggle disalin dari /workspace/kaggle.json"
elif [ -f "./kaggle.json" ]; then
    cp ./kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    echo "✔ Kredensial Kaggle disalin dari folder saat ini"
elif [ -f ~/.kaggle/kaggle.json ]; then
    chmod 600 ~/.kaggle/kaggle.json
    echo "✔ Kredensial Kaggle sudah dikonfigurasi di ~/.kaggle/"
else
    echo "❌ ERROR: kaggle.json tidak ditemukan!"
    echo "Harap upload file kaggle.json Anda ke /workspace/ lalu jalankan ulang script ini."
    exit 1
fi

# ------------------------------------------------------------------------------
# 4. Download & Ekstrak Dataset dari Kaggle
# ------------------------------------------------------------------------------
echo "[4/5] Downloading dataset: $KAGGLE_DATASET..."
mkdir -p dataset/tiles

# Download dataset
kaggle datasets download -d "$KAGGLE_DATASET" -p dataset/

ZIP_FILE=$(find dataset/ -maxdepth 1 -name "*.zip" | head -n 1)

if [ -n "$ZIP_FILE" ]; then
    echo "Meng-ekstrak dataset: $ZIP_FILE..."
    mkdir -p dataset/temp_extract
    unzip -q "$ZIP_FILE" -d dataset/temp_extract
    
    # Deteksi struktur dataset kaggle Anda (tiles_7ch/7ch dan tiles_procanet/procanet)
    if [ -d "dataset/temp_extract/tiles_7ch/7ch" ] || [ -d "dataset/temp_extract/tiles_procanet/procanet" ]; then
        echo "Struktur terdeteksi: tiles_7ch/7ch & tiles_procanet/procanet"
        if [ -d "dataset/temp_extract/tiles_7ch/7ch" ]; then
            mv dataset/temp_extract/tiles_7ch/7ch dataset/tiles/
        fi
        if [ -d "dataset/temp_extract/tiles_procanet/procanet" ]; then
            mv dataset/temp_extract/tiles_procanet/procanet dataset/tiles/
        fi
    elif [ -d "dataset/temp_extract/7ch" ] || [ -d "dataset/temp_extract/procanet" ]; then
        mv dataset/temp_extract/* dataset/tiles/
    elif [ -d "dataset/temp_extract/tiles/7ch" ] || [ -d "dataset/temp_extract/tiles/procanet" ]; then
        mv dataset/temp_extract/tiles/* dataset/tiles/
    else
        mv dataset/temp_extract/* dataset/tiles/
    fi
    
    rm -rf dataset/temp_extract
    rm "$ZIP_FILE"
    echo "✔ Ekstraksi dataset selesai ke dataset/tiles/"
else
    echo "❌ ERROR: File zip dataset tidak ditemukan!"
    exit 1
fi

# ------------------------------------------------------------------------------
# 5. Verifikasi GPU & Memulai Training
# ------------------------------------------------------------------------------
echo "[5/5] Verifikasi GPU & Memulai Training..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "-----------------------------------------"
echo "Memulai training: $ARCHITECTURE (Fold: ALL, Tuning: $TUNING_PRESET)"
echo "Output Directory: $OUTPUT_DIR"
echo "-----------------------------------------"

# Jalankan training loop utama
uv run python -m scripts.train_segmentation \
      --architecture "$ARCHITECTURE" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --amp \
      --num-workers "$NUM_WORKERS" \
      --device cuda \
      --output-dir "$OUTPUT_DIR" \
      --fold all \
      --tuning-preset "$TUNING_PRESET" \
      --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
      --tile-root dataset/tiles

echo "========================================="
echo "  Training Selesai dengan Sukses!"
echo "  Time: $(date)"
echo "========================================="
