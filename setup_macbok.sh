#!/bin/bash

echo "🔧 [SETUP] Membuat Conda Environment smart_class_env ..."
conda env create -f environment.yml

echo "✅ Environment berhasil dibuat!"
echo "🔁 Mengaktifkan environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate smart_class_env

echo "📦 Instalasi library tambahan dari requirements.txt (jika ada)..."
pip install -r requirements.txt

echo "🚀 Selesai! Anda siap menjalankan project Smart Class."
