# 🎓 Smart Class Research — DualPath FER Model

Proyek ini mengembangkan model pengenalan ekspresi wajah siswa dalam kelas dengan pendekatan **DualPath Attention-Based Network**. Arsitektur memisahkan fitur **lokal** dan **global**, yang kemudian digabungkan secara adaptif untuk meningkatkan akurasi deteksi ekspresi dalam lingkungan kelas yang padat.

---

## ✅ Fitur Model

- 🔄 Baseline DualPath
- 🧠 DualPath with Partial Attention
- 🧬 DualPath with Modified Fusion + Semantic Enhancements
- 🔁 Auto Resume Training
- 💾 Save Top-K Checkpoints
- 📊 Auto Logging + Summary Extract
- 📈 Accuracy Curve Visualization
- 🔁 Weighted Ensemble Inference

---

## 🚀 Setup Project

### 1. Clone Repository

```bash
git clone https://github.com/AmirDev83/smart_class_research.git
cd smart_class_research
```

### 2. Install Environment (Anaconda)

```bash
conda env create -f environment.yml
conda activate smart_class_env
```

### 3. Atau jalankan langsung :

### Mac/Linux :

Buka Terminal
Pindah ke folder project
Jalankan

```bash
chmod +x setup_project.sh
setup_macbok.sh
```

### Windows :

```bash
setup_project.bat
```

---

## 📁 Struktur Folder

```
smart_class_research/
├── train.py
├── DualPathModel.py
├── module.py
├── FocalLoss.py
├── logs/               # Output logs + checkpoint
├── weights/            # Pretrained EfficientViT
├── requirements.txt
├── environment.yml
├── README.md
└── train_dualpath.bat
```

---

## 🧠 Arsitektur Model: DualPath Overview

```text
         ┌────────────┐
         │  Input RGB │
         └────┬───────┘
              ▼
     ┌────────────────────┐
     │ EfficientViT Backbone │
     └────────┬───────────┘
              │
    ┌─────────┴───────────┐
    │                     │
    ▼                     ▼
Local Pathway        Global Pathway
(PRA / PRA-MHA)      (APP / AGAPP)
    │                     │
    ▼                     ▼
SE Block / DSE       SE Block / DSE
    │                     │
    └────────┬────────────┘
             ▼
         DDGA Fusion
             │
             ▼
      SE Fusion Block
             │
             ▼
     Global Average Pool
             │
             ▼
         Classifier (FC)
             │
             ▼
        Expression Output
```

---

### 🔍 Komponen Utama

| Komponen        | Deskripsi                                                   |
| --------------- | ----------------------------------------------------------- |
| EfficientViT    | Backbone ringan efisien pretrained di VGGFace2              |
| PRA / PRA-MHA   | Jalur fitur lokal, fokus pada area ekspresi mikro           |
| APP / AGAPP     | Jalur global, menyerap konteks wajah menyeluruh             |
| DSE Blocks      | SE Block adaptif berdasarkan posisi spasial                 |
| DDGA            | Dynamic Dual-Gate Attention untuk penggabungan local-global |
| SE Fusion Block | Refine akhir sebelum klasifikasi                            |
| Ensemble Top-K  | Kombinasi 3–5 checkpoint terbaik saat inference real-world  |

---

## 🧪 Dataset dan Evaluasi

Model dilatih dan diuji menggunakan dan simpan pada folder dataset:

- 📁 RAF-DB
- 📁 FER2013
- 🎥 Video real-classroom dengan 30–50 siswa per frame

---

## 📈 Training (Buka folder DualPath_DDGA)

Single model:

```bash
python train.py --model baseline --dataset RAF-DB
```

Batch training di Mac semua model :

```bash
train_dualpath.sh
```

Batch training Win semua model:

```bash
train_dualpath.bat
```

---

## 🤖 Inference (Ensemble)

Menggunakan 3–5 checkpoint terbaik:

```bash
python ensemble_weighted_inference.py
```

---

## 📦 Dataset

Karena keterbatasan ukuran file GitHub (maks 100MB), file dataset tidak disertakan di repository ini.

Silakan unduh secara manual dari sumber berikut:

- [✔️ RAF-DB](https://www.whdeng.cn/RAF/model1.html)
- [✔️ FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [✔️ AffectNet (via permintaan resmi)](http://mohammadmahoor.com/affectnet/)

## ✍️ Penulis

**Mawaddah Harahap**
**Amir Mahmud Husein, M.Kom**

---

## 📄 Lisensi

Lisensi open-source (MIT / Apache 2.0) dapat ditambahkan sesuai kebutuhan.
