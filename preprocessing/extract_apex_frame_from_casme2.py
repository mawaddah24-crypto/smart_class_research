import os
import shutil
import pandas as pd

# Konfigurasi
SOURCE_CROPPED = "../Cropped"  # Lokasi folder Cropped/ CASME2
EXCEL_FILE = "../CASME2-coding-20140508.xlsx"  # File Excel coding CASME2
OUTPUT_DIR = "../CASME2_Apex_Formatted"  # Folder tujuan setelah ekstraksi

# Fungsi utama
def extract_apex_frames():
    df = pd.read_excel(EXCEL_FILE)

    # Pastikan kolom penting ada
    required_cols = ['Subject', 'Filename', 'ApexFrame', 'Estimated Emotion']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Kolom '{col}' tidak ditemukan dalam file Excel!")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_success = 0
    total_fail = 0

    for idx, row in df.iterrows():
        subject = row['Subject']
        filename = row['Filename']
        apex_frame = int(row['ApexFrame'])
        emotion = str(row['Estimated Emotion']).lower()

        src_folder = os.path.join(SOURCE_CROPPED, subject, filename)
        if not os.path.exists(src_folder):
            print(f"⚠️ Folder tidak ditemukan: {src_folder}")
            total_fail += 1
            continue

        # Biasanya frame di-folder berformat img{index}.jpg
        # Namun bisa img1.jpg, img001.jpg, dsb, jadi kita cari file yang cocok
        candidates = os.listdir(src_folder)
        matched = None
        for img_file in candidates:
            img_number = ''.join(filter(str.isdigit, img_file))
            if img_number and int(img_number) == apex_frame:
                matched = img_file
                break

        if matched is None:
            print(f"❌ Frame Apex {apex_frame} tidak ditemukan di {src_folder}")
            total_fail += 1
            continue

        # Tentukan folder tujuan berdasarkan emosi
        dst_label_dir = os.path.join(OUTPUT_DIR, emotion)
        os.makedirs(dst_label_dir, exist_ok=True)

        # Salin file
        src_img_path = os.path.join(src_folder, matched)
        dst_img_filename = f"{subject}_{filename}_apex.jpg"
        dst_img_path = os.path.join(dst_label_dir, dst_img_filename)

        shutil.copyfile(src_img_path, dst_img_path)
        print(f"✅ {src_img_path} → {dst_img_path}")

        total_success += 1

    print(f"\n🎉 Selesai! Berhasil ekstrak {total_success} gambar. Gagal: {total_fail}.")

if __name__ == "__main__":
    extract_apex_frames()
