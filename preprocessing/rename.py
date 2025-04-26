import os
import shutil

# Peta angka → label emosi
LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "anger",
    "7": "neutral"
}

def copy_and_rename(src_root, dst_root):
    for split in ['train', 'test']:
        src_split = os.path.join(src_root, split)
        dst_split = os.path.join(dst_root, split)

        if not os.path.exists(src_split):
            print(f"⚠️ Skip: Folder tidak ditemukan → {src_split}")
            continue

        os.makedirs(dst_split, exist_ok=True)

        for old_label, new_label in LABEL_MAP.items():
            src_label_path = os.path.join(src_split, old_label)
            dst_label_path = os.path.join(dst_split, new_label)

            if not os.path.exists(src_label_path):
                print(f"⚠️ Skip: Label '{old_label}' tidak ditemukan di '{split}'")
                continue

            if os.path.exists(dst_label_path):
                print(f"✅ Sudah ada: {dst_label_path}, lewati")
                continue

            shutil.copytree(src_label_path, dst_label_path)
            print(f"✅ Copy: {src_label_path} → {dst_label_path}")

def main():
    src = r"D:\Bunda\smart_class_research\dataset_cropped_yolo\RAF-DB"
    dst = r"D:\Bunda\smart_class_research\dataset_cropped_yolo\rafdb"
    print(f"🔁 Menyalin dataset dari:\n{src}\n→\n{dst}")
    copy_and_rename(src, dst)
    print("\n🎉 Proses selesai!")

if __name__ == "__main__":
    main()
