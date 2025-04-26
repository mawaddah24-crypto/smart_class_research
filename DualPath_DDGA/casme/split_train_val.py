import os
import shutil
import random

def split_dataset(source_dir, output_dir, split_ratio=0.8, seed=42):
    random.seed(seed)

    classes = sorted(os.listdir(source_dir))

    for cls in classes:
        cls_folder = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_folder):
            continue

        samples = os.listdir(cls_folder)
        random.shuffle(samples)

        split_idx = int(len(samples) * split_ratio)
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]

        # Create output folders
        for split_name, split_samples in [('train', train_samples), ('val', val_samples)]:
            split_cls_folder = os.path.join(output_dir, split_name, cls)
            os.makedirs(split_cls_folder, exist_ok=True)

            for sample in split_samples:
                src = os.path.join(cls_folder, sample)
                dst = os.path.join(split_cls_folder, sample)
                shutil.copytree(src, dst)

    print("\n🎉 Dataset successfully split into Train/Val sets!")

if __name__ == "__main__":
    source_dir = "../CASME2_Cropped_YOLO"  # Dataset hasil crop YOLO
    output_dir = "../CASME2_Sequences_YOLO"  # Folder output split
    split_dataset(source_dir, output_dir, split_ratio=0.8)
