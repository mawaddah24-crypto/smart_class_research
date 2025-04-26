import os
import csv
import argparse
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from tqdm import tqdm

from DualPathModel import (DualPath_Baseline, 
                           DualPath_Baseline_DSE,
                           DualPath_Base_Partial, 
                           DualPath_PartialAttentionModif,
                           DualPath_PartialAttentionSAP)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # Ubah ke RGB jika perlu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def normalize(name):
    return name.lower().replace('-', '').replace(' ', '')

def get_checkpoint_path(model_name, dataset_name):
    folder = f"{normalize(model_name)}_{normalize(dataset_name)}"
    filename = f"{model_name.upper()}_{dataset_name.upper()}_best.pt"
    return os.path.join("logs", folder, filename)

def classify_image(image_path, model):
    try:
        image = Image.open(image_path).convert('RGB')
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        print(f"❗ Gagal membuka: {image_path} | Error: {e}")
        return None
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        prob = F.softmax(output, dim=1).cpu().numpy().flatten()
    return [round(p, 4) for p in prob]

def process_dataset(model_name, dataset_name, dataset_path, checkpoint_path, output_file):
    MODEL_FACTORY = {
    "baseline": DualPath_Baseline,
    "base_dse": DualPath_Baseline_DSE,
    "partial": DualPath_Base_Partial,
    "partialmodif": DualPath_PartialAttentionModif,
    "semantic": DualPath_PartialAttentionSAP
    }

    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model type '{model_name}'")

    model = MODEL_FACTORY[model_name]().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    results = []
    total_success, total_fail = 0, 0

    if not os.path.exists(dataset_path):
        print(f"❌ Path tidak ditemukan: {dataset_path}")
        return

    label_folders = [f for f in os.listdir(dataset_path)
                     if os.path.isdir(os.path.join(dataset_path, f))]

    for label in label_folders:
        folder_path = os.path.join(dataset_path, label)
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n📂 Label '{label}': {len(image_files)} gambar")
        for img_file in tqdm(image_files, desc=f"🔎 {label}"):
            img_path = os.path.join(folder_path, img_file)
            scores = classify_image(img_path, model)
            if scores:
                results.append([img_path, label] + scores)
                total_success += 1
            else:
                total_fail += 1

    header = ['filepath', 'label', 'happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"\n✅ Selesai! {total_success} gambar berhasil, {total_fail} gagal.")
    print(f"📄 Hasil disimpan ke: {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='baseline', help='Nama model: baseline, partial, dst')
    parser.add_argument('--dataset', type=str, required=True, default='RAF-DB', help='Nama dataset: RAF-DB atau FER2013')
    parser.add_argument("--output_dir", type=str, default="logs_eval")
    args = parser.parse_args()

    model_name = args.model
    dataset_name = args.dataset

    dataset_path = f"../dataset_cropped_yolo/{dataset_name}/test"
    checkpoint_path = get_checkpoint_path(model_name, dataset_name)
    #output_file = f"classification_{model_name}_{dataset_name}.csv"
    output_file = os.path.join(args.output_dir,f"classification_{model_name}_{dataset_name}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n🔍 Evaluasi Model: {model_name.upper()} | Dataset: {dataset_name.upper()}")
    print(f"📁 Dataset path: {dataset_path}")
    print(f"🧠 Checkpoint path: {checkpoint_path}")

    process_dataset(model_name, dataset_name, dataset_path, checkpoint_path, output_file)

if __name__ == "__main__":
    main()
