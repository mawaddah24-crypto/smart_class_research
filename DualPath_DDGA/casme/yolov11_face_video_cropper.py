import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# ⚡ Load YOLOv11 Face Model
from ultralytics import YOLO

def crop_faces_from_video(video_dir, output_dir, model_path="yolov11_face.pt", img_size=640, save_size=(224,224)):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(".avi")]

    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        save_folder = os.path.join(output_dir, video_name)
        os.makedirs(save_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLOv11 inference
            results = model.predict(frame, imgsz=img_size, device=device, verbose=False)[0]

            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            if len(boxes) > 0:
                # Ambil wajah confidence tertinggi
                idx = confs.argmax()
                x1, y1, x2, y2 = boxes[idx]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue  # Skip kalau crop kosong

                face_crop = cv2.resize(face_crop, save_size)

                save_path = os.path.join(save_folder, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(save_path, face_crop)

            frame_idx += 1

        cap.release()
        print(f"✅ Processed {frame_idx} frames from {video_name}")

    print("\n🎉 Semua video selesai di-crop wajahnya!")

if __name__ == "__main__":
    video_dir = "../CASME2_RAW_selected"
    output_dir = "../CASME2_Cropped_YOLO"
    model_path = "./yolov11_face.pt"  # Sesuaikan ke path model YOLOv11 Face
    crop_faces_from_video(video_dir, output_dir, model_path)
