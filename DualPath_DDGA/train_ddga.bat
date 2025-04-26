@echo off
echo ==========================================================
echo 🔥 Training DualPath_DDGA on RAF-DB
python train_ddga.py --dataset RAF-DB --output_dir logs/ddga_rafdb

echo ==========================================================
echo 🔥 Training DualPath_DDGA on FER2013
python train_ddga.py --dataset FER2013 --output_dir logs/ddga_fer2013

echo ==========================================================
echo 🔥 Training DualPath_DDGA Fusion on RAF-DB
python train_ddga.py --model=fusion --dataset RAF-DB --output_dir logs/ddga_fusion_rafdb

echo ==========================================================
echo 🔥 Training DualPath_DDGA Fusion on FER2013
python train_ddga.py --model=fusion --dataset FER2013 --output_dir logs/ddga_fusion_fer2013

echo ==========================================================
echo 🔥 Training DualPath_DDGA DRM on RAF-DB
python train_ddga.py --model=drm --dataset RAF-DB --output_dir logs/ddga_drm_rafdb

echo ==========================================================
echo 🔥 Training DualPath_DDGA DRM on FER2013
python train_ddga.py --model=drm --dataset FER2013 --output_dir logs/ddga_drm_fer2013

echo ✅ All Trainings Completed Successfully!
pause
