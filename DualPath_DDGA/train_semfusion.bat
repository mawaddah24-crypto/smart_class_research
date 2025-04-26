@echo off
echo ==========================================================
echo 🔥 Training DualPath_Baseline on RAF-DB
python train_dualpath_fusion.py --dataset RAF-DB --output_dir logs/semfusion_rafdb

echo ==========================================================
echo 🔥 Training DualPath_Baseline on FER2013
python train_dualpath_fusion.py --dataset FER2013 --output_dir logs/semfusion_fer2013

echo ==========================================================
echo 🔥 Training DualPath_PartialAttention (standard) on RAF-DB
python train_dualpath_fusion.py --dataset FER2013Plus --output_dir logs/semfusion_fer2013plus

echo ==========================================================
echo 🔥 Training DualPath_PartialAttention (standard) on FER2013
pythontrain_dualpath_fusion.py --dataset Affectnet --output_dir logs/semfusion_Affectnet

echo ✅ All Trainings Completed Successfully!
pause
