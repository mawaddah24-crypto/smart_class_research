@echo off
echo ==========================================================
echo 🔥 Training DualPath Adaptive on RAF-DB
python train_dualpath_adaptive.py

echo ==========================================================
echo 🔥 Training DualPath Context Residual DRM on RAF-DB
python train.py --model crdrm

echo ==========================================================
echo 🔥 Training DualPath Fusion on RAF-DB
python train.py --model fusion

echo ==========================================================
echo 🔥 Training DualPath Sem Fusion on RAF-DB
python train_dualpath_fusion.py

echo ==========================================================
echo 🔥 Training DualPath Baseline on RAF-DB
python train.py

echo ✅ All Trainings Completed Successfully!
pause
