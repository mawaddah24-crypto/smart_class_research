@echo off
echo ==========================================================
echo 🔥 Training DualPath_PartialAttention_Modified on RAF-DB
python train.py --model partialmodif --dataset RAF-DB --output_dir logs/partialmodif_rafdb

echo ==========================================================
echo 🔥 Training DualPath_PartialAttention_SAP on RAF-DB
python train.py --model semantic --dataset RAF-DB --output_dir logs/semantic_rafdb

echo ==========================================================
echo 🔥 Training DualPath_Simplified on RAF-DB
python train.py --model base --dataset RAF-DB --output_dir logs/simplified_rafdb


echo ==========================================================
echo 🔥 Training DualPath_Baseline on RAF-DB
python train.py --model dual --dataset RAF-DB --output_dir logs/dual_rafdb

echo ==========================================================
echo 🔥 Training DualPath_Baseline on RAF-DB
python train.py --model baseline --dataset RAF-DB --output_dir logs/baseline_rafdb

echo ==========================================================
echo 🔥 Training DualPath_PartialAttention (standard) on RAF-DB
python train.py --model partial --dataset RAF-DB --output_dir logs/partial_rafdb

echo ==========================================================
echo 🔥 Training DualPath_Baseline_DSE on RAF-DB
python train.py --model dse --dataset RAF-DB --output_dir logs/semantic_rafdb

echo ✅ All Trainings Completed Successfully!
pause
