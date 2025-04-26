@echo off
echo ==========================================================
echo 🔥 Training DualPath+LSTM on CASME2 Sequences (YOLOv11 Crop)
python train_casme2_sequence.py ^
  --data_dir ../CASME2_Sequences_YOLO ^
  --output_dir ./logs_casme2/dualpath_lstm_yolo ^
  --num_classes 5 ^
  --batch_size 8 ^
  --epochs 100 ^
  --lr 1e-4 ^
  --early_stop 10 ^
  --seq_len 18 ^
  --feature_dim 256 ^
  --lstm_hidden 128 ^
  --lstm_layers 1
echo ==========================================================
echo ✅ Training Completed!
pause
