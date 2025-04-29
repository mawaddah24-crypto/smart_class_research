#!/bin/bash

echo "Training DualPath_Baseline on RAF-DB"
python train.py 

echo "Training DualPath_Baseline on FER2013"
python train.py --model fusion

echo "Training DualPath_PartialAttention (standard) on RAF-DB"
python train.py --model crdrm


echo "All trainings done!"
