#!/bin/bash

echo "Training DualPath_Baseline on RAF-DB"
python train.py --model baseline --dataset RAF-DB

echo "Training DualPath_Baseline on FER2013"
python train.py --model baseline --dataset FER2013

echo "Training DualPath_PartialAttention (standard) on RAF-DB"
python train.py --model partial --dataset RAF-DB

echo "Training DualPath_PartialAttention (standard) on FER2013"
python train.py --model partial --dataset FER2013

echo "Training DualPath_PartialAttention (modified) on RAF-DB"
python train.py --model partialmodif --dataset RAF-DB

echo "Training DualPath_PartialAttention (modified) on FER2013"
python train.py --model partialmodif --dataset FER2013

echo "All trainings done!"
