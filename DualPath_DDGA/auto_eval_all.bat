@echo off
echo ==========================================================
echo 🔍 Evaluasi Model DualPath_Baseline pada RAF-DB
python eval_dualpath.py --model baseline --dataset RAF-DB

echo ==========================================================
echo 🔍 Evaluasi Model DualPath_Baseline pada FER2013
python eval_dualpath.py --model baseline --dataset FER2013

echo ==========================================================
echo 🔍 Evaluasi DualPath_PartialAttention pada RAF-DB
python eval_dualpath.py --model partial --dataset RAF-DB

echo ==========================================================
echo 🔍 Evaluasi DualPath_PartialAttention pada FER2013
python eval_dualpath.py --model partial --dataset FER2013

echo ==========================================================
echo 🔍 Evaluasi DualPath_PartialAttention_Modified pada RAF-DB
python eval_dualpath.py --model partialmodif --dataset RAF-DB

echo ==========================================================
echo 🔍 Evaluasi DualPath_PartialAttention_Modified pada FER2013
python eval_dualpath.py --model partialmodif --dataset FER2013

echo ==========================================================
echo 🔍 Evaluasi DualPath_PartialAttention_SAP pada RAF-DB
python eval_dualpath.py --model semantic --dataset RAF-DB

echo ==========================================================
echo 🔍 Evaluasi DualPath_PartialAttention_SAP pada FER2013
python eval_dualpath.py --model semantic --dataset FER2013

echo ==========================================================
echo ✅ Semua Evaluasi Selesai!
pause
