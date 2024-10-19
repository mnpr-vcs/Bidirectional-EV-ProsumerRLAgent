#!/usr/bin/env bash
# ---------------------------------------------------
cd ../hems-env/ && pip uninstall hems-env -y && pip install e . && cd ../src/

model_name="sac"

python main.py --model=$model_name --action=init
printf ">>------------------------------- init inference finished.-------------------\n\n"
python main.py --model=$model_name --action=train
printf ">>------------------------------- training finished.-------------------\n\n"
python main.py --model=$model_name --action=test
printf ">>------------------------------- testing finished.-------------------\n\n"
cd ../src/results/graphics/$model_name/ && feh train/
# ---------------------------------------------------
