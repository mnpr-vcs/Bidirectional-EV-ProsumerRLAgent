#!/usr/bin/env bash
# ---------------------------------------------------
#  Script to reproduce results
# ---------------------------------------------------
# steps:
# 1. activate the venv
# 2. run python main with args
#   - run ppo, sac, td3 train in order
#   - run rbc
#   - run ppo, sac, td3 test in any order
# 3. run total cost script
# ---------------------------------------------------

# change to project src
cd ../
# 1. activate the venv
# source .venv/bin/activate
conda activate hems-rlpy
# 2. run python main with args
# install local hems-env
cd ./hems-env/ && pip uninstall hems-env -y && pip install e . && cd ../
# change to project src
cd ./src/

# run inference on initial random policy
python main.py --model=ppo --action=init
printf ">>-------------------------------PPO initial inference finished.-------------------\n\n"
python main.py --model=sac --action=init
printf ">>-------------------------------SAC initial inference finished.-------------------\n\n"
python main.py --model=td3 --action=init
printf ">>-------------------------------TD3 initial inference finished.-------------------\n\n"

# run training
python main.py --model=ppo --action=train
printf ">>-------------------------------PPO training finished.-------------------\n\n"
python main.py --model=sac --action=train
printf ">>-------------------------------SAC training finished.-------------------\n\n"
python main.py --model=td3 --action=train
printf ">>-------------------------------TD3 training finished.-------------------\n\n"
# run rbc
python main.py --model=rbc --action=test
printf ">>-------------------------------RBC run finished.-------------------\n\n"
# run inference on trained policy
python main.py --model=ppo --action=test
printf ">>-------------------------------PPO testing finished.-------------------\n\n"
python main.py --model=sac --action=test
printf ">>-------------------------------SAC testing finished.-------------------\n\n"
python main.py --model=td3 --action=test
printf ">>-------------------------------TD3 testing finished.-------------------\n\n"
# save total cost
python save_total_cost.py
printf ">>---------------------- End of Script -------------------\n\n"
cat results/stats/total_cost.csv
printf "\n >> Run experiments finished.\n\trun \"tensorboard --logdir=results/logs/tensorboard/\" for results\n"
