import argparse
from datetime import datetime
from enum import Enum
import os
import random

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.utils import set_random_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hems_env
from components.battery import HEMSBattery
from modeling.rbc import get_rule_based_stats
from utils.configs import (
    data_config,
    battery_config,
    env_config,
    policy_config,
    ppo_meta,
    sac_meta,
    td3_meta,
    results_paths,
    RANDOM_SEED,
)
from utils.processing.etl import get_dataset_as_dataframe, merge_transform_dataframes
from utils.statistics import save_stats_csv
from utils.graphics import polt_test_graphics, plot_evaluate_policy
from utils.event_logger import logger

from modeling.train import train_policy
from modeling.retrain import retrain_policy
from modeling.sample import sample_actions


set_random_seed(RANDOM_SEED, using_cuda=True)

# observation dataset preparation :
# ---------------------------------------------------
prosumers = get_dataset_as_dataframe(
    directory_path=data_config["prosumer_dir"],
    index_col=data_config["index_col"],
    usecols=data_config["prosumer_usecols"],
    index_freq=data_config["index_freq"],
)
assert len(prosumers) == 2

pricing = get_dataset_as_dataframe(
    directory_path=data_config["pricing_dir"],
    index_col=data_config["index_col"],
    usecols=data_config["pricing_usecols"],
    index_freq=data_config["index_freq"],
)
assert len(pricing) == 1

power_household, power_pv, auction_price, soc = merge_transform_dataframes(
    prosumers, pricing, data_config["index_col"], data_config["index_freq"]
)
assert (
    len(power_household) == len(power_pv) == len(soc) == len(auction_price) == 35041 * 2
)
initial_soc = soc.iloc[0]

# Train/Test Split
subset_index = int(len(power_household) * data_config["train_test_split_ratio"])
power_household_train = power_household[:subset_index]
power_pv_train = power_pv[:subset_index]
auction_price_train = auction_price[:subset_index]

test_index = 24*4*7
power_household_test = power_household[subset_index:subset_index+test_index]
power_pv_test = power_pv[subset_index:subset_index+test_index]
auction_price_test = auction_price[subset_index:subset_index+test_index]
logger.info(
    f"train dataset: {len(power_household_train)}, test dataset: {len(power_pv_test)}"
)

# environments (Train/Evaluation/Test)
# --------------------------------------------------
battery = HEMSBattery(
    initial_soc=initial_soc, max_capacity=battery_config["max_capacity"]
)
logger.info(f">> battery initialized with soc: {battery.current_soc}")
ENV_ID = env_config["env_id"]
ENV_KWARGS_TRAIN = {
    "power_household": power_household_train,
    "power_pv": power_pv_train,
    "auction_price": auction_price_train,
    "observation_window": env_config["observation_window_train"],
    "battery": battery,
}
ENV_KWARGS_TEST = {
    "power_household": power_household_test,
    "power_pv": power_pv_test,
    "auction_price": auction_price_test,
    "observation_window": len(power_household_test),  # test dataset length
    "battery": battery,
}

eval_env = Monitor(gym.make(ENV_ID, **ENV_KWARGS_TRAIN))
check_env(eval_env, warn=True)
logger.info(">> evaluation environment initialized")

if env_config["num_envs"] == 1:
    train_env = DummyVecEnv([lambda: gym.make(ENV_ID, **ENV_KWARGS_TRAIN)])
else:
    train_env = make_vec_env(
        env_id=ENV_ID,
        env_kwargs=ENV_KWARGS_TRAIN,
        n_envs=env_config["num_envs"],
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs=dict(start_method="fork"),
    )
logger.info(f">> {env_config['num_envs']} training environment/s initialized")

test_env = gym.make(ENV_ID, **ENV_KWARGS_TEST)
check_env(eval_env, warn=True)
logger.info(">> testing environment initialized")


def main():
    """Main()

    Args:
        None
    Returns:
        None
    """

    parser = argparse.ArgumentParser(description="Main module arguments")

    class ActionChoices(str, Enum):
        init = "init"
        train = "train"
        retrain = "retrain"
        test = "test"

    class ModelChoices(str, Enum):
        ppo = "ppo"
        sac = "sac"
        td3 = "td3"
        rbc = "rbc"

    parser.add_argument(
        "--model",
        type=ModelChoices,
        choices=list(ModelChoices),
        default="ppo",
        required=True,
        help="choices for model",
    )

    parser.add_argument(
        "--action",
        type=ActionChoices,
        choices=list(ActionChoices),
        default="train",
        required=True,
        help="choices for policy action: train, retrain, test ",
    )

    args = parser.parse_args()

    # -------------- Rule Based  --------------------------
    if args.model.value == "rbc":
        get_rule_based_stats(
            power_household_test,
            power_pv_test,
            auction_price_test,
            battery,
            data_config["save_graphics_index"],
        )
        return None

    action = args.action.value
    MODEL_USED = args.model.value
    logger.info(
        f">> action: {args.action.value}ing, using model: {args.model.value} <<"
    )

    # -------------------------------------------------
    TENSORBOARD_LOGS = os.path.join(
        results_paths["logs_dir"], f"tensorboard/{MODEL_USED}_core_metrics"
    )
    TENSORBOARD_LOGS_CUSTOM = os.path.join(
        results_paths["logs_dir"],
        f"tensorboard/{MODEL_USED}_{action}_{datetime.now().strftime('%m-%d_%H-%M')}",
    )
    POLICY_CHECKPOINT = os.path.join(results_paths["checkpoints_dir"], f"{MODEL_USED}")
    RESULTS_CSV_DIR = os.path.join(results_paths["stats_dir"], f"{MODEL_USED}/{action}")
    os.makedirs(RESULTS_CSV_DIR, exist_ok=True)
    GRAPHICS_DIR = os.path.join(results_paths["graphics_dir"], f"{MODEL_USED}/{action}")
    os.makedirs(GRAPHICS_DIR, exist_ok=True)

    # agents (PPO/SAC/TD3) | (Train/Evaluation/Test)
    # --------------------------------------------------
    # Noise for SAC and TD3
    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
    )
    if MODEL_USED == "ppo":
        policy = PPO(
            policy_config["policy_nw"],
            env=train_env,
            tensorboard_log=TENSORBOARD_LOGS,
            **ppo_meta,
        )
    elif MODEL_USED == "sac":
        policy = SAC(
            policy_config["policy_nw"],
            env=train_env,
            action_noise=action_noise,
            tensorboard_log=TENSORBOARD_LOGS,
            **sac_meta,
        )
    elif MODEL_USED == "td3":
        policy = TD3(
            policy_config["policy_nw"],
            env=train_env,
            action_noise=action_noise,
            tensorboard_log=TENSORBOARD_LOGS,
            **td3_meta,
        )
    else:
        logger.error("the model doesnot exist")
        raise ValueError("the model doesnot exist")
    logger.info(f">> policy with {MODEL_USED} initialized")
    
    # -------------- Untrained Evaluation --------------------------
    if action == "init":
        policy_dir = os.path.join(results_paths["checkpoints_dir"], f"{MODEL_USED}_{action}")
        policy.save(policy_dir)
        logger.info(f">> init policy checkpoint for {MODEL_USED} saved")
    
        replay_buffer = sample_actions(
            policy,
            test_env,
            MODEL_USED,
            TENSORBOARD_LOGS_CUSTOM,
            policy_dir,
            random_seed=RANDOM_SEED,
        )
        save_stats_csv(
            experiences=replay_buffer,
            timestep_indices=power_household_test.index,
            dir_to_save_csv=RESULTS_CSV_DIR,
        )
        polt_test_graphics(
            power_household_test,
            power_pv_test,
            dir_to_read_csv=RESULTS_CSV_DIR,
            dir_to_save_graphics=GRAPHICS_DIR,
            model_name=MODEL_USED,
            subset_index_to_plot=data_config["save_graphics_index"],
        )
    # -------------- Training/Evaluation --------------------------
    elif action == "train":
        rewards_acc, costs_acc = train_policy(
            policy,
            train_env,
            eval_env,
            MODEL_USED,
            TENSORBOARD_LOGS_CUSTOM,
            POLICY_CHECKPOINT,
        )
        plot_evaluate_policy(
            rewards_acc,
            costs_acc,
            dir_to_save_graphics=GRAPHICS_DIR,
            figure_name=f"{MODEL_USED}_train_evaluate_policy",
        )
    # ---------------- Retraining/Evaluation ----------------------
    elif action == "retrain":
        rewards_acc, costs_acc = retrain_policy(
            policy,
            train_env,
            eval_env,
            MODEL_USED,
            TENSORBOARD_LOGS_CUSTOM,
            POLICY_CHECKPOINT,
        )
        plot_evaluate_policy(
            rewards_acc,
            costs_acc,
            dir_to_save_graphics=GRAPHICS_DIR,
            figure_name=f"{MODEL_USED}_retrain_evaluate_policy",
        )
    # --------------- Sampling Actions ------------------------
    elif action == "test":
        replay_buffer = sample_actions(
            policy,
            test_env,
            MODEL_USED,
            TENSORBOARD_LOGS_CUSTOM,
            POLICY_CHECKPOINT,
            random_seed=RANDOM_SEED,
        )
        save_stats_csv(
            experiences=replay_buffer,
            timestep_indices=power_household_test.index,
            dir_to_save_csv=RESULTS_CSV_DIR,
        )
        polt_test_graphics(
            power_household_test,
            power_pv_test,
            dir_to_read_csv=RESULTS_CSV_DIR,
            dir_to_save_graphics=GRAPHICS_DIR,
            model_name=MODEL_USED,
            subset_index_to_plot=data_config["save_graphics_index"],
        )
    else:
        logger.error("invalid action")
        raise ValueError("invalid action")


if __name__ == "__main__":

    main()
