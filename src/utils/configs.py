import os
from enum import Enum
from datetime import datetime

RANDOM_SEED = 111

# input
# ------------------------------------------------
data_config = {
    "prosumer_dir": os.path.join("./dataset/", "prosumer/"),
    "pricing_dir": os.path.join("./dataset/", "pricing/"),
    "index_col": "Timestamp UTC",
    "prosumer_usecols": ["Timestamp UTC", "Power PV", "Power Household", "SoC"],
    "pricing_usecols": ["Timestamp UTC", "auction_price"],
    "index_freq": "15min",
    "train_test_split_ratio": 0.9,  # train 93 weeks | test 10 weeks
    "dataset_size": 70082,
    "save_graphics_index": 24 * 4 * 7,  # 5 days
}

# environment
# ------------------------------------------------
battery_config: dict = {
    "max_capacity": 50,  # kwh
    "max_charge_rate": 11,  # kwh
    "max_discharge_rate": -11,  # kwh
}

env_config: dict = {
    "env_id": "hems_env/HouseholdEnv-v0",
    "observation_window_train": int(24 * 4 * 7 * 2),  # 2 weeks
    "num_envs": 1,
}

# agent
# ------------------------------------------------
policy_config: dict = {
    "policy_nw": "MlpPolicy",
    "reset_num_timesteps": False,
    "num_train_eval_cycles": 200,
    "num_retrain_eval_cycles": 200,
    "num_eval_episodes": 2,
    "num_test_episodes": 5,
    "train_timesteps": env_config["observation_window_train"] * 2, # 2 * 2 weeks
    "retrain_timesteps": env_config["observation_window_train"] * 2,
}
ppo_meta: dict = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "clip_range": 0.20,
    "verbose": 1,
    "policy_kwargs": {"net_arch": {"pi": [64, 64, 16], "vf": [64, 64, 16]}},
}
sac_meta: dict = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "tau": 0.005,
    "verbose": 1,
    "policy_kwargs": {"net_arch": {"pi": [256, 256, 16], "qf": [256, 256, 16]}},
}
td3_meta: dict = {
    "learning_rate": 0.001,
    "gamma": 0.99,
    "tau": 0.005,
    "verbose": 1,
    "policy_kwargs": {"net_arch": {"pi": [128, 128, 16], "qf": [128, 128, 16]}},
}

# results
# ------------------------------------------------
RESULTS_PATH = "./results/"
os.makedirs(RESULTS_PATH, exist_ok=True)
results_paths: dict = {
    "logs_dir": os.path.join(RESULTS_PATH, "logs/"),
    "checkpoints_dir": os.path.join(RESULTS_PATH, "checkpoints/"),
    "stats_dir": os.path.join(RESULTS_PATH, "stats/"),
    "graphics_dir": os.path.join(RESULTS_PATH, "graphics/"),
}
{os.makedirs(path, exist_ok=True) for path in results_paths.values()}
