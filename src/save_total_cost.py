"""
Save total costs for each policy as csv
"""

import pandas as pd
import os
import numpy as np

from utils.event_logger import logger
from utils.configs import policy_config, results_paths

PPO_STATS_DIR = os.path.join(results_paths["stats_dir"], "ppo/test")
SAC_STATS_DIR = os.path.join(results_paths["stats_dir"], "sac/test")
TD3_STATS_DIR = os.path.join(results_paths["stats_dir"], "td3/test")
RBC_STATS_DIR = os.path.join(results_paths["stats_dir"], "rbc")


def read_csv_from_dir(dir_to_read_csv):
    """
    A function that reads a CSV file from a directory, extracts specific columns, and returns a DataFrame.
    Parameters:
        dir_to_read_csv (str): The directory path where the CSV file is located.
    Returns:
        pandas.DataFrame: A DataFrame containing the specified columns from the CSV file.
    """
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(
            dir_to_read_csv,
            f"step_stats_episode{policy_config['num_test_episodes']-1}.csv",
        ),
        usecols=["Timestamp UTC", "cost"],
    )
    return df


ppo_df = read_csv_from_dir(os.path.join(PPO_STATS_DIR))
sac_df = read_csv_from_dir(os.path.join(SAC_STATS_DIR))
td3_df = read_csv_from_dir(os.path.join(TD3_STATS_DIR))
rbc_df = pd.read_csv(
    filepath_or_buffer=os.path.join(RBC_STATS_DIR, "step_stats.csv"),
    usecols=["Timestamp UTC", "cost"],
)

total_cost_ppo = ppo_df["cost"].sum()
total_cost_sac = sac_df["cost"].sum()
total_cost_td3 = td3_df["cost"].sum()
total_cost_rbc = rbc_df["cost"].sum()

total_cost_df = pd.DataFrame(
    data={
        "ppo": [total_cost_ppo],
        "sac": [total_cost_sac],
        "td3": [total_cost_td3],
        "rbc": [total_cost_rbc],
    },
    dtype=np.float32,
)

if __name__ == "__main__":
    total_cost_df.to_csv(os.path.join(results_paths["stats_dir"], "total_cost.csv"))
    logger.info("total cost csv saved")
