import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.event_logger import logger
from utils.configs import policy_config


GRID_DRAW_RATE_EUR = 0.08
GRID_FEED_RATE_EUR = 0.33

def save_stats_csv(
    experiences: list,
    timestep_indices: list,
    dir_to_save_csv: str,
):
    """
    Save the test statistics as a CSV file.
    Args:
        experiences (list): A list of tuples containing the following information per step:
            - pw_household (float): Power consumption of the household in kW.
            - pw_pv (float): Power consumption from PV in kW.
            - auction_price (float): Auctioning price in EUR/kWh.
            - soc (float): State of Charge.
            - reward (float): The reward received.
            - action (float): The action taken.
            - exchange (float): The exchange value.
            - cost (float): The cost incurred.
        timestep_indices (list): A list of timestep indices.
        dir_to_save_csv (str): The directory to save the CSV file.
    Returns:
        None
    """
    df = pd.DataFrame(
        experiences,
        columns=[
            "pw_household",
            "pw_pv",
            # "feed_rate",
            # "draw_rate",
            "auction_price",
            "soc",
            "action",
            "reward",
            "exchange",
            "cost",
        ],
    )
    df["action"] = df["action"].map(lambda x: x.item())
    df["exchange"] = df["exchange"].map(lambda x: x.item())

    df.to_csv(os.path.join(dir_to_save_csv, "step_stats.csv"))
    logger.info(">> total step test stats saved as csv")

    # split into episodes, set timestamp_indices, and save as csv
    episodic_stats = np.array_split(df, policy_config["num_test_episodes"], axis=0)
    # timestep_indices = timestep_indices[: int(len(episodic_stats[0]))]
    [
        episodic_stats[i]
        .set_index(timestep_indices)
        .to_csv(os.path.join(dir_to_save_csv, f"step_stats_episode{i}.csv"))
        for i in range(len(episodic_stats))
    ]
    episodic_cost = [episodic_stats[i]["cost"].sum() for i in range(len(episodic_stats))]
    logger.info(f">> episodic average cost of {len(episodic_cost)} episodes: {np.sum(episodic_cost)/len(episodic_cost)}")
    logger.info(">> episodic test stats saved as csv")
