"""
rule based control
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils.configs import battery_config, results_paths, env_config
from utils.event_logger import logger
from utils import graphics

GRID_DRAW_RATE_EUR = 0.33
GRID_FEED_RATE_EUR = 0.08

RBC_GRAPHICS_DIR = os.path.join(results_paths["graphics_dir"], "rbc")
RBC_STATS_DIR = os.path.join(results_paths["stats_dir"], "rbc")
os.makedirs(RBC_GRAPHICS_DIR, exist_ok=True)
os.makedirs(RBC_STATS_DIR, exist_ok=True)


def get_rule_based_stats(
    demand,
    generation,
    auction_price, 
    battery, 
    plot_index
):
    """
    get rule based stats

    Args:
        demand (pd.Series): household demand
        generation (pd.Series): pv generation
        auction_price (pd.Series): auction price
        battery (HEMSBattery): battery
        plot_index (int): index range to plot
    """
    exchanges = []
    socs = []
    costs = []
    actions = []

    for i in range(len(demand)):
        exchange = generation[i] + demand[i]
        if exchange < 0:  # excess pv
            if battery.current_soc == 1:  # if battery full : sell
                action = 0
                net_exchange = exchange + action
                cost = net_exchange * auction_price[i]
                # cost = net_exchange * GRID_FEED_RATE_EUR
                soc = battery.get_current_soc(action)
                actions.append(action)
                exchanges.append(net_exchange)
                costs.append(cost)
                socs.append(soc)
                logger.info(f"selling ..., exchange : {net_exchange}, soc : {soc}")
            else:  # if battery not full : charge
                action = battery_config["max_charge_rate"]
                net_exchange = exchange + action
                cost = net_exchange * auction_price[i]
                # cost = net_exchange * GRID_DRAW_RATE_EUR
                soc = battery.get_current_soc(action)
                actions.append(action)
                exchanges.append(net_exchange)
                costs.append(cost)
                socs.append(soc)
                logger.info(f"charging ..., exchange : {net_exchange}, soc : {soc}")
        else:  # excess demand
            if battery.current_soc == 0:  # if battery empty : buy
                action = 0
                net_exchange = exchange + action
                cost = net_exchange * auction_price[i]
                # cost = net_exchange * GRID_DRAW_RATE_EUR
                soc = battery.get_current_soc(action)
                actions.append(action)
                exchanges.append(net_exchange)
                costs.append(cost)
                socs.append(soc)
                logger.info(f"buying ..., exchange : {net_exchange}, soc : {soc}")
            else:  # if battery not empty : discharge
                action = battery_config["max_discharge_rate"]
                net_exchange = exchange + action
                cost = net_exchange * auction_price[i]
                # cost = net_exchange * GRID_FEED_RATE_EUR
                soc = battery.get_current_soc(action)
                actions.append(action)
                exchanges.append(net_exchange)
                costs.append(cost)
                socs.append(soc)
                logger.info(f"discharging ..., exchange : {net_exchange}, soc : {soc}")
    logger.info(f"total cost: {np.sum(costs)}")
    # if item is a list convert to float
    socs = [float(i) for i in socs]
    df = pd.DataFrame(
        {
            "pw_household": demand,
            "pw_pv": generation,
            "soc": socs,
            "action": actions,
            "exchange": exchanges,
            # "feed_rate": GRID_FEED_RATE_EUR,
            # "draw_rate": GRID_DRAW_RATE_EUR,
            "auction_price": auction_price,
            "cost": costs,
        }
    )
    df.to_csv(os.path.join(RBC_STATS_DIR, "step_stats.csv"))
    logger.info(">> rule based stats saved as csv")

    # rbc graphics
    df = df[:plot_index]
    figure_name = f"rbc_comparison"
    fig, axs = plt.subplots(7, 1, figsize=(40, 25), sharex=True)

    axs[0].plot(df["pw_household"], label="household consumption", color="blue")
    axs[0].fill_between(
        df.index, df["pw_household"], 0, color="blue", alpha=0.2
    )
    axs[0].fill_between(
        df.index, -df["pw_pv"], 0, color="green", alpha=0.1
    )
    axs[0].set_ylabel("(kWh)")
    axs[0].legend()

    axs[1].plot(df["pw_pv"], label="pv production", color="green")
    axs[1].fill_between(
        df.index, df["pw_pv"], 0, color="green", alpha=0.2
    )
    axs[1].set_ylabel("(kWh)")
    axs[1].legend()

    # 1. ------------------------------------------------------------------
    axs[2].plot(df["auction_price"], label="exchange rate", color="orange")
    axs[2].fill_between(
        df.index, df["auction_price"], 0, color="orange", alpha=0.1
    )
    axs[2].set_ylabel("(Eur/kWh)")

    # 2. ------------------------------------------------------------------
    # axs[2].plot(df["feed_rate"], label="feed rate", color="green")
    # axs[2].fill_between(
    #     df.index, df["feed_rate"], 0, color="green", alpha=0.1
    # )
    # axs[2].plot(df["draw_rate"], label="draw rate", color="orange")
    # axs[2].fill_between(
    #     df.index, df["draw_rate"], 0, color="orange", alpha=0.1
    # )
    # axs[2].set_ylabel("(Eur/kWh)")
    # axs[2].legend(["feed rate", "draw rate"])


    axs[3].plot(df["soc"], label="soc", color="indigo")
    axs[3].fill_between(
        df.index, df["soc"], 0, color="indigo", alpha=0.2
    )
    axs[3].set_ylabel("[0, 1]")
    axs[3].legend()

    axs[4].plot(df["action"], label="action", color="red")
    axs[4].fill_between(
        df.index, df["action"], 0, color="red", alpha=0.2
    )
    axs[4].set_ylabel("(kWh)")
    axs[4].legend()

    axs[5].plot(df["exchange"], label="grid exchange", color="black")
    axs[5].plot(df["exchange"].rolling(4).mean(), label="(hourly average)", color="purple")
    axs[5].fill_between(
        df.index, df["exchange"], 0, color="black", alpha=0.2
    )
    axs[5].set_ylabel("(Eur/kWh)")
    axs[5].legend()

    axs[6].plot(df["cost"], label="cost", color="violet")
    axs[6].fill_between(
        df.index, df["cost"], 0, color="violet", alpha=0.2
    )
    axs[6].set_ylabel("(Eur)")
    axs[6].legend()

    fig.suptitle(f"RBC comparison")
    fig.supxlabel("Timesteps (UTC)", x=0.5, y=0.015)
    plt.savefig(os.path.join(RBC_GRAPHICS_DIR, figure_name))
    logger.info(">> rule based graphics saved")
    plt.close()

    return None
