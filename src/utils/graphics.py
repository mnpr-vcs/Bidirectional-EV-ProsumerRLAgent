import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import os
import pandas as pd
import seaborn as sns

from utils.configs import policy_config
from utils.event_logger import logger


rcParams["figure.figsize"] = 45, 15
rcParams["figure.titlesize"] = "large"
rcParams["figure.labelsize"] = "large"
rcParams["figure.autolayout"] = True
rcParams["font.style"] = "italic"
rcParams["font.variant"] = "normal"
rcParams["font.stretch"] = "condensed"
rcParams["font.family"] = "monospace"
rcParams["font.serif"] = ["Bitstream Vera Sans Mono"]
rcParams["font.size"] = 20
rcParams["grid.linestyle"] = "--"
rcParams["lines.linewidth"] = 3 
rcParams["lines.linestyle"] = "-" 
rcParams["xtick.major.size"] = 3
rcParams["xtick.minor.size"] = 1
rcParams["xtick.alignment"] = "center"
rcParams["ytick.major.size"] = 3
rcParams["ytick.minor.size"] = 1
rcParams["ytick.alignment"] = "center"
rcParams["legend.frameon"] = False
rcParams["legend.facecolor"] = "inherit"
rcParams["legend.fancybox"] = False
sns.set_style("whitegrid")
sns.set_palette("muted")

GRID_DRAW_RATE_EUR = 0.08
GRID_FEED_RATE_EUR = 0.33

def polt_test_graphics(
    power_household_test,
    power_pv_test,
    dir_to_read_csv,
    dir_to_save_graphics,
    model_name,
    subset_index_to_plot,
):
    """
    Generate a set of test graphics based on the given data.
    Args:
        power_household_test (array-like): The power household test data.
        power_pv_test (array-like): The power PV test data.
        dir_to_read_csv (str): The directory to read the CSV file from.
        dir_to_save_graphics (str): The directory to save the graphics.
        model_name (str): The name of the model.
        subset_index_to_plot (int): The index to subset the data for plotting.
    Returns:
        None
    """
    df = pd.read_csv(
        filepath_or_buffer=os.path.join(
            dir_to_read_csv,
            f"step_stats_episode{policy_config['num_test_episodes']-1}.csv",
        ),
    )
    df["Timestamp UTC"] = pd.to_datetime(df["Timestamp UTC"])
    df.set_index(df["Timestamp UTC"], inplace=True)
    df = df.iloc[:subset_index_to_plot]

    # power_household, power_pv, action, exchange
    figure_name = f"{model_name}_power_balance"
    fig, axs = plt.subplots(3, 1, figsize=(45, 20), sharex=True)
    axs[0].plot(df["pw_household"], label="pw_household", color="blue")
    axs[0].plot(df["pw_pv"], label="pw_pv", color="green")
    axs[0].fill_between(
        df.index, df["pw_household"], df["pw_pv"], color="green", alpha=0.2
    )
    axs[0].fill_between(
        df.index, df["pw_household"], 0, color="blue", alpha=0.2
    )
    axs[0].fill_between(
        df.index, -df["pw_pv"], 0, color="green", alpha=0.1
    )
    axs[0].set_ylabel("(kWh)")
    axs[0].legend()
    axs[1].plot(df["action"], label="action", color="red")
    axs[1].fill_between(
    df.index, df["action"], 0, color="red", alpha=0.2
    )
    axs[1].set_ylabel("(kWh)")
    axs[1].legend()
    axs[2].plot(df["exchange"], label="grid exchange", color="black")
    axs[2].plot(df["exchange"].rolling(4).mean(), label="(hourly average)", color="purple")
    axs[2].fill_between(
        df.index, df["exchange"], 0, color="black", alpha=0.2
    )
    axs[2].set_ylabel("(kWh)")
    axs[2].legend()

    fig.suptitle("Power Balance Components")
    fig.supxlabel("Timesteps (UTC)", x=0.5, y=0.015)
    fig.supylabel("Power (kWh)", x=0.005, y=0.5)
    plt.savefig(os.path.join(dir_to_save_graphics, figure_name))
    plt.close(fig)

    # power_household, power_pv, auction_price, soc, action
    figure_name = f"{model_name}_observation_relative_action"
    fig, axs = plt.subplots(4, 1, figsize=(45, 20), sharex=True)

    axs[0].plot(df["pw_household"], label="household consumption", color="blue")
    axs[0].plot(df["pw_pv"], label="pv production", color="green")
    axs[0].fill_between(
        df.index, df["pw_household"], df["pw_pv"], color="green", alpha=0.2
    )
    axs[0].fill_between(
        df.index, df["pw_household"], 0, color="blue", alpha=0.2
    )
    axs[0].fill_between(
        df.index, -df["pw_pv"], 0, color="green", alpha=0.1
    )
    axs[0].set_ylabel("(kWh)")
    axs[0].legend()

    # 1. -------------------------------------------------------------------
    axs[1].plot(df["auction_price"], label="exchange rate", color="orange")
    axs[1].set_ylabel("(Eur/kWh)")
    axs[1].fill_between(
        df.index, df["auction_price"], 0, color="orange", alpha=0.1
    )
    axs[1].legend()

    # 2. -------------------------------------------------------------------
    # axs[1].plot(df["feed_rate"], label="feed rate", color="green")
    # axs[1].fill_between(
    #     df.index, df["feed_rate"], 0, color="green", alpha=0.1
    # )
    # axs[1].plot(df["draw_rate"], label="draw rate", color="orange")
    # axs[1].fill_between(
    #     df.index, df["draw_rate"], 0, color="orange", alpha=0.1
    # )
    # axs[1].set_ylabel("(Eur/kWh)")
    # axs[1].legend(["feed rate", "draw rate"])

    axs[2].plot(df["soc"], label="soc", color="indigo")
    axs[2].set_ylabel("[0, 1]")
    axs[2].fill_between(
        df.index, df["soc"], 0, color="indigo", alpha=0.2
    )
    axs[2].legend()
    # y_ticks = [0, 0.5, 1.0]
    # axs[2].set_yticks(y_ticks)

    axs[3].plot(df["action"], label="action", color="red")
    axs[3].set_ylabel("(kWh)")
    axs[3].plot(
        df["action"].rolling(4).mean(), label="(hourly average)", color="purple"
    )
    axs[3].fill_between(
        df.index, df["action"], 0, color="red", alpha=0.2
    )
    axs[3].legend()

    fig.suptitle("Observations relative to action")
    fig.supxlabel("Timesteps (UTC)", x=0.5, y=0.015)
    plt.savefig(os.path.join(dir_to_save_graphics, figure_name))
    plt.close(fig)

    # rbc comparison graphics
    df = df[
        [
            "pw_household",
            "pw_pv",
            "soc",
            # "feed_rate",
            # "draw_rate",
            "auction_price",
            "action",
            "reward",
            "exchange",
            "cost"
        ]
    ]
    figure_name = f"{model_name}_rbc_comparison"
    fig, axs = plt.subplots(8, 1, figsize=(40, 25), sharex=True)

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

    # 1. -------------------------------------------------------------------
    axs[2].plot(df["auction_price"], label="exchange rate", color="orange")
    axs[2].fill_between(
        df.index, df["auction_price"], 0, color="orange", alpha=0.1
    )
    axs[2].set_ylabel("(Eur/kWh)")
    
    #2. ------------------------------------------------------------------- 
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

    axs[7].plot(df["reward"], label="reward", color="teal")
    axs[7].fill_between(
        df.index, df["reward"], 0, color="teal", alpha=0.2
    )
    axs[7].set_ylabel("(-)")
    axs[7].legend()

    # df.plot(subplots=True, figsize=(45, 25), sharex=True)
    fig.suptitle(f"Overall rbc comparison {model_name}")
    fig.supxlabel("Timesteps (UTC)", x=0.5, y=0.015)
    plt.savefig(os.path.join(dir_to_save_graphics, figure_name))
    logger.info(">> rbc comparison graphics saved")
    plt.close()


def plot_evaluate_policy(
    episodic_rewards: list,
    episodic_costs: list,
    dir_to_save_graphics: str,
    figure_name: str,
):
    """
    Plot and save a graph to evaluate policy based on episodic rewards.
    Args:
        episodic_rewards (list): List of episodic rewards for evaluation.
        dir_to_save_graphics (str): Directory path to save the evaluation graphics.
        figure_name (str): Name of the figure to be saved.
    Returns:
        None
    """
    # fig, ax = plt.subplots(sharex=True)
    # ax.plot(episodic_rewards, label="reward", color="pink")
    fig, axs = plt.subplots(2, 1, figsize=(45, 20), sharex=True)
    
    axs[0].plot(episodic_rewards, label="reward", color="teal")
    axs[0].fill_between(
        range(len(episodic_rewards)), episodic_rewards, 0, color="teal", alpha=0.2
    )
    axs[0].set_ylabel("(-)")
    axs[0].legend()

    axs[1].plot(episodic_costs, label="cost", color="violet")
    axs[1].fill_between(
        range(len(episodic_costs)), episodic_costs, 0, color="violet", alpha=0.2
    )
    axs[1].set_ylabel("(-)")
    axs[1].legend()

    fig.suptitle("Cumulative rewards")
    fig.supxlabel("Num of episodes (int) :", x=0.5, y=0.015)
    plt.savefig(os.path.join(dir_to_save_graphics, figure_name))
    plt.close(fig)
    logger.info(">> evaluate policy graphics saved")


def plot_line_subplots(
    power_household_o,
    power_pv_o,
    power_household_result,
    power_pv_result,
    time_index,
    dir_to_save_graphics,
    subset_index,
):
    """
    Plot and save a line plot of the household power consumption and solar power generation,
    as well as the predicted power consumption and solar power generation for a subset of the data.

    Args:
        power_household_o (numpy.ndarray): The original household power consumption data.
        power_pv_o (numpy.ndarray): The original solar power generation data.
        power_household_result (numpy.ndarray): The predicted household power consumption data.
        power_pv_result (numpy.ndarray): The predicted solar power generation data.
        time_index (numpy.ndarray): The time index for the data.
        dir_to_save_graphics (str): The directory path to save the graphics.
        subset_index (int): The index to subset the data.

    Returns:
        None

    This function plots and saves a line plot of the original household power consumption and solar power generation,
    as well as the predicted power consumption and solar power generation for a subset of the data. The plot includes
    a title, grid lines, and a legend with labels for each line. The resulting figure is saved to the specified directory.
    """
    power_household_o = power_household_o[:subset_index]
    power_pv_o = power_pv_o[:subset_index]

    figure_name = "pw_household_pv_original_line"
    fig, ax = plt.subplots()

    ax.plot(time_index, power_household_o)
    ax.plot(time_index, power_pv_o)
    ax.plot(time_index, power_household_result, linestyle="dotted")
    ax.plot(time_index, power_pv_result, linestyle="dotted")

    ax.set_title(figure_name)
    ax.grid(True)
    ax.legend(["pw_h_o", "pw_p_o", "pw_h_r", "pw_p_r"])
    plt.savefig(os.path.join(dir_to_save_graphics, figure_name))
    plt.close(fig)

    logger.info(">> comparison graphics saved")
