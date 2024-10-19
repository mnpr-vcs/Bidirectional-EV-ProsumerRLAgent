import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from utils.event_logger import logger


# read all csv files in a directory as a single dataframe
def get_dataset_as_dataframe(
    directory_path: str, index_col: str, usecols: list, index_freq: str
) -> pd.DataFrame:
    """
    Reads all CSV files in a given directory and returns them as a list of pandas DataFrames.
    Parameters:
        directory_path (str): The path to the directory containing the CSV files.
        index_col (str): The name of the column to be used as the index of the DataFrames.
        usecols (list): A list of column names to include in the DataFrames.
        index_freq (str): The frequency of the index values.
    Returns:
        dataframes (list of pandas.DataFrame): A list of pandas DataFrames, each representing a CSV file in the directory.
    """
    dataframes = []

    try:
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                try:
                    data_series = pd.read_csv(
                        filepath_or_buffer=os.path.join(directory_path, filename),
                        parse_dates=[index_col],
                        index_col=index_col,
                        usecols=usecols,
                    )
                    data_series.index.freq = index_freq
                    dataframes.append(data_series)
                except Exception as e:
                    print(f"Error reading file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory {directory_path} not found.")

    return dataframes


# transform and merge them into observation inputs
def merge_transform_dataframes(
    prosumers: pd.DataFrame, pricing: pd.DataFrame, index_col: str, index_freq: str
) -> pd.DataFrame:
    """
    Merge and transform two dataframes containing prosumer and pricing data.
    Parameters:
        prosumers (pd.DataFrame): A list of two pandas DataFrames representing prosumer data.
        pricing (pd.DataFrame): A list of two pandas DataFrames representing pricing data.
        index_col (str): The name of the column to be used as the index of the merged DataFrame.
        index_freq (str): The frequency of the index values.
    Returns:
        df (pd.DataFrame): A merged and transformed DataFrame containing the following columns:
            - "Power Household" (np.float32): Power consumption of the household in kWh.
            - "Power PV" (np.float32): Power consumption from PV in kWh.
            - "auction_price" (np.float32): Selling price in EUR/kWh.
            - "SoC" (np.float32): State of Charge.
    """
    df = pd.concat(
        [
            pd.merge(prosumers[0], pricing[0], on=index_col, how="left"),
            pd.merge(prosumers[1], pricing[0], on=index_col, how="left"),
        ]
    ).astype(
        {
            "Power Household": np.float32,
            "Power PV": np.float32,
            "auction_price": np.float32,
            "SoC": np.float32,
        }
    )
    # fill na with 0
    df = df.fillna(0)
    # invert power pv sign
    df["Power PV"] = df["Power PV"]
    # clip power pv, auction price
    df["Power PV"] = df["Power PV"].clip(lower=df["Power PV"].min(), upper=0, axis=0)
    df["auction_price"] = df["auction_price"].clip(lower=0, upper=df["auction_price"].max())
    # unit conversions
    # power in kwh
    df["Power Household"] /= 1000
    df["Power PV"] /= 1000
    # price in EUR/kWh
    df["auction_price"] /= 1000

    return (
        df["Power Household"],
        df["Power PV"],
        df["auction_price"],
        df["SoC"],
    )


def z_score_normalization(data: np.array) -> np.array:
    """
    Normalize the given data array using z-score normalization.
    Parameters:
        data (np.array): The input data array to be normalized.
    Returns:
        np.array: The normalized data array.
    """
    return (data - np.mean(data)) / np.std(data)


def reverse_z_score_normalization(
    data: np.array, original_mean, original_std
) -> np.array:
    """
    Reverse z-score normalization to transform normalized data back to original scale.

    Parameters:
        data (np.array): The normalized data array.
        original_mean: The mean value used for normalization.
        original_std: The standard deviation used for normalization.

    Returns:
        np.array: The data array transformed back to the original scale.
    """
    return data * original_std + original_mean
