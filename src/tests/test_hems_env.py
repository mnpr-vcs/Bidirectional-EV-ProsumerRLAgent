import numpy as np
import pandas as pd
import pytest
from gymnasium import spaces

from hems_env.envs import HouseholdEnv
from src.components.battery import HEMSBattery


@pytest.fixture
def mock_env_data():
    data_length = 10
    consumption = pd.Series(np.random.rand(data_length))
    generation = pd.Series(np.random.rand(data_length))
    auction_price = pd.Series(np.random.rand(data_length))
    observation_window = 96
    battery = HEMSBattery(initial_soc=0.1, max_capacity=50)
    return consumption, generation, auction_price, battery, observation_window


class TestHEMSEnv:
    def test_hemsenv_initialization(self, mock_env_data):
        (
            consumption,
            generation,
            auction_price,
            observation_window,
            battery,
        ) = mock_env_data
        env = HouseholdEnv(
            consumption, generation, auction_price, battery, observation_window
        )
        assert isinstance(
            env.action_space, spaces.Box
        ), "Action space should be a Box space."
        assert isinstance(
            env.observation_space, spaces.Box
        ), "Observation space should be a Box space."

    def test_hemsenv_reset(self, mock_env_data):
        (
            consumption,
            generation,
            auction_price,
            observation_window,
            battery,
        ) = mock_env_data
        env = HouseholdEnv(
            consumption, generation, auction_price, battery, observation_window
        )
        initial_state, initial_info = env.reset()
        assert env.current_step == 0, "Current step should be reset to 0."
        assert isinstance(
            initial_state, np.ndarray
        ), "Reset should return an instance of numpy array."
        assert isinstance(
            initial_info["exchange"], np.ndarray
        ), "Reset should return an instance of numpy array."

    def test_hemsenv_step(self, mock_env_data):
        (
            consumption,
            generation,
            auction_price,
            observation_window,
            battery,
        ) = mock_env_data
        env = HouseholdEnv(
            consumption, generation, auction_price, battery, observation_window
        )
        env.reset()  # Ensure the environment is at its initial state
        action = np.random.rand(1) * 22 - 11  # Random action within the action space
        next_state, reward, terminated, truncated, info = env.step(action)
        assert isinstance(next_state, np.ndarray), "Next state should be a numpy array."
        assert isinstance(reward, float), "Reward should be a float."
        assert isinstance(terminated, bool), "Terminated should be a boolean."
        assert isinstance(truncated, bool), "Truncated should be a boolean."
        assert isinstance(info, dict), "Info should be a dictionary."
