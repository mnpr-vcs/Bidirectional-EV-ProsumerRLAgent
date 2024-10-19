import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple
from pandas import Series
import random


class HouseholdEnv(gym.Env):
    """
    Home Energy Management System (Gymnasium) environment.
    """

    MAX_CHARGE_RATE = 11
    MAX_DISCHARGE_RATE = -11
    GRID_DRAW_RATE_EUR = 0.08
    GRID_FEED_RATE_EUR = 0.33
    

    def __init__(
        self,
        power_household,
        power_pv,
        auction_price,
        observation_window,
        battery,
    ):
        """
        Initializes the environment with data and parameters.
        Args:
            power_household: A pandas Series of normalized power_household data.
            power_pv: A pandas Series of normalized power_pv data.
            auction_price: A pandas Series of auction_price data.
            observation_window: An integer representing the number of timesteps
            battery : A battery of type HEMSBattery
        """
        super().__init__()

        self.power_household_o = power_household
        self.power_pv_o = power_pv
        self.auction_price_o = auction_price
        self.observation_window = observation_window
        self.battery = battery

        self.obs_start_index = 0
        self.obs_end_index = self.obs_start_index + self.observation_window
        self.power_household = self.power_household_o[
            self.obs_start_index : self.obs_end_index
        ]
        self.power_pv = self.power_pv_o[self.obs_start_index : self.obs_end_index]
        self.auction_price = self.auction_price_o[
            self.obs_start_index : self.obs_end_index
        ]

        self.current_exchange = 0.0
        self.current_cost = 0.0
        self.current_step = 0
        self.max_steps = len(self.power_household)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    power_household.min(),
                    power_pv.min(),
                    auction_price.min(),
                    0,
                ]
            ),
            high=np.array(
                [
                    power_household.max(),
                    power_pv.max(),
                    auction_price.max(),
                    1,
                ]
            ),
            shape=(4,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        """
        Returns the current state based on current and historical data.

        Returns:
            A numpy array representing the current state.
        """
        state = [
            self.power_household.iloc[self.current_step],
            self.power_pv.iloc[self.current_step],
            self.auction_price.iloc[self.current_step],
        ]
        state = np.hstack(
            [0.0 if np.isnan(x) else x for x in state[:3]] + [self.battery.current_soc]
        )

        return np.array(state, dtype=np.float32)

    def _get_info(self) -> dict:
        """
        Returns additional information about the environment.

        Returns:
            A dictionary containing any 'exchange', 'cost' information.
        """
        return {
            "exchange": np.array(self.current_exchange),
            "cost": np.array(self.current_cost),
        }

    def _get_reward_combined(self, net_exchange: float) -> float:
        """
        Calculates the reward based on the action, state, and additional factors.

        Args:
            action: A float representing the chosen charging/discharging action.

        Returns:
            A float representing the reward.
        """
        soc_reward = self._get_soc_retain_reward()
        cost_reward = self._get_net_exchange_cost_reward(net_exchange)

        self.current_cost = cost_reward

        return float(cost_reward + soc_reward) # add cost rwd as +ve

    def _get_net_exchange_cost_reward(self, net_exchange: float) -> float:
        """
        Calculates the reward based on the action, state, and additional factors.

        Args:
            net_exchange:: A float representing the net power exchange to the grid.

        Returns:
            A float representing the reward.
        """
        power_household_cost = (
            max(0, net_exchange) * self.auction_price.iloc[self.current_step]
        )
        power_sell_cost = (
            min(0, net_exchange) * self.auction_price.iloc[self.current_step]
        )

        return float((power_household_cost + power_sell_cost) * -1)
    
    def _get_constant_exchange_rate_cost_reward(self, net_exchange: float) -> float:
        """
        Calculates the reward based on the net exchange, exchange rate (EUR), and additional factors.

        Args:
            net_exchange:: A float representing the net power exchange to the grid.

        Returns:
            A float representing the reward.
        """
        # constant rate of grid feed/draw: 0.08  and 0.33 EUR/kWh resp. 
        power_household_cost = (
            max(0, net_exchange) * self.GRID_DRAW_RATE_EUR
        )
        power_sell_cost = (
            min(0, net_exchange) * self.GRID_DRAW_RATE_EUR
        )

        return float((power_household_cost + power_sell_cost) * -1)


    def _get_soc_retain_reward(self, retain_threshold=0.45):
        """retain threshold constraint on soc."""
        if self.battery.current_soc < retain_threshold:
            soc_reward = -0.25
        else:
            soc_reward = 0.25

        return soc_reward

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state.

        Args:
            seed (int, optional): Seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options for the reset. Defaults to None.

        Returns:
            Observation : A numpy array representing the initial observation.
            Info : A dictionary containing any additional information.
        """
        super().reset(seed=seed, options=options)

        self.current_step = 0
        self.current_exchange = 0.0
        self.obs_start_index = int(
            random.uniform(0, len(self.power_household_o) - self.observation_window)
        )  # set uniformly random start index for observation window

        return self._get_obs(), self._get_info()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, dict]:

        """
        Takes an action, simulates the environment, and returns the next state, reward, and done flag.

        Args:
            action: A float representing the chosen charging/discharging action.

        Returns:
            A tuple containing the next state (numpy array), reward (float), terminated (boolean), truncated (boolean), and info (dictionary).
        """
        info = self._get_info()
        observation = self._get_obs()

        truncated = False
        terminated = self.current_step >= (self.max_steps - 1)
        if terminated:
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()

        action = (action + 1) * (
            self.MAX_CHARGE_RATE - self.MAX_DISCHARGE_RATE
        ) / 2 + self.MAX_DISCHARGE_RATE
        
        # soc constraint
        if self.battery.current_soc == 0.0:  # if battery empty
            action = np.clip(action, 0, self.MAX_CHARGE_RATE)  # only charge action
        elif self.battery.current_soc == 1.0:  # if battery full
            action = np.clip(
                action, self.MAX_DISCHARGE_RATE, 0
            )  # only discharge action
        
        net_exchange = (
            self.power_household.iloc[self.current_step]
            + self.power_pv.iloc[self.current_step]
            + action
        )

        self.current_exchange = net_exchange
        self.battery.get_current_soc(action)

        # reward = self._get_constant_exchange_rate_cost_reward(net_exchange)
        # reward = self._get_reward_cost(net_exchange)
        # self.current_cost = reward
        reward = self._get_reward_combined(net_exchange)
        
        self.current_step += 1

        return self._get_obs(), reward, terminated, truncated, self._get_info()
