import numpy as np


class HEMSBattery:
    """
    The `Battery` class represents a battery in a Battery Management System (BMS).
    It tracks the state of charge (SOC) of the battery and provides methods to charge and discharge the battery.
    """

    DISCHARGING_EFFICIENCY = 0.9
    CHARGING_EFFICIENCY = 0.9

    def __init__(
        self,
        initial_soc: float,
        max_capacity: float,
    ):
        """
        Initializes the battery with the given initial SOC, charge rate, discharge rate, and capacity.
        Args:
            - soc: The initial state of charge (SOC) of the battery.
            - max_capacity: The maximum capacity of the battery in kw.
        """
        self.min_capacity = 0.0
        self.max_capacity = max_capacity
        self.current_soc = initial_soc
        self.energy_content = self.max_capacity * initial_soc

        if not (1 >= initial_soc >= 0):
            raise ValueError("soc must be within 0,1 range")

    def get_current_soc(self, action):
        """
        Method to get the current SoC of the battery.
        Args:
            - charge_discharge_rate ( float ): charge( +ve ) or discharge( -ve ) rate
        Returns:
            float: the current state of charge (SOC).
        """
        energy_flow = action / 4  # energy to add or substract
        if action > 0:
            energy_flow *= self.CHARGING_EFFICIENCY
            self.energy_content = min(
                self.energy_content + energy_flow, self.max_capacity
            )
        else:
            energy_flow *= self.DISCHARGING_EFFICIENCY
            self.energy_content = max(
                self.energy_content + energy_flow, self.min_capacity
            )
        self.current_soc = self.energy_content / self.max_capacity
        return self.current_soc
