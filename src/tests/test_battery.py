import pytest
import numpy as np
from src.components.battery import HEMSBattery


class TestHEMSBattery:
    def test_battery_initialization_with_valid_params(self):
        initial_soc = 0.5
        max_capacity = 50
        battery = HEMSBattery(initial_soc, max_capacity)
        assert battery.min_capacity == 0.0, "battery min capacity should be 0 kwh"
        assert battery.max_capacity == 50, "battery max capacity should be 50 kwh"
        assert (
            battery.current_soc == 0.5
        ), "battery current soc should be initial battery content"
        assert battery.energy_content == 25

    def test_initialize_battery_with_invalid_soc(self):
        initial_soc = -0.5
        max_capacity = 50

        with pytest.raises(ValueError):
            battery = HEMSBattery(initial_soc, max_capacity)

    def test_get_current_soc_with_valid_charge_rate(self):
        initial_soc = 0.5
        max_capacity = 50
        action = 10.0
        battery = HEMSBattery(initial_soc, max_capacity)

        current_soc = battery.get_current_soc(action)
        assert current_soc == 27.25 / 50

    def test_get_current_soc_with_valid_discharge_rate(self):
        initial_soc = 0.5
        max_capacity = 50
        charge_rate = -10.0
        battery = HEMSBattery(initial_soc, max_capacity)

        current_soc = battery.get_current_soc(charge_rate)
        assert current_soc == 22.75 / 50
