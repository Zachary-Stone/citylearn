"""Reward functions for CityLearn control experiments."""

from typing import Any

import numpy as np
from citylearn.reward_function import RewardFunction


class CustomReward(RewardFunction):
    """
    Reward battery charging from export and discharging during grid imports.

    Parameters
    ----------
    env_metadata : dict[str, Any]
        Static CityLearn environment metadata supplied by the environment.
    """

    def __init__(self, env_metadata: dict[str, Any]):
        """
        Initialize the reward function with CityLearn environment metadata.

        Parameters
        ----------
        env_metadata : dict[str, Any]
            Static CityLearn environment metadata.
        """
        super().__init__(env_metadata)

    def calculate(self, observations: list[dict[str, int | float]]) -> list[float]:
        """
        Calculate the aggregate custom reward for the current time step.

        Parameters
        ----------
        observations : list[dict[str, int | float]]
            Current observations for all controlled buildings.

        Returns
        -------
        list[float]
            Single district-level reward that penalizes costly grid imports
            while stored energy remains and penalizes exports before batteries
            are full.
        """
        rewards = []
        for observation in observations:
            cost = (
                observation["net_electricity_consumption"]
                * observation["electricity_pricing"]
            )
            battery_soc = observation["electrical_storage_soc"]
            penalty = -(1.0 + np.sign(cost) * battery_soc)
            rewards.append(float(penalty * abs(cost)))
        return [sum(rewards)]
