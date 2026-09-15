"""Unit tests for CityLearn reward functions."""

import unittest

from citylearn_tutorial.rewards import CustomReward


class CustomRewardTests(unittest.TestCase):
    """Validate custom-reward cost and battery behavior."""

    def test_reward_penalizes_grid_import_with_available_storage(self) -> None:
        """Apply the notebook reward formula to a positive-cost observation."""
        reward = CustomReward({"buildings": []}).calculate(
            [
                {
                    "net_electricity_consumption": 2.0,
                    "electricity_pricing": 0.5,
                    "electrical_storage_soc": 0.5,
                }
            ]
        )

        self.assertEqual(reward, [-1.5])
