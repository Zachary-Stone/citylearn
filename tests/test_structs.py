"""Unit tests for typed CityLearn experiment structures."""

import unittest

from citylearn_tutorial.structs import (
    ExperimentConfig,
    ExperimentResult,
    SimulationConfig,
    SimulationPeriod,
)


class SimulationStructTests(unittest.TestCase):
    """Validate configuration and result data structures."""

    def test_default_simulation_config_matches_tutorial_scenario(self) -> None:
        """Use the original notebook's default two-building configuration."""
        config = SimulationConfig()

        self.assertEqual(config.building_count, 2)
        self.assertEqual(config.day_count, 7)
        self.assertEqual(config.active_observations, ("hour",))
        self.assertTrue(config.central_agent)

    def test_invalid_simulation_values_are_rejected(self) -> None:
        """Reject invalid values before a CityLearn environment is built."""
        invalid_configs = [
            {"dataset_name": ""},
            {"random_seed": -1},
            {"building_count": 0},
            {"day_count": 0},
            {"active_observations": ()},
        ]

        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SimulationConfig(**kwargs)

    def test_result_keeps_typed_experiment_metadata(self) -> None:
        """Store actions and metadata alongside a selected simulation period."""
        config = ExperimentConfig("Baseline", SimulationConfig())
        period = SimulationPeriod(0, 23)
        result = ExperimentResult(
            config=config,
            period=period,
            environment=object(),
            model=None,
            actions=[[0.0]],
        )

        self.assertEqual(period.time_steps, 24)
        self.assertEqual(result.actions, [[0.0]])

    def test_invalid_period_and_experiment_name_are_rejected(self) -> None:
        """Reject invalid period bounds and empty experiment identifiers."""
        with self.assertRaises(ValueError):
            SimulationPeriod(-1, 0)
        with self.assertRaises(ValueError):
            SimulationPeriod(1, 0)
        with self.assertRaises(ValueError):
            ExperimentConfig("", SimulationConfig())
