"""Unit tests for deterministic CityLearn selection helpers."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from citylearn_tutorial.data.selection import (
    select_buildings,
    select_simulation_period,
)
from citylearn_tutorial.structs import SimulationPeriod


class SelectionTests(unittest.TestCase):
    """Validate selection behavior without loading a CityLearn dataset."""

    def setUp(self) -> None:
        """Create a minimal schema shared by selection tests."""
        self.schema = {
            "root_directory": "/example/data",
            "buildings": {
                "Building_1": {"carbon_intensity": "carbon.csv"},
                "Building_2": {"carbon_intensity": "carbon.csv"},
                "Building_3": {"carbon_intensity": "carbon.csv"},
            },
        }

    @patch("citylearn_tutorial.data.selection.DataSet.get_schema")
    def test_building_selection_is_reproducible_without_global_rng_mutation(
        self, get_schema
    ) -> None:
        """Use a local seeded generator rather than changing NumPy global state."""
        get_schema.return_value = self.schema
        np.random.seed(17)
        expected_next_value = np.random.random()
        np.random.seed(17)

        first = select_buildings("example", 2, 0, ["Building_3"])
        second = select_buildings("example", 2, 0, ["Building_3"])

        self.assertEqual(first, second)
        self.assertEqual(first, ["Building_1", "Building_2"])
        self.assertEqual(np.random.random(), expected_next_value)

    @patch("citylearn_tutorial.data.selection.pd.read_csv")
    @patch("citylearn_tutorial.data.selection.DataSet.get_schema")
    def test_period_selection_stays_within_source_data(
        self, get_schema, read_csv
    ) -> None:
        """Choose only whole-day periods that fit the source time series."""
        get_schema.return_value = self.schema
        read_csv.return_value = pd.DataFrame({"carbon_intensity": range(49)})

        period = select_simulation_period("example", 1, 0, "Building_1")

        self.assertEqual(period.time_steps, 24)
        self.assertLess(period.end_time_step, 49)

        with self.assertRaises(ValueError):
            select_simulation_period(
                "example",
                1,
                0,
                "Building_1",
                periods_to_exclude=[SimulationPeriod(0, 23), SimulationPeriod(24, 47)],
            )
