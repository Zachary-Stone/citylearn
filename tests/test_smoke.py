"""Fast end-to-end smoke test for the baseline CityLearn workflow."""

import unittest
from dataclasses import replace

from citylearn_tutorial.agents import run_baseline
from citylearn_tutorial.evaluation import get_kpis
from config import settings


class BaselineSmokeTests(unittest.TestCase):
    """Exercise the real dataset through a short completed episode."""

    def test_baseline_experiment_completes_one_day_scenario(self) -> None:
        """Construct, run, and evaluate a deterministic one-day baseline."""
        simulation = replace(settings.default_simulation, day_count=1)
        result = run_baseline(simulation)

        self.assertTrue(result.environment.terminated)
        self.assertEqual(len(result.actions), result.period.time_steps - 1)
        self.assertFalse(get_kpis(result.environment).empty)
