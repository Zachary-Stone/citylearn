"""Load TOML experiment settings and run selected CityLearn workflows."""

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from citylearn_tutorial.agents import (
    DEFAULT_HOUR_ACTION_MAP,
    run_baseline,
    run_hour_rbc,
    run_random_agent,
    run_sac,
    run_tabular_q_learning,
)
from citylearn_tutorial.evaluation import get_kpis, plot_simulation_summary
from citylearn_tutorial.rewards import CustomReward
from citylearn_tutorial.structs import ExperimentResult, SimulationConfig

VALID_EXPERIMENTS = frozenset(
    {"baseline", "random", "rbc", "tabular_q_learning", "sac"}
)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Define selected experiments and their TOML-configured settings."""

    experiments: tuple[str, ...]
    simulation: SimulationConfig
    output_directory: Path
    show_figures: bool
    tabular_q_learning_episodes: int
    sac_episodes: int
    sac_reward_function: str
    sac_agent_kwargs: Mapping[str, Any]


def load_run_config(path: Path) -> RunConfig:
    """
    Load a CityLearn run configuration from a TOML file.

    Parameters
    ----------
    path : pathlib.Path
        TOML configuration file to load.

    Returns
    -------
    RunConfig
        Parsed experiment, simulation, output, and training settings.
    """
    with path.open("rb") as file:
        raw = tomllib.load(file)
    run = raw["run"]
    simulation = raw["simulation"]
    tql = raw.get("tabular_q_learning", {})
    sac = raw.get("sac", {})
    experiments = tuple(run["experiments"])
    invalid = set(experiments).difference(VALID_EXPERIMENTS)
    if not experiments or invalid:
        raise ValueError(f"Unknown or empty experiment selection: {sorted(invalid)}")
    reward_function = sac.get("reward_function", "custom")
    if reward_function not in {"custom", "default"}:
        raise ValueError("sac.reward_function must be 'custom' or 'default'.")
    return RunConfig(
        experiments=experiments,
        simulation=SimulationConfig(
            dataset_name=simulation["dataset_name"],
            random_seed=simulation["random_seed"],
            building_count=simulation["building_count"],
            day_count=simulation["day_count"],
            active_observations=tuple(simulation["active_observations"]),
            central_agent=simulation["central_agent"],
        ),
        output_directory=(path.parent / run["output_directory"]).resolve(),
        show_figures=run["show_figures"],
        tabular_q_learning_episodes=tql.get("episodes", 100),
        sac_episodes=sac.get("episodes", 50),
        sac_reward_function=reward_function,
        sac_agent_kwargs=sac.get("agent_kwargs", {}),
    )


def run_experiments(config: RunConfig) -> dict[str, ExperimentResult]:
    """
    Run all workflows selected in a parsed TOML configuration.

    Parameters
    ----------
    config : RunConfig
        Parsed selection and experiment settings.

    Returns
    -------
    dict[str, ExperimentResult]
        Completed results keyed by their configured experiment identifier.
    """
    results = {}
    for experiment in config.experiments:
        if experiment == "baseline":
            results[experiment] = run_baseline(config.simulation)
        elif experiment == "random":
            results[experiment] = run_random_agent(config.simulation)
        elif experiment == "rbc":
            results[experiment] = run_hour_rbc(
                config.simulation, DEFAULT_HOUR_ACTION_MAP
            )
        elif experiment == "tabular_q_learning":
            results[experiment] = run_tabular_q_learning(
                config.simulation, episodes=config.tabular_q_learning_episodes
            )
        else:
            reward_function = (
                CustomReward if config.sac_reward_function == "custom" else None
            )
            results[experiment] = run_sac(
                config.simulation,
                episodes=config.sac_episodes,
                reward_function=reward_function,
                agent_kwargs=config.sac_agent_kwargs,
            )
    return results


def save_results(
    results: Mapping[str, ExperimentResult], output_directory: Path
) -> None:
    """
    Save KPI tables and standard comparison figures for completed experiments.

    Parameters
    ----------
    results : collections.abc.Mapping[str, ExperimentResult]
        Completed experiment results to serialize.
    output_directory : pathlib.Path
        Directory where CSV and PNG artifacts are written.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    kpis = []
    environments = {}
    for name, result in results.items():
        table = get_kpis(result.environment)
        table.insert(0, "experiment", name)
        kpis.append(table)
        environments[name] = result.environment
    pd.concat(kpis, ignore_index=True).to_csv(
        output_directory / "kpis.csv", index=False
    )
    for name, figure in plot_simulation_summary(environments).items():
        figure.savefig(output_directory / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
