"""Select reproducible CityLearn buildings and simulation periods."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from citylearn.data import DataSet

from citylearn_tutorial.structs.experiments import SimulationConfig, SimulationPeriod


def _building_sort_key(building_name: str) -> tuple[int, int, str]:
    """
    Return a stable sort key for CityLearn building names.

    Parameters
    ----------
    building_name : str
        Building name from a CityLearn dataset schema.

    Returns
    -------
    tuple[int, int, str]
        Key that sorts names ending in a numeric suffix by that suffix, then
        sorts all other names lexicographically.
    """
    prefix, separator, suffix = building_name.rpartition("_")
    if separator and prefix and suffix.isdecimal():
        return 0, int(suffix), building_name
    return 1, 0, building_name


def select_buildings(
    dataset_name: str,
    count: int,
    seed: int,
    excluded_buildings: Iterable[str] = (),
) -> list[str]:
    """
    Randomly select and order buildings from a CityLearn dataset.

    The legacy ``RandomState`` generator deliberately reproduces the notebook's
    seed-to-selection behaviour without modifying NumPy's global random state.

    Parameters
    ----------
    dataset_name : str
        CityLearn dataset whose schema contains the candidate buildings.
    count : int
        Number of buildings to select.
    seed : int
        Seed for the local pseudo-random number generator.
    excluded_buildings : collections.abc.Iterable[str], optional
        Building names removed from the candidate pool. Default is an empty
        iterable.

    Returns
    -------
    list[str]
        Selected building names in stable numeric-name order.

    Raises
    ------
    ValueError
        If the requested count is invalid, the seed is negative, or fewer than
        ``count`` eligible buildings are available.
    """
    if count < 1:
        raise ValueError("count must be at least 1.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    excluded = set(excluded_buildings)
    schema = DataSet.get_schema(dataset_name)
    available = [
        building_name
        for building_name in schema["buildings"]
        if building_name not in excluded
    ]
    if count > len(available):
        raise ValueError(
            f"Requested {count} buildings but only {len(available)} are available."
        )

    generator = np.random.RandomState(seed)
    selected = generator.choice(available, size=count, replace=False).tolist()
    return sorted(selected, key=_building_sort_key)


def _as_period(period: SimulationPeriod | tuple[int, int]) -> SimulationPeriod:
    """
    Convert supported period representations to a validated period object.

    Parameters
    ----------
    period : SimulationPeriod or tuple[int, int]
        Existing period or ``(start_time_step, end_time_step)`` tuple.

    Returns
    -------
    SimulationPeriod
        Validated inclusive simulation period.
    """
    if isinstance(period, SimulationPeriod):
        return period
    return SimulationPeriod(*period)


def select_simulation_period(
    dataset_name: str,
    day_count: int,
    seed: int,
    reference_building: str,
    periods_to_exclude: Iterable[SimulationPeriod | tuple[int, int]] = (),
) -> SimulationPeriod:
    """
    Select a whole-day CityLearn simulation period that fits the dataset.

    Candidate periods are non-overlapping and always remain within the source
    time-series bounds. This avoids the notebook's dependency on global
    variables and its possibility of choosing an end time past the data.

    Parameters
    ----------
    dataset_name : str
        CityLearn dataset that supplies the carbon-intensity time series.
    day_count : int
        Number of complete 24-hour days in the selected period.
    seed : int
        Seed for the local pseudo-random number generator.
    reference_building : str
        Building whose carbon-intensity file determines the available length.
    periods_to_exclude : collections.abc.Iterable[
        SimulationPeriod or tuple[int, int]
    ], optional
        Candidate periods excluded from random selection. Default is an empty
        iterable.

    Returns
    -------
    SimulationPeriod
        Randomly selected inclusive simulation period.

    Raises
    ------
    ValueError
        If the requested period cannot fit the dataset or all valid candidates
        are excluded.
    """
    if not 1 <= day_count <= 365:
        raise ValueError("day_count must be between 1 and 365.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")

    schema = DataSet.get_schema(dataset_name)
    try:
        carbon_intensity_file = schema["buildings"][reference_building][
            "carbon_intensity"
        ]
    except KeyError as error:
        raise ValueError(
            f"Building {reference_building!r} is not available in {dataset_name!r}."
        ) from error

    file_path = Path(schema["root_directory"]) / carbon_intensity_file
    time_steps = len(pd.read_csv(file_path, usecols=[0]))
    period_time_steps = 24 * day_count
    candidate_starts = range(0, time_steps - period_time_steps + 1, period_time_steps)
    candidates = [
        SimulationPeriod(start, start + period_time_steps - 1)
        for start in candidate_starts
    ]
    excluded = {_as_period(period) for period in periods_to_exclude}
    candidates = [period for period in candidates if period not in excluded]
    if not candidates:
        raise ValueError("No valid simulation periods are available for selection.")

    generator = np.random.RandomState(seed)
    return candidates[int(generator.choice(len(candidates)))]


def select_simulation(config: SimulationConfig) -> tuple[list[str], SimulationPeriod]:
    """
    Select buildings and a corresponding reproducible simulation period.

    Parameters
    ----------
    config : SimulationConfig
        Dataset, building-selection, and period-selection settings.

    Returns
    -------
    tuple[list[str], SimulationPeriod]
        Selected building names and a valid inclusive simulation period.
    """
    buildings = select_buildings(
        dataset_name=config.dataset_name,
        count=config.building_count,
        seed=config.random_seed,
        excluded_buildings=config.excluded_buildings,
    )
    period = select_simulation_period(
        dataset_name=config.dataset_name,
        day_count=config.day_count,
        seed=config.random_seed,
        reference_building=buildings[0],
    )
    return buildings, period
