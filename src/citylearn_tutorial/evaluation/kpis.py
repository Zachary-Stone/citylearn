"""Normalize CityLearn evaluation metrics for experiment comparisons."""

import pandas as pd
from citylearn.citylearn import CityLearnEnv

KPI_NAMES = {
    "cost_total": "Cost",
    "carbon_emissions_total": "Emissions",
    "daily_peak_average": "Avg. daily peak",
    "ramping_average": "Ramping",
    "monthly_one_minus_load_factor_average": "1 - load factor",
}


def get_kpis(environment: CityLearnEnv) -> pd.DataFrame:
    """
    Return the tutorial's normalized KPI table for a completed environment.

    Parameters
    ----------
    environment : citylearn.citylearn.CityLearnEnv
        Completed CityLearn environment, optionally wrapped by a compatible
        Gymnasium wrapper.

    Returns
    -------
    pandas.DataFrame
        Selected KPI rows with a human-readable ``kpi`` column and values
        rounded to two decimal places.
    """
    kpis = environment.unwrapped.evaluate().copy()
    kpis = kpis[kpis["cost_function"].isin(KPI_NAMES)].dropna().copy()
    kpis["kpi"] = kpis["cost_function"].map(KPI_NAMES)
    kpis["value"] = kpis["value"].round(2)
    return kpis.drop(columns="cost_function")
