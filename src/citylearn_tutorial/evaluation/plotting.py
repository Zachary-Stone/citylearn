"""Create CityLearn experiment-comparison figures without displaying them."""

from collections.abc import Mapping

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from citylearn.citylearn import CityLearnEnv
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from citylearn_tutorial.evaluation.kpis import get_kpis

EnvironmentMap = Mapping[str, CityLearnEnv]


def _daily_average(values: np.ndarray) -> np.ndarray:
    if len(values) % 24 != 0:
        raise ValueError("Daily averages require a whole number of 24-hour days.")
    return values.reshape(-1, 24).mean(axis=0)


def _require_environments(environments: EnvironmentMap) -> None:
    if not environments:
        raise ValueError("environments must not be empty.")


def plot_building_kpis(environments: EnvironmentMap) -> Figure:
    """
    Plot building-level KPIs for one or more completed experiments.

    Parameters
    ----------
    environments : collections.abc.Mapping[str, CityLearnEnv]
        Mapping from experiment label to completed environment.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one horizontal bar chart per building-level KPI.
    """
    _require_environments(environments)
    tables = []
    for label, environment in environments.items():
        kpis = get_kpis(environment)
        kpis = kpis[kpis["level"] == "building"].copy()
        kpis["experiment"] = label
        tables.append(kpis)
    data = pd.concat(tables, ignore_index=True)
    if data.empty:
        raise ValueError("No building-level KPIs are available.")

    names = data["kpi"].unique()
    columns = min(3, len(names))
    rows = int(np.ceil(len(names) / columns))
    figure, _ = plt.subplots(rows, columns, figsize=(3 * columns, 2.5 * rows))
    for index, (axis, (name, subset)) in enumerate(
        zip(figure.axes, data.groupby("kpi", sort=False), strict=True)
    ):
        sns.barplot(data=subset, x="value", y="name", hue="experiment", ax=axis)
        axis.set(title=name, xlabel=None, ylabel=None)
        for container in axis.containers:
            axis.bar_label(container, fmt="%.2f")
        legend = axis.get_legend()
        if legend is not None and index != len(names) - 1:
            legend.remove()
    figure.tight_layout()
    return figure


def plot_district_kpis(environments: EnvironmentMap) -> Figure:
    """
    Plot district-level KPIs for one or more completed experiments.

    Parameters
    ----------
    environments : collections.abc.Mapping[str, CityLearnEnv]
        Mapping from experiment label to completed environment.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing a district-level KPI comparison bar chart.
    """
    _require_environments(environments)
    tables = []
    for label, environment in environments.items():
        kpis = get_kpis(environment)
        kpis = kpis[kpis["level"] == "district"].copy()
        kpis["experiment"] = label
        tables.append(kpis)
    data = pd.concat(tables, ignore_index=True)
    if data.empty:
        raise ValueError("No district-level KPIs are available.")

    figure, axis = plt.subplots(figsize=(6, 0.45 * len(data["kpi"].unique())))
    sns.barplot(data=data, x="value", y="kpi", hue="experiment", ax=axis)
    axis.set(xlabel=None, ylabel=None)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.2f")
    figure.tight_layout()
    return figure


def plot_building_load_profiles(
    environments: EnvironmentMap, daily_average: bool = False
) -> Figure:
    """
    Plot per-building net-electricity-consumption profiles.

    Parameters
    ----------
    environments : collections.abc.Mapping[str, CityLearnEnv]
        Mapping from experiment label to completed environment.
    daily_average : bool, optional
        Whether to aggregate each full day into an average 24-hour profile.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one line plot per selected building.
    """
    _require_environments(environments)
    first_environment = next(iter(environments.values())).unwrapped
    building_count = len(first_environment.buildings)
    columns = min(4, building_count)
    rows = int(np.ceil(building_count / columns))
    figure, _ = plt.subplots(rows, columns, figsize=(4 * columns, 2.2 * rows))
    for index, axis in enumerate(figure.axes[:building_count]):
        for label, environment in environments.items():
            building = environment.unwrapped.buildings[index]
            values = np.asarray(building.net_electricity_consumption)
            values = _daily_average(values) if daily_average else values
            axis.plot(values, label=label)
            axis.set_title(building.name)
        axis.set_ylabel("kWh")
        axis.set_xlabel("Hour" if daily_average else "Time step")
        axis.xaxis.set_major_locator(ticker.MultipleLocator(2 if daily_average else 24))
    for axis in figure.axes[building_count:]:
        axis.remove()
    figure.axes[building_count - 1].legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), framealpha=0.0
    )
    figure.tight_layout()
    return figure


def plot_district_load_profiles(
    environments: EnvironmentMap, daily_average: bool = False
) -> Figure:
    """
    Plot district net-electricity-consumption profiles.

    Parameters
    ----------
    environments : collections.abc.Mapping[str, CityLearnEnv]
        Mapping from experiment label to completed environment.
    daily_average : bool, optional
        Whether to aggregate each full day into an average 24-hour profile.
        Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the district load-profile comparison.
    """
    _require_environments(environments)
    figure, axis = plt.subplots(figsize=(5, 2.5))
    for label, environment in environments.items():
        values = np.asarray(environment.unwrapped.net_electricity_consumption)
        values = _daily_average(values) if daily_average else values
        axis.plot(values, label=label)
    axis.set_ylabel("kWh")
    axis.set_xlabel("Hour" if daily_average else "Time step")
    axis.xaxis.set_major_locator(ticker.MultipleLocator(2 if daily_average else 24))
    axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), framealpha=0.0)
    figure.tight_layout()
    return figure


def plot_battery_soc_profiles(environments: EnvironmentMap) -> Figure:
    """
    Plot per-building electrical-storage state-of-charge profiles.

    Parameters
    ----------
    environments : collections.abc.Mapping[str, CityLearnEnv]
        Mapping from experiment label to completed environment.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one state-of-charge plot per selected building.
    """
    _require_environments(environments)
    first_environment = next(iter(environments.values())).unwrapped
    building_count = len(first_environment.buildings)
    columns = min(4, building_count)
    rows = int(np.ceil(building_count / columns))
    figure, _ = plt.subplots(rows, columns, figsize=(4 * columns, 2.2 * rows))
    for index, axis in enumerate(figure.axes[:building_count]):
        for label, environment in environments.items():
            building = environment.unwrapped.buildings[index]
            axis.plot(building.electrical_storage.soc, label=label)
            axis.set_title(building.name)
        axis.set(xlabel="Time step", ylabel="SoC", ylim=(0.0, 1.0))
        axis.xaxis.set_major_locator(ticker.MultipleLocator(24))
    for axis in figure.axes[building_count:]:
        axis.remove()
    figure.axes[building_count - 1].legend(
        loc="upper left", bbox_to_anchor=(1.0, 1.0), framealpha=0.0
    )
    figure.tight_layout()
    return figure


def plot_actions(
    actions: list[list[float]], building_names: list[str], title: str
) -> Figure:
    """
    Plot action time series for each building in a control experiment.

    Parameters
    ----------
    actions : list[list[float]]
        Ordered action vectors produced during one evaluation episode.
    building_names : list[str]
        Building names corresponding to action-vector columns.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing one action line per building.
    """
    data = pd.DataFrame(actions, columns=building_names)
    figure, axis = plt.subplots(figsize=(6, 2.5))
    for name in data:
        axis.plot(data[name], label=name)
    axis.set(title=title, xlabel="Time step", ylabel=r"$\frac{kWh}{kWh_{capacity}}$")
    axis.xaxis.set_major_locator(ticker.MultipleLocator(24))
    axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), framealpha=0.0)
    figure.tight_layout()
    return figure


def plot_rewards(rewards: list[float], title: str) -> Figure:
    """
    Plot aggregate reward by training episode.

    Parameters
    ----------
    rewards : list[float]
        Aggregate reward recorded for each episode.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the reward history.
    """
    figure, axis = plt.subplots(figsize=(5, 2.5))
    axis.plot(rewards)
    axis.set(title=title, xlabel="Episode", ylabel="Reward")
    figure.tight_layout()
    return figure


def plot_table(
    axis: Axes,
    table: np.ndarray,
    title: str,
    cmap: str,
    colorbar_label: str,
    xlabel: str,
    ylabel: str | None,
) -> Axes:
    """
    Plot a two-dimensional table as a heat map.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes on which to draw the heat map.
    table : numpy.ndarray
        Two-dimensional values indexed by state and action.
    title : str
        Axes title.
    cmap : str
        Matplotlib colormap name.
    colorbar_label : str
        Label for the table colorbar.
    xlabel : str
        X-axis label.
    ylabel : str or None
        Y-axis label.

    Returns
    -------
    matplotlib.axes.Axes
        The supplied axes after plotting.
    """
    mesh = axis.pcolormesh(table.T, shading="nearest", cmap=cmap)
    axis.figure.colorbar(
        mesh,
        ax=axis,
        orientation="horizontal",
        label=colorbar_label,
        fraction=0.025,
        pad=0.08,
    )
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return axis


def plot_simulation_summary(environments: EnvironmentMap) -> dict[str, Figure]:
    """
    Create all standard KPI, load, and state-of-charge comparison figures.

    Parameters
    ----------
    environments : collections.abc.Mapping[str, CityLearnEnv]
        Mapping from experiment label to completed environment.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Named figures for building and district KPI, load, and battery
        comparisons. The caller controls display or file output.
    """
    return {
        "building_kpis": plot_building_kpis(environments),
        "building_load_profiles": plot_building_load_profiles(environments),
        "building_daily_load_profiles": plot_building_load_profiles(
            environments, daily_average=True
        ),
        "battery_soc_profiles": plot_battery_soc_profiles(environments),
        "district_kpis": plot_district_kpis(environments),
        "district_load_profiles": plot_district_load_profiles(environments),
        "district_daily_load_profiles": plot_district_load_profiles(
            environments, daily_average=True
        ),
    }
