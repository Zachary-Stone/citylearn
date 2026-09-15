"""Run baseline, random, and rule-based CityLearn control experiments."""

from collections.abc import Mapping

from citylearn.agents.base import Agent, BaselineAgent
from citylearn.agents.rbc import HourRBC
from citylearn.citylearn import CityLearnEnv

from citylearn_tutorial.data.selection import select_simulation
from citylearn_tutorial.environment import build_environment
from citylearn_tutorial.structs.experiments import (
    ExperimentConfig,
    ExperimentResult,
    SimulationConfig,
)

DEFAULT_HOUR_ACTION_MAP = {
    hour: 1 / 12 if hour <= 12 else -1 / 12 for hour in range(1, 25)
}


def _record_actions(actions: list[list[float]]) -> list[float]:
    """Flatten CityLearn actions into a building-aligned action vector."""
    return [float(action) for agent_actions in actions for action in agent_actions]


def run_episode(
    environment: CityLearnEnv, agent: Agent, deterministic: bool = True
) -> list[list[float]]:
    """
    Run one CityLearn episode until the environment reaches its terminal state.

    Parameters
    ----------
    environment : citylearn.citylearn.CityLearnEnv
        Initialized CityLearn environment to reset and control.
    agent : citylearn.agents.base.Agent
        Agent used to select an action from each observation.
    deterministic : bool, optional
        Whether the agent should choose deterministic actions when supported.
        Default is True.

    Returns
    -------
    list[list[float]]
        Building-aligned action vector for every environment step.
    """
    observations, _ = environment.reset()
    actions_list = []
    while not environment.terminated:
        actions = agent.predict(observations, deterministic=deterministic)
        observations, _, _, _, _ = environment.step(actions)
        actions_list.append(_record_actions(actions))
    return actions_list


def run_baseline(
    simulation: SimulationConfig, name: str = "Baseline"
) -> ExperimentResult[CityLearnEnv, BaselineAgent]:
    """
    Run CityLearn's no-control baseline for one selected simulation period.

    Parameters
    ----------
    simulation : SimulationConfig
        Dataset and environment settings for the experiment.
    name : str, optional
        Human-readable experiment name. Default is ``"Baseline"``.

    Returns
    -------
    ExperimentResult[CityLearnEnv, citylearn.agents.base.BaselineAgent]
        Completed baseline environment, model, selected period, and actions.
    """
    buildings, period = select_simulation(simulation)
    environment = build_environment(simulation, buildings, period)
    model = BaselineAgent(environment)
    actions = run_episode(environment, model)
    return ExperimentResult(
        config=ExperimentConfig(name=name, simulation=simulation),
        period=period,
        environment=environment,
        model=model,
        actions=actions,
    )


def run_random_agent(
    simulation: SimulationConfig, name: str = "Random"
) -> ExperimentResult[CityLearnEnv, Agent]:
    """
    Run CityLearn's random-action agent for one selected simulation period.

    Parameters
    ----------
    simulation : SimulationConfig
        Dataset and environment settings for the experiment.
    name : str, optional
        Human-readable experiment name. Default is ``"Random"``.

    Returns
    -------
    ExperimentResult[CityLearnEnv, citylearn.agents.base.Agent]
        Completed random-agent environment, model, selected period, and actions.
    """
    buildings, period = select_simulation(simulation)
    environment = build_environment(simulation, buildings, period)
    model = Agent(environment)
    actions = run_episode(environment, model, deterministic=False)
    return ExperimentResult(
        config=ExperimentConfig(name=name, simulation=simulation),
        period=period,
        environment=environment,
        model=model,
        actions=actions,
    )


def run_hour_rbc(
    simulation: SimulationConfig,
    action_map: Mapping[int, float] = DEFAULT_HOUR_ACTION_MAP,
    name: str = "RBC",
) -> ExperimentResult[CityLearnEnv, HourRBC]:
    """
    Run an hourly rule-based controller for one selected simulation period.

    Parameters
    ----------
    simulation : SimulationConfig
        Dataset and environment settings for the experiment.
    action_map : collections.abc.Mapping[int, float], optional
        Battery action value for each hour from 1 through 24. Default is the
        notebook's reference charge-then-discharge schedule.
    name : str, optional
        Human-readable experiment name. Default is ``"RBC"``.

    Returns
    -------
    ExperimentResult[CityLearnEnv, citylearn.agents.rbc.HourRBC]
        Completed rule-based-control environment, model, period, and actions.

    Raises
    ------
    ValueError
        If the action map does not define every hour from 1 through 24.
    """
    required_hours = set(range(1, 25))
    if set(action_map) != required_hours:
        raise ValueError("action_map must define exactly the hours 1 through 24.")

    buildings, period = select_simulation(simulation)
    environment = build_environment(simulation, buildings, period)
    model = HourRBC(environment, action_map=dict(action_map))
    actions = run_episode(environment, model)
    return ExperimentResult(
        config=ExperimentConfig(name=name, simulation=simulation),
        period=period,
        environment=environment,
        model=model,
        actions=actions,
    )
