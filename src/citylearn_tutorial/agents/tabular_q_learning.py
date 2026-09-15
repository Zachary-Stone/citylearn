"""Train and evaluate tabular Q-learning CityLearn control experiments."""

from collections.abc import Mapping

import numpy as np
from citylearn.agents.q_learning import TabularQLearning
from citylearn.wrappers import TabularQLearningWrapper

from citylearn_tutorial.agents.baselines import run_episode
from citylearn_tutorial.data.selection import select_simulation
from citylearn_tutorial.environment import build_environment
from citylearn_tutorial.structs.experiments import (
    ExperimentConfig,
    ExperimentResult,
    SimulationConfig,
    SimulationPeriod,
)

DEFAULT_OBSERVATION_BINS = {"hour": 24}
DEFAULT_ACTION_BINS = {"electrical_storage": 12}
DEFAULT_TQL_KWARGS = {
    "epsilon": 1.0,
    "minimum_epsilon": 0.01,
    "epsilon_decay": 0.0001,
    "learning_rate": 0.005,
    "discount_factor": 0.99,
}


class CompatibleTabularQLearningWrapper(TabularQLearningWrapper):
    """
    Provide the observation names required by CityLearn's Q-learning agent.

    CityLearn 2.2b0 exposes ``observation_names`` on the wrapped environment
    but not on ``TabularQLearningWrapper`` itself. This adapter keeps the
    compatibility behaviour local to tabular-Q-learning workflows.
    """

    @property
    def observation_names(self) -> list[list[str]]:
        """
        Return observation names from the underlying CityLearn environment.

        Returns
        -------
        list[list[str]]
            Per-agent active observation names.
        """
        return self.unwrapped.observation_names


def build_tabular_environment(
    simulation: SimulationConfig,
    buildings: list[str],
    period: SimulationPeriod,
    observation_bins: Mapping[str, int] = DEFAULT_OBSERVATION_BINS,
    action_bins: Mapping[str, int] = DEFAULT_ACTION_BINS,
) -> CompatibleTabularQLearningWrapper:
    """
    Construct a CityLearn environment with discrete observation and action spaces.

    Parameters
    ----------
    simulation : SimulationConfig
        Dataset and environment settings for the experiment.
    buildings : list[str]
        Building names enabled in the environment.
    period : SimulationPeriod
        Inclusive source-data time-step bounds for the environment.
    observation_bins : collections.abc.Mapping[str, int], optional
        Discretization bin count for each active observation. Default is 24
        bins for ``hour``.
    action_bins : collections.abc.Mapping[str, int], optional
        Discretization bin count for each active action. Default is 12 bins
        for electrical storage.

    Returns
    -------
    CompatibleTabularQLearningWrapper
        Discretized environment compatible with ``TabularQLearning``.
    """
    environment = build_environment(simulation, buildings, period)
    observation_bin_sizes = [dict(observation_bins) for _ in environment.buildings]
    action_bin_sizes = [dict(action_bins) for _ in environment.buildings]
    return CompatibleTabularQLearningWrapper(
        environment,
        observation_bin_sizes=observation_bin_sizes,
        action_bin_sizes=action_bin_sizes,
    )


def calculate_training_episodes(
    environment: CompatibleTabularQLearningWrapper, exploration_multiplier: int = 3
) -> int:
    """
    Calculate the notebook's state-action exploration budget in episodes.

    Parameters
    ----------
    environment : CompatibleTabularQLearningWrapper
        Discretized environment that defines Q-table dimensions.
    exploration_multiplier : int, optional
        Desired visits per state-action pair before converting to episodes.
        Default is 3.

    Returns
    -------
    int
        Positive number of Q-learning training episodes.
    """
    if exploration_multiplier < 1:
        raise ValueError("exploration_multiplier must be at least 1.")
    state_count = environment.observation_space[0].n
    action_count = environment.action_space[0].n
    return max(
        1,
        int(
            state_count
            * action_count
            * exploration_multiplier
            / (environment.unwrapped.time_steps - 1)
        ),
    )


def q_table_diagnostics(
    model: TabularQLearning, agent_index: int = 0
) -> dict[str, np.ndarray]:
    """
    Return Q-value and exploration-count tables for one controlled agent.

    Parameters
    ----------
    model : citylearn.agents.q_learning.TabularQLearning
        Trained Q-learning model.
    agent_index : int, optional
        Controlled-agent index. Default is 0.

    Returns
    -------
    dict[str, numpy.ndarray]
        Q-values and exploration/exploitation-count matrices.
    """
    return {
        "q_values": model.q[agent_index],
        "exploration_counts": model.q_exploration[agent_index],
        "exploitation_counts": model.q_exploitation[agent_index],
    }


def epsilon_schedule(model: TabularQLearning, episodes: int = 100_000) -> np.ndarray:
    """
    Calculate the model's configured epsilon value at each training episode.

    Parameters
    ----------
    model : citylearn.agents.q_learning.TabularQLearning
        Q-learning model whose epsilon hyperparameters are used.
    episodes : int, optional
        Number of episode values to calculate. Default is 100,000.

    Returns
    -------
    numpy.ndarray
        Epsilon values bounded below by the model's minimum epsilon.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    indices = np.arange(episodes)
    return np.maximum(
        model.minimum_epsilon,
        model.epsilon_init * np.exp(-model.epsilon_decay * indices),
    )


def run_tabular_q_learning(
    simulation: SimulationConfig,
    episodes: int | None = None,
    observation_bins: Mapping[str, int] = DEFAULT_OBSERVATION_BINS,
    action_bins: Mapping[str, int] = DEFAULT_ACTION_BINS,
    name: str = "Tabular Q-Learning",
    **agent_kwargs: float,
) -> ExperimentResult[CompatibleTabularQLearningWrapper, TabularQLearning]:
    """
    Train and deterministically evaluate a tabular Q-learning controller.

    Parameters
    ----------
    simulation : SimulationConfig
        Dataset and environment settings for the experiment.
    episodes : int or None, optional
        Training episode count. When omitted, uses the notebook's state-action
        exploration budget.
    observation_bins : collections.abc.Mapping[str, int], optional
        Discretization bin count for each active observation.
    action_bins : collections.abc.Mapping[str, int], optional
        Discretization bin count for each active action.
    name : str, optional
        Human-readable experiment name. Default is ``"Tabular Q-Learning"``.
    **agent_kwargs : float
        Hyperparameter overrides accepted by ``TabularQLearning``.

    Returns
    -------
    ExperimentResult[CompatibleTabularQLearningWrapper, TabularQLearning]
        Completed trained environment, model, deterministic actions, rewards,
        and Q-learning metadata.
    """
    buildings, period = select_simulation(simulation)
    environment = build_tabular_environment(
        simulation, buildings, period, observation_bins, action_bins
    )
    episodes = (
        calculate_training_episodes(environment) if episodes is None else episodes
    )
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    hyperparameters = {**DEFAULT_TQL_KWARGS, **agent_kwargs}
    model = TabularQLearning(
        environment, random_seed=simulation.random_seed, **hyperparameters
    )
    model.learn(episodes=episodes)
    actions = run_episode(environment, model, deterministic=True)
    rewards = [
        float(np.sum(reward["sum"])) for reward in environment.unwrapped.episode_rewards
    ]
    return ExperimentResult(
        config=ExperimentConfig(name=name, simulation=simulation),
        period=period,
        environment=environment,
        model=model,
        actions=actions,
        episode_rewards=rewards,
        metadata={"episodes": episodes, "agent_kwargs": hyperparameters},
    )
