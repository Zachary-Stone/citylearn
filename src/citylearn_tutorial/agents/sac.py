"""Train and evaluate Stable-Baselines SAC CityLearn controllers."""

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import numpy as np
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import StableBaselines3Wrapper
from stable_baselines3 import SAC

from citylearn_tutorial.data.selection import select_simulation
from citylearn_tutorial.environment import build_stable_baselines_environment
from citylearn_tutorial.rewards import CustomReward
from citylearn_tutorial.structs.experiments import (
    ExperimentConfig,
    ExperimentResult,
    SimulationConfig,
)


def run_sac(
    simulation: SimulationConfig,
    episodes: int,
    reward_function: type[RewardFunction] | None = CustomReward,
    name: str = "SAC",
    agent_kwargs: Mapping[str, Any] | None = None,
) -> ExperimentResult[StableBaselines3Wrapper, SAC]:
    """
    Train and deterministically evaluate a Soft Actor-Critic controller.

    Parameters
    ----------
    simulation : SimulationConfig
        Dataset and environment settings for the experiment.
    episodes : int
        Number of full environment episodes used for SAC training.
    reward_function : type[citylearn.reward_function.RewardFunction] or None, optional
        Reward-function class used by CityLearn. Pass None for CityLearn's
        default reward. Default is ``CustomReward``.
    name : str, optional
        Human-readable experiment name. Default is ``"SAC"``.
    agent_kwargs : collections.abc.Mapping[str, Any] or None, optional
        Keyword arguments forwarded to ``stable_baselines3.SAC``. Default is
        None.

    Returns
    -------
    ExperimentResult[citylearn.wrappers.StableBaselines3Wrapper, SAC]
        Completed environment, trained model, evaluation actions, rewards, and
        training metadata.

    Raises
    ------
    ValueError
        If ``episodes`` is less than 1.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")

    buildings, period = select_simulation(simulation)
    environment = build_stable_baselines_environment(
        simulation, buildings, period, reward_function
    )
    model = SAC(
        policy="MlpPolicy",
        env=environment,
        seed=simulation.random_seed,
        **dict(agent_kwargs or {}),
    )
    episode_time_steps = environment.unwrapped.time_steps - 1
    training_started_at = datetime.now(UTC)
    for _ in range(episodes):
        model.learn(total_timesteps=episode_time_steps, reset_num_timesteps=False)
    training_finished_at = datetime.now(UTC)

    observations, _ = environment.reset()
    actions_list = []
    while not environment.unwrapped.terminated:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _, _ = environment.step(actions)
        actions_list.append(np.asarray(actions, dtype=float).reshape(-1).tolist())

    rewards = [
        float(np.sum(reward["sum"])) for reward in environment.unwrapped.episode_rewards
    ]
    return ExperimentResult(
        config=ExperimentConfig(name=name, simulation=simulation),
        period=period,
        environment=environment,
        model=model,
        actions=actions_list,
        episode_rewards=rewards,
        training_started_at=training_started_at,
        training_finished_at=training_finished_at,
        metadata={"episodes": episodes, "agent_kwargs": dict(agent_kwargs or {})},
    )
