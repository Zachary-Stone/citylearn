"""Construct and inspect CityLearn environments."""

from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

from citylearn_tutorial.structs.experiments import SimulationConfig, SimulationPeriod


def build_environment(
    config: SimulationConfig,
    buildings: list[str],
    period: SimulationPeriod,
    reward_function: type[RewardFunction] | None = None,
) -> CityLearnEnv:
    """
    Construct a CityLearn environment for one selected simulation period.

    Parameters
    ----------
    config : SimulationConfig
        Shared dataset and environment settings.
    buildings : list[str]
        Building names enabled in the environment.
    period : SimulationPeriod
        Inclusive source-data time-step bounds for the environment.
    reward_function : type[citylearn.reward_function.RewardFunction], optional
        CityLearn reward-function class. When omitted, CityLearn uses its
        default reward function.

    Returns
    -------
    citylearn.citylearn.CityLearnEnv
        Configured, unwrapped CityLearn environment.

    Raises
    ------
    ValueError
        If no buildings are provided.
    """
    if not buildings:
        raise ValueError("buildings must not be empty.")

    environment_kwargs: dict[str, object] = {
        "buildings": buildings,
        "central_agent": config.central_agent,
        "active_observations": list(config.active_observations),
        "simulation_start_time_step": period.start_time_step,
        "simulation_end_time_step": period.end_time_step,
        "random_seed": config.random_seed,
    }
    if reward_function is not None:
        environment_kwargs["reward_function"] = reward_function
    return CityLearnEnv(config.dataset_name, **environment_kwargs)


def build_stable_baselines_environment(
    config: SimulationConfig,
    buildings: list[str],
    period: SimulationPeriod,
    reward_function: type[RewardFunction] | None = None,
) -> StableBaselines3Wrapper:
    """
    Construct a CityLearn environment compatible with Stable Baselines3.

    Observations are normalized before the Stable Baselines3 interface wrapper
    is applied, matching the preprocessing sequence in the original tutorial.

    Parameters
    ----------
    config : SimulationConfig
        Shared dataset and environment settings.
    buildings : list[str]
        Building names enabled in the environment.
    period : SimulationPeriod
        Inclusive source-data time-step bounds for the environment.
    reward_function : type[citylearn.reward_function.RewardFunction], optional
        CityLearn reward-function class. When omitted, CityLearn uses its
        default reward function.

    Returns
    -------
    citylearn.wrappers.StableBaselines3Wrapper
        Normalized CityLearn environment exposed through the Stable Baselines3
        interface.
    """
    environment = build_environment(config, buildings, period, reward_function)
    normalized_environment = NormalizedObservationWrapper(environment)
    return StableBaselines3Wrapper(normalized_environment)


def describe_environment(environment: CityLearnEnv) -> dict[str, object]:
    """
    Return a serializable summary of an initialized CityLearn environment.

    Parameters
    ----------
    environment : citylearn.citylearn.CityLearnEnv
        Unwrapped CityLearn environment to inspect.

    Returns
    -------
    dict[str, object]
        Current time step, environment dimensions, configuration flags, and
        per-building storage, photovoltaic, observation, and action details.
    """
    building_details: dict[str, dict[str, object]] = {}
    for building in environment.buildings:
        electrical_storage = building.electrical_storage
        building_details[building.name] = {
            "electrical_storage_capacity": electrical_storage.capacity,
            "electrical_storage_nominal_power": electrical_storage.nominal_power,
            "electrical_storage_loss_coefficient": electrical_storage.loss_coefficient,
            "electrical_storage_soc": electrical_storage.soc[building.time_step],
            "electrical_storage_efficiency": electrical_storage.efficiency,
            "electrical_storage_electricity_consumption": (
                electrical_storage.electricity_consumption[building.time_step]
            ),
            "electrical_storage_capacity_loss_coefficient": (
                electrical_storage.capacity_loss_coefficient
            ),
            "pv_nominal_power": building.pv.nominal_power,
            "active_observations": building.active_observations,
            "active_actions": building.active_actions,
        }

    return {
        "time_step": environment.time_step,
        "time_steps": environment.time_steps,
        "central_agent": environment.central_agent,
        "building_count": len(environment.buildings),
        "buildings": building_details,
    }
