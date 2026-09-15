"""Typed configurations and results shared by CityLearn experiments."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """
    Define the CityLearn environment settings shared by experiments.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the CityLearn dataset to load. Default is
        ``"citylearn_challenge_2022_phase_all"``.
    random_seed : int, optional
        Seed used for deterministic building and time-period selection.
        Default is 0.
    building_count : int, optional
        Number of buildings selected for an experiment. Default is 2.
    day_count : int, optional
        Number of simulated days. Default is 7.
    active_observations : tuple[str, ...], optional
        CityLearn observations exposed to the control agent. Default is
        ``("hour",)``.
    central_agent : bool, optional
        Whether one agent controls all selected buildings. Default is True.
    excluded_buildings : tuple[str, ...], optional
        Buildings excluded from selection because their data is outside this
        tutorial's scope. Default is ``("Building_12", "Building_15")``.

    Raises
    ------
    ValueError
        If the dataset name or active observations are empty, or if the
        building count and day count are outside the tutorial's supported
        ranges.
    """

    dataset_name: str = "citylearn_challenge_2022_phase_all"
    random_seed: int = 0
    building_count: int = 2
    day_count: int = 7
    active_observations: tuple[str, ...] = ("hour",)
    central_agent: bool = True
    excluded_buildings: tuple[str, ...] = ("Building_12", "Building_15")

    def __post_init__(self) -> None:
        """
        Validate simulation settings that do not require loading a dataset.

        Raises
        ------
        ValueError
            If one or more settings are invalid.
        """
        if not self.dataset_name:
            raise ValueError("dataset_name must not be empty.")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")
        if not 1 <= self.building_count <= 15:
            raise ValueError("building_count must be between 1 and 15.")
        if not 1 <= self.day_count <= 365:
            raise ValueError("day_count must be between 1 and 365.")
        if not self.active_observations:
            raise ValueError("active_observations must not be empty.")


@dataclass(frozen=True, slots=True)
class SimulationPeriod:
    """
    Define an inclusive range of CityLearn simulation time steps.

    Parameters
    ----------
    start_time_step : int
        First time step included in the simulation.
    end_time_step : int
        Last time step included in the simulation.

    Raises
    ------
    ValueError
        If either time step is negative or the end precedes the start.
    """

    start_time_step: int
    end_time_step: int

    def __post_init__(self) -> None:
        """
        Validate the inclusive simulation-period bounds.

        Raises
        ------
        ValueError
            If the period bounds are invalid.
        """
        if self.start_time_step < 0:
            raise ValueError("start_time_step must be non-negative.")
        if self.end_time_step < self.start_time_step:
            raise ValueError("end_time_step must not precede start_time_step.")

    @property
    def time_steps(self) -> int:
        """
        Return the number of time steps in the inclusive period.

        Returns
        -------
        int
            Number of time steps from start through end, inclusive.
        """
        return self.end_time_step - self.start_time_step + 1


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """
    Identify an experiment and its common CityLearn simulation settings.

    Parameters
    ----------
    name : str
        Human-readable experiment identifier.
    simulation : SimulationConfig
        CityLearn settings used to construct the experiment environment.

    Raises
    ------
    ValueError
        If ``name`` is empty.
    """

    name: str
    simulation: SimulationConfig

    def __post_init__(self) -> None:
        """
        Validate the experiment identifier.

        Raises
        ------
        ValueError
            If the experiment name is empty.
        """
        if not self.name:
            raise ValueError("name must not be empty.")


@dataclass(slots=True)
class ExperimentResult[EnvironmentT, ModelT]:
    """
    Store the outputs produced by a completed CityLearn experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment settings used to produce this result.
    period : SimulationPeriod
        Concrete time period selected for the environment.
    environment : EnvironmentT
        CityLearn environment after training or inference.
    model : ModelT or None
        Control model used in the experiment. Baseline experiments may not
        retain a trainable model.
    actions : list[list[float]], optional
        Ordered action vectors taken during deterministic evaluation. Default
        is an empty list.
    episode_rewards : list[float], optional
        Aggregate rewards recorded during training or evaluation. Default is
        an empty list.
    training_started_at : datetime.datetime or None, optional
        UTC timestamp recorded before training begins. Default is None.
    training_finished_at : datetime.datetime or None, optional
        UTC timestamp recorded after training finishes. Default is None.
    metadata : dict[str, object], optional
        Additional experiment-specific values, such as agent hyperparameters.
        Default is an empty dictionary.
    """

    config: ExperimentConfig
    period: SimulationPeriod
    environment: EnvironmentT
    model: ModelT | None
    actions: list[list[float]] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    training_started_at: datetime | None = None
    training_finished_at: datetime | None = None
    metadata: dict[str, object] = field(default_factory=dict)
