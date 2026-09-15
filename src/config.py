"""Project-wide defaults for CityLearn control experiments."""

from dataclasses import dataclass, field
from pathlib import Path

from citylearn_tutorial.structs.experiments import SimulationConfig


@dataclass(frozen=True, slots=True)
class Settings:
    """
    Provide project paths and default CityLearn simulation settings.

    Parameters
    ----------
    default_simulation : citylearn_tutorial.structs.experiments.SimulationConfig,
        optional
        Settings used when an experiment does not provide its own simulation
        configuration. Default is the tutorial's two-building, seven-day
        centralized-control scenario.
    """

    default_simulation: SimulationConfig = field(default_factory=SimulationConfig)

    @property
    def project_root(self) -> Path:
        """
        Return the absolute repository root that contains ``src``.

        Returns
        -------
        pathlib.Path
            Absolute project root directory.
        """
        return Path(__file__).resolve().parent.parent

    @property
    def output_directory(self) -> Path:
        """
        Return the default directory for generated experiment artifacts.

        Returns
        -------
        pathlib.Path
            Directory intended for generated figures, models, and summaries.
        """
        return self.project_root / "outputs"


settings = Settings()
