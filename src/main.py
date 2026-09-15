"""Run CityLearn experiments selected in the root TOML configuration file."""

from pathlib import Path

import matplotlib.pyplot as plt

from citylearn_tutorial.evaluation import plot_simulation_summary
from citylearn_tutorial.runner import load_run_config, run_experiments, save_results

CONFIG_PATH = Path(__file__).resolve().parent.parent / "experiment.toml"


def main() -> None:
    """Load the user-editable TOML configuration and run its experiments."""
    config = load_run_config(CONFIG_PATH)
    results = run_experiments(config)
    save_results(results, config.output_directory)
    print(f"Saved results to {config.output_directory}")
    if config.show_figures:
        plot_simulation_summary(
            {name: result.environment for name, result in results.items()}
        )
        plt.show()


if __name__ == "__main__":
    main()
