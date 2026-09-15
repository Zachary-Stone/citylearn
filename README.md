# CityLearn: Reinforcement Learning Control for Grid-Interactive Buildings

This project provides reproducible CityLearn control experiments as a Python
codebase. It includes baseline, random, rule-based, tabular Q-learning, and
Soft Actor-Critic workflows, along with KPI and load-profile evaluation.

## Run an experiment set

Install the locked dependencies, edit [experiment.toml](experiment.toml), then
run the launcher:

```bash
poetry install
cd src
poetry run python main.py
```

The `run.experiments` list selects the workflows to execute. The sample
configuration runs the lightweight baseline and rule-based controller. Add
`tabular_q_learning` or `sac` only when their configured training episode
counts are appropriate for your machine.

Each run writes `kpis.csv` and seven comparison figures to the configured
`run.output_directory`. Set `run.show_figures` to `false` for non-interactive
runs.

## Configuration

`experiment.toml` is the sole user-facing execution interface. It contains:

- `run`: selected experiments, figure display, and output directory;
- `simulation`: dataset, seed, selected buildings, duration, and observations;
- `tabular_q_learning`: training episode count; and
- `sac`: training episode count, reward choice, and Stable-Baselines settings.

Use `reward_function = "custom"` for the supplied battery-aware reward or
`"default"` for CityLearn's standard reward.

## Development checks

```bash
poetry run ruff check src tests
poetry run ruff format --check src tests
PYTHONPATH=src poetry run python -m unittest discover -s tests -v
```

## License and citation

Usage is subject to the MIT License. The tutorial content originates from:

Nweye, K., Wu, A., Almilaify, Y., Mohammadi, A., & Nagy, Z. (2026).
*CityLearn: Reinforcement Learning Control for Grid-Interactive Efficient
Buildings and Communities* [Tutorial]. Climate Change AI Summer School.
https://doi.org/10.5281/zenodo.11639022
