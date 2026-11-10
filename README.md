# Location–Scale Calibration for Generalized Posteriors

Simulation code for
"[**Location-Scale Calibration for Generalized Posteriors**](https://arxiv.org/abs/2511.15320)".

## What is this repository?

This repository contains simulation code for our paper, "Location-Scale Calibration for Generalized Posteriors".

The Huber random-intercept study compares sandwich Wald intervals (`wald`), uncalibrated generalized posterior intervals (`raw`), and location-scale calibrated posterior intervals (`cal`).
The experiment uses common data sets across 20 learning rates from 0.01 to 100, with two Gibbs chains per data set and learning rate.
Intervals have a nominal coverage probability of 95%.

## Requirements and setup

Python 3.13 or later, [Poetry](https://python-poetry.org/), and a C++17 compiler available as `c++` are required.

```bash
# Clone the repository.
git clone git@github.com:shutech2001/ls-calib-gp.git
cd ls-calib-gp

# Install the environment and CLI commands.
poetry install

# Check the installed commands.
poetry run lscalibgb-experiment --help
poetry run lscalibgb-report --help
```

Shell activation is optional when using `poetry run`:

```bash
eval "$(poetry env activate)"

# For an editor, use the Python executable path printed by this command.
poetry env info --executable
```

The sampler compiles automatically on its first use.
Its shared library is cached in `.cache/lscalibgb/` and rebuilt when the C++ source changes.

## Run the experiment

The default run uses 200 data sets, 10,000 retained draws per chain, and 1,000 burn-in updates per chain:

```bash
poetry run lscalibgb-experiment --output outputs
```

The equivalent script command is:

```bash
poetry run python src/run_experiment.py --output outputs
```

`--replications`, `--kept`, and `--burn` override the compute budget.
`--output` selects the destination directory and defaults to the repository's `outputs/` directory.
Running an experiment replaces the two files there.

## Outputs

The experiment produces exactly two files:

```text
outputs/
├── summary_results.csv
└── gibbs_four_panel.tex
```

The CSV contains one row per learning rate and method:

| Column | Meaning |
|---|---|
| `eta` | Learning rate |
| `method` | `wald`, `raw`, or `cal` |
| `coverage` | Fraction of intervals containing the penalized population target |
| `coverage_mcse` | Monte Carlo standard error of coverage |
| `mean_width` | Mean interval width |
| `mean_width_mcse` | Monte Carlo standard error of mean interval width |
| `mean_bias` | Mean point-estimation error relative to the population target |
| `sd_bias` | Sample standard deviation of point-estimation errors |
| `mean_bias_mcse` | Monte Carlo standard error of mean bias |

The TeX figure embeds the CSV's numerical coordinates and shows coverage, mean interval width, mean bias, and the standard deviation of estimation errors.
Include it in a LaTeX document using `tikz`, `pgfplots`, and `subcaption`.

To regenerate only the figure from an existing summary:

```bash
poetry run lscalibgb-report --output outputs
```

## Repository layout

```text
src/
├── model.py                # Data generation, population target, piecewise loss
├── sampler.py              # Compilation and Python interface to the sampler
├── energy_gibbs.cpp        # Direct full-conditional Gibbs updates
├── run_experiment.py       # Experiment runner and CSV aggregation
├── export_tex.py           # Four-panel TeX figure generation
├── report_results.py       # Figure regeneration CLI
└── figure_template.tex     # Editable PGFPlots template
```

## Citation

```bibtex
@article{tamano2025location,
    author={Tamano, Shu and Tomo, Yui},
    journal={arXiv preprint arXiv:2511.15320},
    title={Location–Scale Calibration for Generalized Posteriors},
    year={2025},
}
```

## Contact and license

If you have any question, please feel free to contact: tamano.s@jihs.go.jp