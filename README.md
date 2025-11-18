# LSCalibGB
Materials for "Location-Scale Calibration for Generalized Posterior"

## What is This Repository?

This repository includes an implementation of our paper, "Location-Scale Calibration for Generalized Posterior".
It also contains some experiments presented in the paper.

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/LSCalibGB.git

# build the environment with poetry
poetry install

# active virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

### Executing Simulation
- `python src/ex_random_intercept.py`
  - `--n_sim` | `--N`
    - Number of simulations (default: `500`).
  - `--J`
    - Number of groups (default: `60`).
  - `--n_per_group` | `--NG`
    - Number of observations per group (default: `5`).
  - `--p`
    - Dimension of covariates (default: `1`),
  - `--M`
    - Number of samples from the posterior (default: `1000`).
  - `--alpha`
    - Significance level (default: `0.05`).
  - `--seed`
    - Random seed (default: `42`).
  - `--verbose`
    - Whether to print verbose output (default: `False`).
  - `--one_scenario` | `--OS`
    - Whether to run a single scenario (default: `False`).
    - If this is `True`,
      - `--eta` | `--LR`
        - Learning rate for running a single scenario (default: `1.0`).
    - If this is `False`,
      - `--min_learning_rate` | `--minLR`
        - Minimum learning rate (default: `0.01`).
      - `--max_learning_rate` | `--maxLR`
        - Maximum learning rate (default: `100.0`).
      - `--n_learning_rate` | `--NLR`
        - Number of learning rates (default: `20`).

### File Description
- src/random_intercept_lmm
  - class for random intercept linear mixed model with fixed Gaussian prior
    - this class includes frequentist method (`fit_frequentist_lmm`)

## Contact

If you have any question, please feel free to contact: stamano@niid.go.jp