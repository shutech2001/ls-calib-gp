# Location–Scale Calibration for Generalized Posterior
Materials for "[**Location-Scale Calibration for Generalized Posterior**](https://arxiv.org/abs/2511.15320)"

## What is This Repository?

This repository includes an implementation of our paper, "Location-Scale Calibration for Generalized Posterior".
It also contains some experiments presented in the paper.

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/ls-calib-gp.git

# build the environment with poetry
poetry install

# active virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter
poetry env info --path
```

### Executing Simulation
- `python src/ex_huber_random_intercept.py`
  - `--n_sim` | `--N`
    - Number of simulations (default: `200`).
  - `--G`
    - Number of groups (default: `100`).
  - `--n_per_group` | `--NG`
    - Number of observations per group (default: `5`).
  - `--p`
    - Dimension of covariates (default: `1`),
  - `--beta_true`
    - True coefficients (default: `2.0`).
  - `--G_large`
    - Number of groups for pseudo-true beta (default: `5000`).
  - `--large_replication`
    - Number of replications for pseudo-true beta (default: `1000`).
  - `--tau`
    - True random intercept variance (default: `1.0`).
  - `--sigma`
    - True residual variance (default: `1.0`).
  - `--c`
    - Huber tuning parameter (default: `1.0`).
  - `--lam`
    - Ridge penalty parameter (default: `0.1`).
  - `--outlier_prob`
    - Outlier probability (default: `0.1`).
  - `--outlier_scale`
    - Outlier scale (default: `10.0`).
  - `--n_iter`
    - Number of Gibbs iterations (default: `1000`).
  - `--n_burn_in`
    - Number of burn-in iterations (default: `500`).
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
- src/huber_random_intercept/huber_random_intercept.py
  - class for random intercept linear mixed model with fixed Gaussian prior
    - this class includes frequentist method (`fit_frequentist_penalized_ee`)

## Citation
```
@article{tamano2025location,
    author={Tamano, Shu and Tomo, Yui},
    journal={arXiv preprint arXiv:2511.15320},
    title={Location–Scale Calibration for Generalized Posteriors},
    year={2025},
}
```

## Contact

If you have any question, please feel free to contact: stamano@niid.go.jp