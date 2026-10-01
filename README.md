# Midge wing lengths: prior impact analysis

This project studies the effective prior sample size of an informative Normal-Inverse-Gamma prior for midge wing lengths example. One command-line runner composes a shared model, sampling strategy, target loss, and risk-matching rule. Each configuration produces a table and a plot comparing prior locations for the marginal mean and joint mean/log-variance targets.

Source: Hoff (2009), *A First Course in Bayesian Statistical Methods*, printed pp. 73 and 76, attributes the data to Grogan and Wirth (1981). The nine observations are wing lengths in mm. They are public and included in `data/raw/wing_lengths.csv`.

Both methods evaluate 82 prior locations, with the original NIG prior `mu | sigma2 ~ N(mu0, sigma2)`, `sigma2 ~ IG(0.5, 0.005)` and independent Jeffreys baseline `p(mu,sigma2) ∝ 1/sigma2`. They report two posterior targets: scalar `mu/0.1` and joint `(mu/0.1, log(sigma2/0.01))`. By default, Reimherr matches mean posterior MSE under conditional proper-baseline bootstrap on `m=-3,...,40`; Wiesenfarth matches plug-in likelihood squared posterior-mean error on `m=-40,...,5`. Both use 1000 pseudo-datasets per risk estimate. These are the project's explicit NIG implementations of the risk-matching ideas; implementation choices are documented below. A boundary result is a limit of the finite search grid.

## Project structure

```text
original/                   # Preserved initial version
data/raw/wing_lengths.csv   # Original observations
scripts/
  runner.py                 # Single entry point; method selected with --method
src/
  model.py                  # NIG posterior update and reference parameters
  sampling.py               # Conditional bootstrap and plug-in likelihood
  targets.py                # mu and joint_log coordinates
  losses.py                 # Posterior MSE and squared mean error
  matching.py               # R/W sample-size rules and minimum-gap selection
  config.py                 # Validated settings, defaults, and prior grid
  analysis.py               # Compose components and run prior locations
  data.py                   # Input loading and validation
  reporting.py              # CSV and configuration records
  plotting.py               # Plot validated CSV files
  validation.py             # Output completeness and consistency checks
  cli.py                    # Shared command-line options
tests/                      # Data, model, framework, and import checks
pytest.ini                  # Adds src/ and scripts/ to the test import path
results/
  tables/                   # CSV results
  figures/                  # PNG plots
  metadata/                 # JSON settings and table hashes
  validation.json           # Successful full-workflow validation and file hashes
Makefile                    # Reproduce, plot, validate, and test targets
requirements.txt
README.md
.gitignore
```

The runner locates `src/` relative to its own file, so no editable
installation or manual `PYTHONPATH` is needed. It can run from another working
directory and creates missing results directories automatically. Core modules
perform no analysis on import. `original/` remains the comparison archive.

## Reproduce the complete project

After completing [Setup](#setup), run from this repository directory:

```bash
make reproduce
```

It runs the eight method/sampling/loss configurations in a fixed serial order,
with both targets, all 82 prior locations, and 1000 risk repetitions. Each
configuration is computed and saved, its CSV is validated, and the figure is
then generated **by reading that CSV**. Final validation checks all eight
configurations and writes `results/validation.json` only on success.

Other workflow commands are:

```bash
make test       # Run pytest
make plot       # Rebuild all figures from existing CSVs; no analysis computation
make validate   # Validate saved tables, metadata, and figures; no recomputation
make clean      # Delete generated CSVs, PNGs, metadata, and validation report
```

If Make is not installed, the equivalent full workflow is:

```bash
python scripts/runner.py --all
```

## Setup

Python 3.11 or newer is recommended. Start in your clone's root directory
(`STATS-607-Project` by default):

```bash
cd STATS-607-Project
python3 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run an individual analysis

After activating the environment, from the repository directory:

```bash
python scripts/runner.py --method reimherr
python scripts/runner.py --method wiesenfarth
```

Choose sampling and loss independently, for example:

```bash
# W matching with bootstrap and posterior MSE:
python scripts/runner.py --method wiesenfarth --sampling bootstrap --loss posterior_mse
# R matching with likelihood sampling and squared posterior-mean error:
python scripts/runner.py --method reimherr --sampling likelihood --loss squared_mean
# Restrict the output to the marginal mean (still the full prior grid):
python scripts/runner.py --method wiesenfarth --sampling bootstrap --loss squared_mean --target mu
```

-----
## Method framework

The two matching rules build on [Reimherr, Meng, and Nicolae (2021)](https://doi.org/10.1111/rssb.12414)
and [Wiesenfarth and Calderazzo (2020)](https://doi.org/10.1111/biom.13124).
Reimherr and colleagues quantify prior impact through risk matching, including
negative prior sample sizes under discordance. Wiesenfarth and Calderazzo
formulate effective current sample size in terms of observations from the
current data model. The exact samplers, losses, target scales, and finite grids
used here are specified below; the eight configurations are project comparisons,
not eight original methods claimed by the papers.

Separating the model, sampling, posterior update, target, loss, and matching rule
makes these choices visible and independently testable. It also prepares the
code for methods that compare different probability distributions:
[Jones, Trangucci, and Chen (2022)](https://doi.org/10.1214/21-BA1271) condition
on the observed data and compare posteriors along future-data paths using
Wasserstein distance; [Clarke (1996)](https://doi.org/10.1080/01621459.1996.10476674)
compares an informative prior with a reference-prior posterior through relative
entropy. 

The current processing order is:

```text
observations + configuration
  -> model: baseline reference and likelihood plug-in parameters
  -> matching rule: fixed prior/size and candidate prior/sizes
  -> sampling strategy: pseudo-dataset bank shared by sample size
  -> model.update: posterior under the chosen prior for each pseudo-dataset
  -> target + loss: per-dataset errors
  -> mean aggregation: fixed risk and candidate risk curve
  -> selection: minimum absolute risk gap
  -> CSV + figure + JSON configuration
```

`NIGModel`, `BootstrapSampler`, `LikelihoodSampler`, `RiskLoss`, and
`MatchingRule` are small classes with explicit responsibilities. `AnalysisConfig`
stores the choices; `analyze_location` composes them for one prior location.
This keeps method-specific decisions separate from the shared numerical model.

### Model and posterior update

The model is a Normal likelihood with either the fixed-strength informative NIG
prior above or the independence-Jeffreys baseline. Each pseudo-dataset is a
**complete replacement dataset** for a posterior fit; the observed nine values
are not appended to it. Updating the posterior and changing the candidate sample
size are separate operations. Priors are distinguished by `mu0`: a scalar for
the informative prior, `None` for the baseline.

The target reference is always the baseline posterior mean in target
coordinates, calculated from the observed data. In particular,
`E[log(sigma2/0.01)|y]` is not `log(E[sigma2|y]/0.01)`.

### Matching direction

| Method | Fixed risk | Candidate risk | Candidate m |
| --- | --- | --- | --- |
| Reimherr | Informative posterior, size n | Baseline posterior, size n+m | -3 through 40 |
| Wiesenfarth | Baseline posterior, size n | Informative posterior, size n-m | -40 through 5 |

Both choose the candidate minimizing the absolute difference from the fixed
risk. The first candidate in the ascending m grid wins an exact tie. A boundary
flag records selection of either finite-grid endpoint. Changing the sampler or
loss does not change this matching direction.

### Sampling strategy

- `bootstrap`: sample with replacement from the observed wing lengths.
  At **every** candidate size, redraw constant rows until their baseline
  posterior is proper. Record all rejected rows; never drop a candidate.
  This conditional law also applies when W uses bootstrap and when the
  current fit is informative. It is not an unconditional bootstrap estimator.
- `likelihood`: sample from a Normal distribution with fixed parameters
  `E_baseline[mu|y]` and `E_baseline[sigma2|y]`. The variance is
  `sum((y-y.mean())**2)/(n-3)`, not the MLE variance and not posterior predictive
  sampling.

For each prior location, size-specific draws follow candidate-grid order with
seed `20260312 + original_grid_index`. The two targets and both sides reuse the
same data at the same size, including size n. Different sizes are drawn
separately. Switching loss or selecting only one target preserves the draws.

### Target loss and aggregation

For target coordinates d, posterior means a_d, posterior variances v_d, and
fixed reference r_d:

| CLI loss | Per-dataset loss |
| --- | --- |
| `squared_mean` | sum_d (a_d - r_d)^2 |
| `posterior_mse` | sum_d [(a_d - r_d)^2 + v_d] |

`mu` uses only `mu/0.1`; `joint_log` also includes `log(sigma2/0.01)`.
The joint loss is a sum of coordinate losses (trace-MSE), so it requires no
cross-covariance term. Risk is the mean of 1000 per-dataset losses. Both losses
retain the project's shape-greater-than-one admissibility requirement.

All **2 matching methods x 2 samplers x 2 losses** are supported. Defaults are
R + bootstrap + posterior_mse and W + likelihood + squared_mean. Other choices
are explicitly labelled comparison variants, not relabelled as the default
published-method implementations.

## References

- Reimherr, M., Meng, X.-L., and Nicolae, D. L. (2021).
  [Prior sample size extensions for assessing prior impact and prior-likelihood discordance](https://doi.org/10.1111/rssb.12414).
  *Journal of the Royal Statistical Society: Series B*, 83(3), 413–437.
- Wiesenfarth, M., and Calderazzo, S. (2020).
  [Quantification of prior impact in terms of effective current sample size](https://doi.org/10.1111/biom.13124).
  *Biometrics*, 76(1), 326–336. First published online in 2019.
- Jones, D. E., Trangucci, R. N., and Chen, Y. (2022).
  [Quantifying Observed Prior Impact](https://doi.org/10.1214/21-BA1271).
  *Bayesian Analysis*, 17(3), 737–764. First published online in 2021.
- Clarke, B. (1996).
  [Implications of Reference Priors for Prior Information and for Sample Size](https://doi.org/10.1080/01621459.1996.10476674).
  *Journal of the American Statistical Association*, 91(433), 173–184.
- Hoff, P. D. (2009). *A First Course in Bayesian Statistical Methods*,
  pp. 73 and 76 (the data source, citing Grogan and Wirth, 1981).
