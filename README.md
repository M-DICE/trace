# TRACE: Temporal Restoration Analysis for Changepoint Estimation

TRACE is a tool for ecologists that helps answer the question: **how quickly does a restoration intervention produce a measurable effect?** It simulates ecological time series, applies statistical detection methods, and reports how reliably and how fast each method can detect the moment change begins.

The interactive web app lets you explore pre-computed results or run small custom simulations directly in your browser.

---

## Quick Start

> These steps get the web app running on your computer in about 10 minutes.

1. **Install R** — download from [cran.r-project.org](https://cran.r-project.org/) and follow the installer for your system.

2. **Install uv** (a Python package manager). Open a terminal and run:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies** (run once, from the project folder):
   ```bash
   uv sync
   ```

4. **Generate pre-computed results** (optional but recommended — takes a few minutes):
   ```bash
   uv run trace-sim all --quick
   ```

5. **Start the app:**
   ```bash
   uv run streamlit run Home.py
   ```
   Then open **http://localhost:8501** in your browser.

> If you skip step 4, the app still runs but result pages will appear empty until simulations have been generated.

---

## Overview

TRACE simulates time series with **trend change** and **distribution change** and evaluates how well different changepoint detection methods can identify the moment an ecological intervention starts to have a measurable effect.

Detection methods covered:

| Method | Type | Noise | Use case |
|--------|------|-------|----------|
| **AMOC** (At Most One Change) | Offline batch | i.i.d. & AR(1) | End-of-monitoring analysis |
| **BOCPD** (Bayesian Online Changepoint Detection) | Online | i.i.d. | Real-time monitoring |
| **Forecast** (Forecast error-based method) | Online | i.i.d. & AR(1) | Real-time monitoring |

Change types:

- **Trend change**: a shift in the slope (monthly rate of change) of a univariate time series
- **Distribution change**: a shift in the mean/variance of the population, summarised as Wasserstein distance or AUC difference between control and intervention distributions

Study designs:

- **BACI** (Before-After Control-Intervention): paired control and intervention series; the test statistic is computed on the difference, removing shared environmental variation
- **BA** (Before-After): intervention series only

### Reference implementation

The reference implementation is tracked as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules): [SimRewilding](https://github.com/GMY2018/SimRewilding).

---

## Developers guide

### Python simulations (command-line)

```
uv run trace-sim <simulation> [--quick] [--plots-only] [--no-cache]
```

`<simulation>` is one of: `all`, `trend-amoc`, `trend-forecast`, `trend-bocpd`, `distribution-amoc`, `distribution-forecast`, `distribution-bocpd`

Examples:
```bash
# Show help
uv run trace-sim -h

# Run all simulations (full)
uv run trace-sim all

# Run all simulations (fast smoke-test, Nsim=10, simN=10)
uv run trace-sim all --quick

# Run a single simulation
uv run trace-sim trend-amoc

# Regenerate plots from saved results (no simulation)
uv run trace-sim trend-amoc --plots-only

# Ignore saved results and re-run all stages from scratch
uv run trace-sim trend-amoc --no-cache
```

### Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
uv run ruff check .          # check
uv run ruff check --fix .    # fix auto-fixable issues
uv run ruff format .         # format
```

### R data export scripts

Some Python modules load pre-computed data that originates in the R codebase.
These scripts convert R binary files to JSON so that Python can read them at
runtime without an R dependency.

#### Critical value table (`CritValTable.json`)

The weighted Page-CUSUM detector reads its critical values from
`data/CritValTable.json`. This file is derived from the
`CritValTable.rds` lookup table shipped with the SimRewilding R codebase and
pre-simulated by the `changepoint.forecast` package authors.

**When to run**: after the initial clone and after any `git submodule update`
that changes `SimRewilding/CritValTable.rds`.

**Requires**: R with the `jsonlite` package (`install.packages("jsonlite")`).

```bash
Rscript scripts/export_crit_val_table.R
```

Custom paths (optional):

```bash
Rscript scripts/export_crit_val_table.R \
  --input  SimRewilding/CritValTable.rds \
  --output data/CritValTable.json
```

The output is a JSON array of records with fields `Detector`, `Gamma`, `Alpha`,
and `CritVal`, covering 4 detectors × 19 gamma values × 3 alpha levels (228
rows total).
