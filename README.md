# TRACE: Temporal Rewilding Analysis for Changepoint Estimation

## Overview

We simulate time series with **trend change** and **distribution change** and evaluate how well different changepoint detection methods can identify the moment an ecological intervention starts to have a measurable effect.

Detection methods covered:

| Method | Type | Noise | Use case |
|--------|------|-------|----------|
| **AMOC** (At Most One Change) | Offline batch | i.i.d. & AR(1) | End-of-monitoring analysis |
| **BOCPD** (Bayesian Online Changepoint Detection) | Online | i.i.d. | Real-time monitoring |
| **Forecast-based** (`changepoint.forecast`) | Online | AR(1) | Autocorrelated real-time monitoring |

Change types:

- **Trend change**: a shift in the slope (monthly rate of change) of a univariate time series
- **Distribution change**: a shift in the mean/variance of the population, summarised as Wasserstein distance or AUC difference between control and intervention distributions

Study designs:

- **BACI** (Before-After Control-Intervention): paired control and intervention series; the test statistic is computed on the difference, removing shared environmental variation
- **BA** (Before-After): intervention series only

### Reference implementation

The reference implementation is tracked as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules): [SimRewilding](https://github.com/GMY2018/SimRewilding).

---

## Streamlit web app

The interactive app lets you explore pre-computed simulation results and run small custom simulations directly in the browser.

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

### Run the app

```bash
uv run streamlit run Home.py
```

Open **http://localhost:8501** in your browser.

### Generate pre-computed results (first time only)

The app loads results from `results/`. Run the simulations before launching the app (these may take some time... grab a cup of ☕!):

```bash
uv run python rewild_trend_change_amoc.py
uv run python rewild_distribution_change_amoc.py
```

## Python simulations (command-line)

### Trend change AMOC

```bash
# Full run
uv run python rewild_trend_change_amoc.py

# Regenerate plots from saved results (no simulation)
uv run python rewild_trend_change_amoc.py --plots-only

# Fast smoke-test (Nsim=10, simN=10)
uv run python rewild_trend_change_amoc.py --quick
```

### Distribution change AMOC

```bash
# Full run
uv run python rewild_distribution_change_amoc.py

# Regenerate plots from saved results (no simulation)
uv run python rewild_distribution_change_amoc.py --plots-only

# Fast smoke-test (Nsim=10, simN=10)
uv run python rewild_distribution_change_amoc.py --quick
```
