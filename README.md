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

- **Trend change** — a shift in the slope (monthly rate of change) of a univariate time series
- **Distribution change** — a shift in the mean/variance of the population, summarised as Wasserstein distance or AUC difference between control and intervention distributions

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

The app loads results from `results/*`. If files are missing, run the full simulation first (may take about 30/60 minutes):

```bash
uv run python rewild_trend_change_amoc.py
```

## Python simulation (command-line)

Run the full trend-change AMOC simulation:

```bash
uv run python rewild_trend_change_amoc.py
```

To regenerate plots from previously saved results:

```bash
uv run python rewild_trend_change_amoc.py --plots-only # It exits with an error if no saved results are found.
```
