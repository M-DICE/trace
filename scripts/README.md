# R data export scripts

Some Python modules load pre-computed data that originates in the R codebase.
These scripts convert R binary files to JSON so that Python can read them at
runtime without an R dependency.

## Critical value table (`CritValTable.json`)

The weighted Page-CUSUM detector reads its critical values from
`data/CritValTable.json`. This file is derived from the
`CritValTable.rds` lookup table shipped with the SimRewilding R codebase and
pre-simulated by the `changepoint.forecast` package authors.

**When to run**: after the initial clone and after any `git submodule update`
that changes `SimRewilding/CritValTable.rds`.

**Requires**: R (the script installs `jsonlite` automatically if needed).

```bash
uv run Rscript scripts/export_crit_val_table.R
```

Custom paths (optional):

```bash
uv run Rscript scripts/export_crit_val_table.R \
  --input  SimRewilding/CritValTable.rds \
  --output data/CritValTable.json
```

The output is a JSON array of records with fields `Detector`, `Gamma`, `Alpha`,
and `CritVal`, covering 4 detectors × 19 gamma values × 3 alpha levels (228
rows total).
