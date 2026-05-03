#!/usr/bin/env Rscript
# export_crit_val_table.R
#
# Reads SimRewilding/CritValTable.rds and writes it as JSON to
# r_exports/CritValTable.json so that the Python simulation code can
# load critical values at runtime without hard-coding them.
#
# Usage (from the repo root):
#   Rscript scripts/export_crit_val_table.R
#   Rscript scripts/export_crit_val_table.R --input path/to/CritValTable.rds \
#                                            --output path/to/CritValTable.json
#
# Dependencies: jsonlite (install.packages("jsonlite"))

library(jsonlite)

# ── argument parsing ──────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

parse_flag <- function(flag, default) {
  idx <- which(args == flag)
  if (length(idx) == 1 && idx < length(args)) args[idx + 1] else default
}

input_path  <- parse_flag("--input",  file.path("SimRewilding", "CritValTable.rds"))
output_path <- parse_flag("--output", file.path("r_exports",    "CritValTable.json"))

# ── read ──────────────────────────────────────────────────────────────────────
if (!file.exists(input_path)) {
  stop(paste("Input file not found:", input_path))
}

tbl <- readRDS(input_path)

# Normalise types: Gamma and Alpha as numeric, Detector as character
tbl$Detector <- as.character(tbl$Detector)
tbl$Gamma    <- as.numeric(tbl$Gamma)
tbl$Alpha    <- as.numeric(tbl$Alpha)
tbl$CritVal  <- as.numeric(tbl$CritVal)

# ── write ─────────────────────────────────────────────────────────────────────
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)

json_str <- toJSON(tbl, pretty = TRUE, digits = 10, dataframe = "rows")
writeLines(json_str, output_path)

cat(sprintf("Wrote %d rows to %s\n", nrow(tbl), output_path))
