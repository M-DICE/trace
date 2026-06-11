"""Shared formatting helpers for the simulation CLIs.

All CLI output goes through these helpers so the six pipelines stay visually
consistent.  Stages are named rather than numbered: adding, removing, or
reordering a stage needs no renumbering.
"""

RULE = "=" * 70


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def print_run_header(title: str, params: list[tuple[str, object]]) -> None:
    """Print the run title and an aligned block of ``label: value`` config lines."""
    print(title)
    print(RULE)
    width = max((len(label) for label, _ in params), default=0)
    for label, value in params:
        print(f"{label + ':':<{width + 1}}  {value}")
    print()


def print_stage(title: str, cached: int | None = None, total: int | None = None) -> None:
    """Print a named stage banner (replaces the old numbered ``PHASE N`` banners).

    When ``cached`` and ``total`` are given, the count of already-cached results
    is appended so a resumed run shows how much work it is skipping.
    """
    if cached is not None and total is not None:
        title = f"{title}  ({cached}/{total} cached)"
    print(RULE)
    print(title)
    print(RULE)


def print_summary_header(title: str, width: int, sublines: tuple[str, ...] = ()) -> None:
    """Print a results-summary banner sized to the table that follows it."""
    bar = "=" * width
    print(bar)
    print(title)
    for line in sublines:
        print(line)
    print(bar)


def print_complete(plots_dir: str) -> None:
    """Print the closing banner after all plots have been written."""
    print(RULE)
    print(f"All plots saved to: {plots_dir}/")
    print("Analysis complete")
    print(RULE)
