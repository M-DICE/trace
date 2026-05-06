"""Timing utilities shared across simulation and changepoint modules."""


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as 'X.Xs' or 'Xm YYs'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"
