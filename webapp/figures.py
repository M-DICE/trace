from __future__ import annotations

import plotly.graph_objects as go


def add_power_thresholds(
    fig: go.Figure,
    *,
    labels: tuple[str, str] = ("80% power", "95% power"),
    row: int | None = None,
    col: int | None = None,
) -> None:
    kwargs = {} if row is None else {"row": row, "col": col}
    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="grey",
        annotation_text=labels[0],
        annotation_position="right",
        **kwargs,
    )
    fig.add_hline(
        y=0.95,
        line_dash="dot",
        line_color="grey",
        annotation_text=labels[1],
        annotation_position="right",
        **kwargs,
    )


def add_pre_shading(
    fig: go.Figure,
    npre: int,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    kwargs = {} if row is None else {"row": row, "col": col}
    fig.add_vrect(
        x0=1,
        x1=npre,
        fillcolor="rgba(100,149,237,0.07)",
        line_width=0,
        **kwargs,
    )


def add_timing_band(
    fig: go.Figure,
    *,
    label: str = "±3 mo",
    row: int | None = None,
    col: int | None = None,
) -> None:
    kwargs = {} if row is None else {"row": row, "col": col}
    fig.add_hrect(
        y0=-3,
        y1=3,
        fillcolor="rgba(0,180,0,0.07)",
        line_width=0,
        annotation_text=label,
        annotation_position="top right",
        annotation_font=dict(color="green", size=10),
        **kwargs,
    )
