import plotly.express as px

NPRE: int = 24
NPOST_YEARS: int = 10
NPOST_MONTHS: int = 12 * NPOST_YEARS  # 120
EFFECT_SIZES_PCT: list[int] = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PALETTE: list[str] = px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])
