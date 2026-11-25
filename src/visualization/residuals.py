"""Each residual (against the learned unit) through the three fault runs, as hourly means."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features.residuals import NAMES, UNITS, compute
from src.models.correction import Correction
from src.visualization import style

TITLES = {"head": "Head", "power": "Power", "suction": "Suction", "discharge": "Discharge"}
RUNS = ("wear", "blockage", "clogging")
LABELS = {"wear": "Impeller wear", "blockage": "Discharge blockage", "clogging": "Suction clogging"}


def hourly(name: str, correction: Correction) -> pd.DataFrame:
    frame = pd.read_parquet(f"data/{name}.parquet")
    residuals = compute(frame, correction)[list(NAMES)]
    onset = frame["time"][frame["severity"] > 0].iloc[0]
    residuals["day"] = (frame["time"] - onset).dt.total_seconds() / 86400
    return residuals.groupby(residuals["day"].floordiv(1 / 24)).mean()


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    correction = Correction.load(Path("models/correction.json"))
    spread = compute(pd.read_parquet("data/healthy.parquet"), correction)[list(NAMES)].std()
    shades = {"wear": colors["strong"], "blockage": colors["base"], "clogging": colors["soft"]}
    runs = {name: hourly(name, correction) for name in RUNS}

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.3), sharex=True)
    for axis, residual in zip(axes, NAMES):
        band = 3 * spread[residual]
        axis.axhspan(-band, band, color=colors["hairline"], alpha=0.5, linewidth=0, zorder=0)
        for name, frame in runs.items():
            axis.plot(frame["day"], frame[residual], color=shades[name], linewidth=1.4)
        axis.axvline(0, color=colors["secondary"], linewidth=0.8, linestyle=(0, (2, 3)))
        axis.set_title(TITLES[residual])
        axis.set_ylabel(UNITS[residual])
        axis.set_xlabel("Days from fault onset")
        axis.set_xlim(-7, 42)

    style.legend(figure, [(shades[name], LABELS[name]) for name in RUNS] + [(colors["hairline"], "Healthy ±3σ")], colors)
    figure.tight_layout(w_pad=2.0)
    style.save(figure, directory, "residuals", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
