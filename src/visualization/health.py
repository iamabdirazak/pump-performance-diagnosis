"""Health estimates through each fault run: datasheet only, learned unit, and truth."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features.residuals import compute
from src.models.correction import Correction
from src.models.health import estimate
from src.visualization import style

PANELS = {  # run: (parameter, title, unit label, true value from severity)
    "wear": ("head", "Head Retained", "Fraction of healthy", lambda s: 1 - 0.12 * s),
    "blockage": ("discharge", "Discharge Friction", "× healthy", lambda s: 1 + s),
    "clogging": ("suction", "Suction Friction", "× healthy", lambda s: 1 + 17 * s),
}


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    correction = Correction.load(Path("models/correction.json"))
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    for axis, (run, (parameter, title, unit, truth)) in zip(axes, PANELS.items()):
        frame = pd.read_parquet(f"data/{run}.parquet")
        onset = frame["time"][frame["severity"] > 0].iloc[0]
        day = (frame["time"] - onset).dt.total_seconds() / 86400
        hourly = slice(None, None, 60)
        for corr, shade in ((None, colors["soft"]), (correction, colors["strong"])):
            health = estimate(frame, compute(frame, corr))[parameter]
            axis.plot(day[hourly], health[hourly], color=shade, linewidth=1.6)
        axis.plot(day[hourly], truth(frame["severity"])[hourly], color=colors["ink"], linewidth=1.0,
                  linestyle=(0, (4, 3)))
        axis.axvline(0, color=colors["secondary"], linewidth=0.8, linestyle=(0, (2, 3)))
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.set_xlabel("Days from fault onset")
        axis.set_xlim(-6, day.max())

    style.legend(figure, [(colors["soft"], "Datasheet only"), (colors["strong"], "Learned unit"),
                          (colors["ink"], "True")], colors)
    figure.tight_layout(w_pad=2.5)
    style.save(figure, directory, "health", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
