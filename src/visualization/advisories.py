"""When each advisory is raised, on the learned health estimate."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.features.residuals import compute
from src.models.correction import Correction
from src.models.health import estimate
from src.monitoring.advisory import RULES, advise
from src.visualization import style

PANELS = {"wear": ("Impeller Wear", "Head retained"), "blockage": ("Discharge Blockage", "Discharge friction, ×"),
          "clogging": ("Suction Clogging", "Suction friction, ×")}


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    correction = Correction.load(Path("models/correction.json"))
    results = json.loads(Path("reports/results.json").read_text())["learned"]
    rules = {rule.cause: rule for rule in RULES}
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    for axis, (run, (title, label)) in zip(axes, PANELS.items()):
        frame = pd.read_parquet(f"data/{run}.parquet")
        health = estimate(frame, compute(frame, correction))
        advice = advise(health)
        onset = frame["time"][frame["severity"] > 0].iloc[0]
        day = (frame["time"] - onset).dt.total_seconds() / 86400
        rule = rules[run]

        start = day[advice.index[advice[run]][0]]
        axis.axvspan(start, day.max(), color=colors["faint"], linewidth=0, zorder=0)
        axis.plot(day[59::60], health[rule.parameter].iloc[59::60], color=colors["strong"], linewidth=1.6)
        axis.axhline(rule.threshold, color=colors["ink"], linewidth=1.0, linestyle=(0, (4, 3)))
        axis.annotate(f"Advisory\n{results[run]['lead_days']:.0f} days before due", (start, 0.94),
                      xycoords=("data", "axes fraction"), xytext=(8, 0), textcoords="offset points",
                      va="top", fontsize=8, color=colors["ink"])
        if rule.rising:  # leave headroom for the label above the rising line
            axis.set_ylim(top=float(health[rule.parameter].max()) * 1.12)
        axis.set_title(title)
        axis.set_ylabel(label)
        axis.set_xlabel("Days from fault onset")
        axis.set_xlim(-6, day.max())

    style.legend(figure, [(colors["strong"], "Health, learned unit"), (colors["ink"], "Threshold"),
                          (colors["faint"], "Advisory raised")], colors)
    figure.tight_layout(w_pad=2.5)
    style.save(figure, directory, "advisories", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
