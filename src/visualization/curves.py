"""Pump curves at four speeds against the system curve."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plant.pump import Pump, System, operating_point
from src.visualization import style

SPEEDS = (1400, 1500, 1600, 1750)  # rpm, slowest first


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    pump, system = Pump(), System()
    shades = [colors["faint"], colors["soft"], colors["base"], colors["strong"]]
    shades[0] = "#A9CBEE" if theme == "light" else "#2F5A8A"

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    flow = np.linspace(0, 80, 400)

    for speed, shade in zip(SPEEDS, shades):
        head = pump.head(flow, speed)
        keep = head >= 0
        axis.plot(flow[keep], head[keep], color=shade, linewidth=1.8, solid_capstyle="round")
        axis.annotate(f"{speed} rpm", (flow[keep][-1], head[keep][-1]), xytext=(4, 0),
                      textcoords="offset points", va="center", fontsize=8, color=colors["secondary"])
        point = operating_point(pump, system, speed)
        axis.plot(point.flow, point.head, "o", markersize=5.5, color=shade,
                  markeredgecolor=colors["background"], markeredgewidth=1.2, zorder=4)

    axis.plot(flow, system.head(flow), color=colors["ink"], linewidth=1.1, linestyle=(0, (4, 3)))
    axis.annotate("System", (flow[-1], system.head(flow[-1])), xytext=(4, 0), textcoords="offset points",
                  va="center", fontsize=8, color=colors["ink"])

    bep = operating_point(pump, system, 1750)
    axis.annotate(f"Duty point\n{bep.flow:.0f} m³/h, {bep.head:.1f} m", (bep.flow, bep.head),
                  xytext=(-14, 14), textcoords="offset points", ha="right", fontsize=8, color=colors["ink"])

    axis.set_xlim(0, 80)
    axis.set_ylim(0, 25)
    axis.set_xlabel("Flow, m³/h")
    axis.set_ylabel("Head, m")
    axis.set_title("Pump and System Curves")
    style.save(figure, directory, "curves", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
