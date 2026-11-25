"""How each fault moves the pump at full speed, against the healthy pump."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.plant import faults
from src.plant.pump import Pump, System, operating_point
from src.visualization import style

SPEED = 1750


def draw(theme: str, directory: Path) -> None:
    colors = style.apply(theme)
    flow = np.linspace(0, 75, 300)
    healthy_pump, healthy_system = Pump(), System()
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    titles = {"wear": "Impeller Wear", "blockage": "Discharge Blockage", "clogging": "Suction Clogging"}

    for axis, fault in zip(axes, faults.FAULTS):
        pump, system = faults.apply(healthy_pump, healthy_system, fault, 1.0)
        before = operating_point(healthy_pump, healthy_system, SPEED)
        after = operating_point(pump, system, SPEED)

        fixed = colors["secondary"]
        if fault == "clogging":
            axis.plot(flow, healthy_system.npsh_available(flow), color=colors["soft"], linewidth=1.8)
            axis.plot(flow, system.npsh_available(flow), color=colors["strong"], linewidth=1.8)
            axis.plot(flow, pump.npsh_required(flow, SPEED), color=fixed, linewidth=1.1, linestyle=(0, (4, 3)))
            axis.annotate("Required", (40, pump.npsh_required(40, SPEED)), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=8, color=fixed)
            axis.annotate("Available", (8, healthy_system.npsh_available(8)), xytext=(0, 8),
                          textcoords="offset points", ha="center", fontsize=8, color=fixed)
            points = [(before.flow, before.npsh_available), (after.flow, after.npsh_available)]
            axis.set_ylim(0, 14)
            axis.set_ylabel("NPSH, m")
        else:
            changed_pump = fault == "wear"
            axis.plot(flow, healthy_pump.head(flow, SPEED), linewidth=1.8,
                      color=colors["soft"] if changed_pump else fixed)
            axis.plot(flow, healthy_system.head(flow), linewidth=1.1, linestyle=(0, (4, 3)),
                      color=fixed if changed_pump else colors["soft"])
            if changed_pump:
                axis.plot(flow, pump.head(flow, SPEED), color=colors["strong"], linewidth=1.8)
            else:
                axis.plot(flow, system.head(flow), color=colors["strong"], linewidth=1.1, linestyle=(0, (4, 3)))
            axis.annotate("Pump", (4, healthy_pump.head(4, SPEED)), xytext=(0, 6), textcoords="offset points",
                          fontsize=8, color=fixed)
            axis.annotate("System", (4, healthy_system.head(4)), xytext=(0, -12), textcoords="offset points",
                          fontsize=8, color=fixed)
            points = [(before.flow, before.head), (after.flow, after.head)]
            axis.set_ylim(0, 25)
            if fault == "wear":
                axis.set_ylabel("Head, m")

        for (x, y), shade in zip(points, (colors["soft"], colors["strong"])):
            axis.plot(x, y, "o", markersize=5.5, color=shade, markeredgecolor=colors["background"],
                      markeredgewidth=1.2, zorder=4)
        axis.annotate("", xy=points[1], xytext=points[0],
                      arrowprops=dict(arrowstyle="-|>", color=colors["ink"], lw=0.9, shrinkA=6, shrinkB=6))
        axis.set_title(titles[fault])
        axis.set_xlim(0, 75)
        axis.set_xlabel("Flow, m³/h")

    style.legend(figure, [(colors["soft"], "Healthy"), (colors["strong"], "Severity 1"), (colors["secondary"], "Unchanged")], colors)
    figure.tight_layout(w_pad=2.5)
    style.save(figure, directory, "faults", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        draw(theme, Path("reports/figures"))
