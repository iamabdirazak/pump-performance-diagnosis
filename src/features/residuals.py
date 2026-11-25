"""Measured minus expected: four residuals, each watching one part of the station.

The healthy model needs only what the sensors give it. Measured flow and
speed fix where the pump should sit on its own curves, and tank level and
flow fix what the pipes should demand. So each residual is blind to normal
operation and moves only when its part of the station changes.

    head       pump head against the pump curve           (impeller)
    power      electrical power against the healthy pump  (impeller, drive)
    suction    inlet pressure against the suction line    (strainer)
    discharge  outlet pressure against the discharge line (pipe, valves)
"""

import numpy as np
import pandas as pd

from src.plant.pump import DENSITY, GRAVITY, Pump, System
from src.plant.sensors import PIPE_AREA

NAMES = ("head", "power", "suction", "discharge")
UNITS = {"head": "m", "power": "kW", "suction": "m", "discharge": "m"}


def compute(frame: pd.DataFrame, correction=None, pump: Pump = Pump(), system: System = System()) -> pd.DataFrame:
    """Residuals against the datasheet, or against the learned unit if a correction is given.

    Also returns the expected head and power, which the health estimate needs.
    """
    flow, speed = frame["flow"].to_numpy(), frame["speed"].to_numpy()
    rho_g = DENSITY * GRAVITY
    velocity_head = (flow / 3600 / PIPE_AREA) ** 2 / (2 * GRAVITY)

    head_measured = (frame["discharge_pressure"] - frame["suction_pressure"]).to_numpy() * 1000 / rho_g
    head_expected = pump.head(flow, speed)
    power_expected = rho_g * flow / 3600 * head_expected / (pump.efficiency(flow, speed) * pump.drive_efficiency) / 1000

    suction_measured = frame["suction_pressure"].to_numpy() * 1000 / rho_g
    suction_expected = frame["level"].to_numpy() - system.friction_suction * flow**2 - velocity_head
    discharge_measured = frame["discharge_pressure"].to_numpy() * 1000 / rho_g
    discharge_expected = system.level_reservoir + system.friction_discharge * flow**2 - velocity_head

    expected = pd.DataFrame({
        "head": head_expected, "power": power_expected,
        "suction": suction_expected, "discharge": discharge_expected,
    }, index=frame.index)
    if correction is not None:
        expected += correction.predict(flow, speed).to_numpy()
    measured = {"head": head_measured, "power": frame["power"].to_numpy(),
                "suction": suction_measured, "discharge": discharge_measured}
    residuals = pd.DataFrame({name: measured[name] - expected[name] for name in NAMES}, index=frame.index)
    residuals["head_expected"] = expected["head"]
    residuals["power_expected"] = expected["power"]
    residuals["flow"] = flow
    return residuals
