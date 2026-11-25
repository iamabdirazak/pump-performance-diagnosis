"""Health in engineering units, solved from the residuals by least squares.

Each fault changes one parameter, and each residual is linear in that
change. Fitting over a six-hour window weights each minute by how much it
says: a pipe-friction change shows most at high flow, so high-flow minutes
count most. The window is one full day, so the daily cycle of speed and
reservoir level averages out and the result no longer depends on how hard
the pump is running. The cavitation margin is a present danger rather than
a slow trend, so it is averaged over the last hour only.

    head        head retained, fraction of healthy        (1.0 = healthy)
    power       power drawn, fraction of healthy          (1.0 = healthy)
    discharge   discharge-line friction, × healthy        (1.0 = healthy)
    suction     suction-line friction, × healthy          (1.0 = healthy)
    margin      cavitation margin, NPSH available − required, m
"""

import numpy as np
import pandas as pd

from src.plant.pump import ATMOSPHERE, DENSITY, GRAVITY, Pump, System, vapour_pressure
from src.plant.sensors import PIPE_AREA

WINDOW = 24 * 60  # minutes
MARGIN_WINDOW = 60  # minutes
PARAMETERS = ("head", "power", "discharge", "suction", "margin")


def estimate(frame: pd.DataFrame, residuals: pd.DataFrame, pump: Pump = Pump(), system: System = System()) -> pd.DataFrame:
    def total(values, window=WINDOW):
        return pd.Series(values, index=frame.index).rolling(window, min_periods=window).sum()

    q2 = residuals["flow"] ** 2
    head_e, power_e = residuals["head_expected"], residuals["power_expected"]
    rho_g = DENSITY * GRAVITY

    velocity_head = (residuals["flow"] / 3600 / PIPE_AREA) ** 2 / (2 * GRAVITY)
    available = (ATMOSPHERE + frame["suction_pressure"] * 1000 - vapour_pressure(frame["temperature"])) / rho_g \
        + velocity_head
    margin = available - pump.npsh_required(residuals["flow"], frame["speed"])

    return pd.DataFrame({
        "head": 1 + total(residuals["head"] * head_e) / total(head_e**2),
        "power": 1 + total(residuals["power"] * power_e) / total(power_e**2),
        "discharge": 1 + total(residuals["discharge"] * q2) / total(q2**2) / system.friction_discharge,
        "suction": 1 - total(residuals["suction"] * q2) / total(q2**2) / system.friction_suction,
        "margin": total(margin, MARGIN_WINDOW) / MARGIN_WINDOW,
    })
