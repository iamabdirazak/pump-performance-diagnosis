"""This unit's healthy behaviour, learned on top of the datasheet physics.

A real pump sits a few percent off its catalogue curve, and its flowmeter
carries a small calibration error. Both leave a smooth, repeatable gap
between the datasheet residuals and zero. The correction learns that gap
on the healthy commissioning month, by least squares.

The gap is given the same structure as the physics, so it stays right
when a fault later pushes the pump somewhere the healthy month never went:

    head       r^2 * p(x)    the affinity law for head
    power      r^3 * p(x)    the affinity law for power
    suction    a + b * Q^2   a pipe only knows flow
    discharge  a + b * Q^2

where x = Q / Q_bep(n) is the flow ratio, r = n / n_rated the speed ratio,
and p a polynomial whose degree is chosen on held-out healthy days.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.residuals import NAMES
from src.plant.pump import Pump

DATASHEET = Pump()


def basis(name: str, flow, speed, degree: int) -> np.ndarray:
    flow, speed = np.asarray(flow, dtype=float), np.asarray(speed, dtype=float)
    if name in ("suction", "discharge"):
        return np.column_stack([np.ones_like(flow), (flow / DATASHEET.flow_bep) ** 2])
    x = DATASHEET.flow_ratio(flow, speed)
    r = speed / DATASHEET.speed_rated
    scale = r**2 if name == "head" else r**3
    return np.column_stack([scale * x**i for i in range(degree + 1)])


@dataclass
class Correction:
    degree: int
    coefficients: dict[str, list[float]]

    @classmethod
    def fit(cls, residuals: pd.DataFrame, flow, speed, degree: int) -> "Correction":
        coefficients = {}
        for name in NAMES:
            solution, *_ = np.linalg.lstsq(basis(name, flow, speed, degree), residuals[name].to_numpy(), rcond=None)
            coefficients[name] = solution.tolist()
        return cls(degree, coefficients)

    def predict(self, flow, speed) -> pd.DataFrame:
        return pd.DataFrame({name: basis(name, flow, speed, self.degree) @ np.array(self.coefficients[name])
                             for name in NAMES})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"degree": self.degree, "coefficients": self.coefficients}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Correction":
        data = json.loads(path.read_text())
        return cls(data["degree"], data["coefficients"])
