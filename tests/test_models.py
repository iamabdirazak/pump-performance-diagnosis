import numpy as np
import pandas as pd
import pytest

from src.models.correction import Correction
from src.plant.pump import Pump


def test_correction_recovers_an_affinity_shaped_gap():
    rng = np.random.default_rng(0)
    speed = rng.uniform(1450, 1750, 2000)
    flow = rng.uniform(0.6, 1.0, 2000) * 57 * speed / 1750
    r, x = speed / 1750, Pump().flow_ratio(flow, speed)
    gap = pd.DataFrame({"head": r**2 * (-0.4 + 0.1 * x), "power": r**3 * 0.05 * x,
                        "suction": 0.01 + 0.02 * (flow / 57) ** 2, "discharge": -0.05 * (flow / 57) ** 2})
    correction = Correction.fit(gap, flow, speed, degree=1)
    predicted = correction.predict(flow, speed)
    for name in gap:
        assert predicted[name].to_numpy() == pytest.approx(gap[name].to_numpy(), abs=1e-9)


def test_correction_extrapolates_by_affinity():
    correction = Correction(1, {"head": [-0.4, 0.0], "power": [0.0, 0.0], "suction": [0.0, 0.0], "discharge": [0.0, 0.0]})
    half = correction.predict(np.array([28.5]), np.array([875.0]))["head"].iloc[0]
    assert half == pytest.approx(-0.4 * 0.25)
