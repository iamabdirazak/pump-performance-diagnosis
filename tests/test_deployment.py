import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from src.deployment.export import INPUTS, RESIDUALS, build
from src.features.residuals import compute
from src.models.correction import Correction
from src.plant import sensors
from src.plant.pump import Pump, System, operating_point


def test_graph_matches_the_python_residuals():
    correction = Correction(1, {"head": [-0.3, 0.1], "power": [0.02, -0.01],
                                "suction": [0.01, -0.02], "discharge": [0.05, 0.1]})
    rows = []
    for speed in (1450, 1550, 1650, 1750):
        system = System(level_suction=1.9)
        rows.append(sensors.read(operating_point(Pump(), system, speed), system, speed, 21.0))
    frame = pd.DataFrame(rows)

    session = ort.InferenceSession(build(correction).SerializeToString(), providers=["CPUExecutionProvider"])
    residuals, _ = session.run(None, {"readings": frame[list(INPUTS)].to_numpy(np.float32)})
    reference = compute(frame, correction)[list(RESIDUALS)].to_numpy()
    assert residuals == pytest.approx(reference, abs=1e-4)
