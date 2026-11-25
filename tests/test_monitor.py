import numpy as np
import pandas as pd
import pytest

from src.deployment.export import build
from src.models.correction import Correction
from src.monitoring.monitor import Monitor
from src.plant import faults, sensors
from src.plant.pump import Pump, System, operating_point


def test_monitor_reads_a_blocked_pipe(tmp_path):
    correction = Correction(1, {name: [0.0, 0.0] for name in ("head", "power", "suction", "discharge")})
    path = tmp_path / "pump.onnx"
    path.write_bytes(build(correction).SerializeToString())
    monitor = Monitor(path)
    rng = np.random.default_rng(0)
    pump, system = faults.apply(Pump(), System(), "blockage", 0.5)
    for minute in range(24 * 60):
        speed = 1500 + 250 * (minute // 120 % 2)
        point = operating_point(pump, system, speed)
        monitor.step(sensors.read(point, system, speed, 20.0, rng))
    assert monitor.health["discharge"] == pytest.approx(1.5, abs=0.05)
    assert monitor.health["head"] == pytest.approx(1.0, abs=0.01)
