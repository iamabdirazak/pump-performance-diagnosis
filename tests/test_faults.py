import pytest

from src.plant import faults, sensors
from src.plant.pump import DENSITY, GRAVITY, Pump, System, operating_point


def point(fault, severity=1.0, speed=1750):
    pump, system = faults.apply(Pump(), System(), fault, severity)
    return operating_point(pump, system, speed), system


def test_zero_severity_is_healthy():
    for fault in faults.FAULTS:
        assert point(fault, 0.0)[0] == point(None)[0]


def test_each_fault_moves_the_pump_its_own_way():
    healthy, _ = point(None)
    wear, _ = point("wear")
    blockage, _ = point("blockage")
    clogging, _ = point("clogging")
    assert wear.flow < healthy.flow and wear.head < healthy.head and wear.efficiency < healthy.efficiency
    assert blockage.flow < healthy.flow and blockage.head > healthy.head
    assert clogging.flow < healthy.flow and clogging.npsh_available < healthy.npsh_available - 3


def test_pressure_difference_is_the_head():
    healthy, system = point(None)
    reading = sensors.read(healthy, system, 1750, 20.0)
    head = (reading["discharge_pressure"] - reading["suction_pressure"]) * 1000 / (DENSITY * GRAVITY)
    assert head == pytest.approx(healthy.head)
