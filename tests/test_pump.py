import numpy as np
import pytest

from src.plant.pump import Pump, System, operating_point, vapour_pressure


def test_duty_point_is_the_bep_at_rated_speed():
    point = operating_point(Pump(), System(), 1750)
    assert point.flow == pytest.approx(57.0, rel=1e-6)
    assert point.head == pytest.approx(18.9, rel=1e-6)
    assert point.efficiency == pytest.approx(0.68, rel=1e-6)


def test_affinity_laws_move_the_bep():
    pump = Pump()
    for speed in (1200, 1500):
        ratio = speed / 1750
        assert pump.head(57 * ratio, speed) == pytest.approx(18.9 * ratio**2)
        assert pump.efficiency(57 * ratio, speed) == pytest.approx(0.68)


def test_slowing_down_reduces_flow_and_power():
    pump, system = Pump(), System()
    points = [operating_point(pump, system, s) for s in (1750, 1600, 1400)]
    assert all(a.flow > b.flow for a, b in zip(points, points[1:]))
    assert all(a.power_electrical > b.power_electrical for a, b in zip(points, points[1:]))


def test_healthy_pump_has_suction_margin():
    point = operating_point(Pump(), System(), 1750)
    assert point.npsh_available - point.npsh_required > 3.0


def test_vapour_pressure_of_water():
    assert vapour_pressure(20.0) == pytest.approx(2339, rel=0.01)
    assert vapour_pressure(100.0) == pytest.approx(101_325, rel=0.01)


def test_too_slow_to_lift_raises():
    with pytest.raises(ValueError):
        operating_point(Pump(), System(), 1000)
