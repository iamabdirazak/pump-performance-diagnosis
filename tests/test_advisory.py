import numpy as np
import pandas as pd

from src.monitoring.advisory import ON_DELAY, advise


def hours(values):
    """Expand hourly values to the minute-by-minute frame advise() expects."""
    frame = pd.DataFrame({"head": np.repeat(values, 60)})
    for column, healthy in (("discharge", 1.0), ("suction", 1.0), ("margin", 10.0)):
        frame[column] = healthy
    return frame


def test_one_bad_hour_raises_nothing():
    assert not advise(hours([1.0] * 5 + [0.9] + [1.0] * 5))["wear"].any()


def test_advisory_waits_for_the_on_delay():
    state = advise(hours([1.0] * 3 + [0.96] * 10))["wear"].tolist()
    assert state.index(True) == 3 + ON_DELAY - 1


def test_deadband_holds_the_advisory_near_the_threshold():
    state = advise(hours([0.96] * 8 + [0.975] * 5 + [0.99] * 2))["wear"].tolist()
    assert all(state[ON_DELAY - 1:13]) and not state[-1]
