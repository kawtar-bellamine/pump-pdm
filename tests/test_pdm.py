import sys

import numpy as np
import pandas as pd

from pdm.data import SENSORS, experiments, fetch
from pdm.detectors import DETECTORS, smooth, alarms
from pdm.metrics import binary, events


def test_dataset_is_the_published_benchmark():
    e = experiments()
    assert len(e) == 34 and sum(x["group"] == "valve1" for x in e) == 16
    assert all(len(x["train"]) == 400 for x in e)
    # a quirk of the benchmark protocol: in one experiment the fault starts inside the 400 training seconds
    assert [x["name"] for x in e if x["train"]["anomaly"].sum() > 0] == ["other/2"]


def test_metric_matches_the_skab_implementation():
    sys.path.insert(0, str(fetch()))
    from core.metrics import chp_score
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=500, freq="s")
    t = [pd.Series((rng.random(500) < .4).astype(int), index=idx) for _ in range(3)]
    p = [pd.Series((rng.random(500) < .5).astype(int), index=idx) for _ in range(3)]
    f1, far, mar = chp_score(t, p, metric="binary")
    ours = binary([x.values for x in t], [x.values for x in p])
    assert (round(ours["f1"], 2), ours["far"], ours["mar"]) == (f1, far, mar)


def test_smoothing_is_causal():
    x = np.zeros(100); x[60:] = 10
    assert np.all(smooth(x, 15)[:60] == 0)


def test_events():
    t = np.array([0, 0, 1, 1, 1, 0, 0, 0]); p = np.array([1, 0, 0, 1, 1, 0, 1, 0])
    ev = events(t, p)
    assert ev["detected"] == 1 and ev["delays"] == [1] and ev["false_episodes"] == 2


def test_detectors_stay_quiet_on_their_training_data():
    e = experiments()[0]
    for D in DETECTORS.values():
        a, _, _ = alarms(D, e["train"][SENSORS].values, e["train"][SENSORS].values, k=1.0, w=5)
        assert a.mean() < 0.05, D.name
