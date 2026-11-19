"""Metrics of the SKAB leaderboard (outlier detection, pooled over all test points of all experiments), plus what a
maintenance team asks: how fast is a fault detected, and how many false alarms per hour of normal operation."""
from __future__ import annotations

import numpy as np


def binary(true: list[np.ndarray], pred: list[np.ndarray]) -> dict:
    """F1, false alarm rate and missed alarm rate (%), pooled as in SKAB's chp_score(metric='binary')."""
    t = np.concatenate(true).astype(bool); p = np.concatenate(pred).astype(bool)
    tp = int((t & p).sum()); fp = int((~t & p).sum()); fn = int((t & ~p).sum()); tn = int((~t & ~p).sum())
    return dict(f1=round(tp / (tp + (fn + fp) / 2), 3), far=round(100 * fp / (fp + tn), 2),
                mar=round(100 * fn / (fn + tp), 2), tp=tp, fp=fp, fn=fn, tn=tn)


def events(t: np.ndarray, p: np.ndarray) -> dict:
    """Event view of one experiment: for each labelled anomaly segment, detected or not and the delay (s) to the
    first alarm inside it; alarm episodes that start outside any anomaly are false alarms."""
    t = t.astype(bool); p = p.astype(bool)
    seg = np.flatnonzero(np.diff(np.r_[0, t.astype(int), 0]))
    starts, ends = seg[::2], seg[1::2]
    delays = []
    for s, e in zip(starts, ends):
        hit = np.flatnonzero(p[s:e])
        delays.append(int(hit[0]) if len(hit) else None)
    ep = np.flatnonzero(np.diff(np.r_[0, p.astype(int)]) == 1)
    false = int(sum(not t[i] for i in ep))
    return dict(segments=len(starts), detected=sum(d is not None for d in delays),
                delays=[d for d in delays if d is not None], false_episodes=false, normal_seconds=int((~t).sum()))
