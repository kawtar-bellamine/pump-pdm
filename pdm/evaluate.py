"""1. Reproduce two published SKAB leaderboard rows with the authors' own code (fetched, not copied).
2. Evaluate our detectors with the leaderboard protocol and fixed, a-priori settings.
3. Choose the settings honestly: leave one group of experiments out (valve1, valve2, other), tune on the two
   others, test on it. 4. Event view: faults detected, delay, false alarms per hour, by fault type."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from .data import ROOT, SENSORS, experiments, fetch
from .detectors import DETECTORS, alarms, alarms_from, raw
from .metrics import binary, events

K = [0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 12, 20]
MAX_FAR = 10.0      # constraint 1: at most 10% of normal seconds in alarm
MAX_FA_H = 2.0      # constraint 2: at most 2 false alarm episodes per hour of normal operation
W = [1, 5, 15, 30, 60]


def reproduce(exps):
    """The leaderboard's T² and T²+Q rows, with the SKAB repository's T2 class and notebook settings."""
    sys.path.insert(0, str(fetch()))
    from core.metrics import chp_score            # noqa: E402  (SKAB code, GPL-3.0, used as published)
    from core.t2 import T2 as SkabT2              # noqa: E402
    import pandas as pd
    out = {}
    for label, pca in (("T-squared", False), ("T-squared+Q (PCA-based)", True)):
        preds, trues = [], []
        for e in exps:
            m = SkabT2(scaling=True, using_pca=pca)
            m.fit(e["train"][SENSORS]); m.predict(e["test"][SENSORS], window_size=5, plot_fig=False)
            if pca:
                p = ((m.t2["T2"].values > m.t2_ucl) | (m.q["Q"].values > m.q_ucl)).astype(int)
            else:
                p = (m.t2["T2"].values > 2 * m.t2_ucl).astype(int)
            preds.append(pd.Series(p, index=e["test"].index)); trues.append(e["test"]["anomaly"])
        f1, far, mar = chp_score(trues, preds, metric="binary", verbose=False) if "verbose" in \
            chp_score.__code__.co_varnames else chp_score(trues, preds, metric="binary")
        ours = binary([t.values for t in trues], [p.values for p in preds])
        out[label] = dict(skab_metric=dict(f1=f1, far=far, mar=mar), our_metric=ours)
    sys.path.pop(0)
    return out


_RAW = {}


def run(exps, name, k, w):
    if name not in _RAW:
        _RAW[name] = [raw(DETECTORS[name], e["train"][SENSORS].values, e["test"][SENSORS].values) for e in exps]
    P, T, S = [], [], []
    for e, (sc, ucl) in zip(exps, _RAW[name]):
        a, s = alarms_from(sc, ucl, k, w)
        P.append(a); S.append(s); T.append(e["test"]["anomaly"].values)
    return P, T, S


def event_summary(exps, P, T):
    rows = [dict(fault=e["fault"], **events(t, p)) for e, p, t in zip(exps, P, T)]
    seg = sum(r["segments"] for r in rows); det = sum(r["detected"] for r in rows)
    delays = [d for r in rows for d in r["delays"]]
    fe = sum(r["false_episodes"] for r in rows); ns = sum(r["normal_seconds"] for r in rows)
    by = {}
    for f in sorted({r["fault"] for r in rows}):
        rr = [r for r in rows if r["fault"] == f]
        dl = [d for r in rr for d in r["delays"]]
        by[f] = dict(experiments=len(rr), faults=sum(r["segments"] for r in rr), detected=sum(r["detected"] for r in rr),
                     median_delay_s=float(np.median(dl)) if dl else None)
    return dict(faults=seg, detected=det, median_delay_s=float(np.median(delays)) if delays else None,
                p90_delay_s=float(np.percentile(delays, 90)) if delays else None,
                false_alarms_per_hour_normal=round(3600 * fe / ns, 2), by_fault=by)


def recovery_share(exps, name, settings, after_s=180):
    """Share of the false-alarm seconds that fall within after_s seconds after a labelled fault ends: the label stops
    when the operator restores the pump, the sensors take minutes to settle."""
    post = other = 0
    for e in exps:
        s = settings[e["group"]]
        a, _, _ = alarms(DETECTORS[name], e["train"][SENSORS].values, e["test"][SENSORS].values, k=s["k"], w=s["w"])
        t = e["test"]["anomaly"].values.astype(bool)
        after = np.zeros_like(t)
        for i in np.flatnonzero(np.diff(np.r_[t.astype(int), 0]) == -1):
            after[i + 1:i + 1 + after_s] = True
        fa = a & ~t
        post += int((fa & after).sum()); other += int((fa & ~after).sum())
    return dict(within_3_min_after_fault=post, elsewhere=other, share_after_fault=round(post / max(post + other, 1), 3))


def main():
    exps = experiments()
    res = dict(experiments=len(exps), test_seconds=int(sum(len(e["test"]) for e in exps)),
               anomalous_share=round(float(np.mean(np.concatenate([e["test"]["anomaly"].values for e in exps]))), 3))
    res["reproduction"] = reproduce(exps)
    print(json.dumps(res["reproduction"], indent=1), flush=True)
    # fixed a-priori settings: limit from the training part, no smoothing
    res["a_priori"] = {n: binary(*run(exps, n, 1.0, 1)[1::-1]) for n in DETECTORS}
    # leave one group out
    groups = ["valve1", "valve2", "other"]
    cache = {}
    for n in DETECTORS:
        for k, w in itertools.product(K, W):
            P, T, _ = run(exps, n, k, w)
            cache[n, k, w] = (P, T)
    def crit_idx(n, kw, idx, mode):
        P, T = cache[n, *kw]
        if mode == "events":                       # most faults found with at most 2 false alarms per hour
            ev = event_summary([exps[i] for i in idx], [P[i] for i in idx], [T[i] for i in idx])
            ok = ev["false_alarms_per_hour_normal"] <= MAX_FA_H
            return (ok, ev["detected"] if ok else -ev["false_alarms_per_hour_normal"],
                    -(ev["median_delay_s"] or 1e9))
        m = binary([T[i] for i in idx], [P[i] for i in idx])
        if mode == "f1":
            return m["f1"]
        return m["f1"] if m["far"] <= MAX_FAR else -m["far"]

    for mode, key in (("f1", "leave_one_group_out"), ("far", "leave_one_group_out_far10"),
                      ("events", "leave_one_group_out_events")):
        logo = {}
        for n in DETECTORS:
            Pall, Tall, chosen = [None] * len(exps), [None] * len(exps), {}
            for g in groups:
                tr = [i for i, e in enumerate(exps) if e["group"] != g]
                te = [i for i, e in enumerate(exps) if e["group"] == g]
                kw = max(itertools.product(K, W), key=lambda kw: crit_idx(n, kw, tr, mode))
                chosen[g] = dict(k=kw[0], w=kw[1])
                for i in te:
                    Pall[i], Tall[i] = cache[n, *kw][0][i], cache[n, *kw][1][i]
            logo[n] = dict(binary=binary(Tall, Pall), settings=chosen, events=event_summary(exps, Pall, Tall))
        res[key] = logo
    best_global = {}
    for n in DETECTORS:
        kw = max(itertools.product(K, W), key=lambda kw: binary(cache[n, *kw][1], cache[n, *kw][0])["f1"])
        best_global[n] = dict(k=kw[0], w=kw[1], **binary(cache[n, *kw][1], cache[n, *kw][0]))
    T_all = [e["test"]["anomaly"].values for e in exps]
    res["always_alarm"] = binary(T_all, [np.ones_like(t) for t in T_all])
    logo = res["leave_one_group_out_events"]
    res["tuned_on_everything_optimistic"] = best_global
    res["false_alarms_after_fault"] = {n: recovery_share(exps, n, res["leave_one_group_out_events"][n]["settings"])
                                       for n in DETECTORS}
    res["a_priori_events"] = {n: event_summary(exps, *run(exps, n, 1.0, 1)[:2]) for n in DETECTORS}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/metrics.json").write_text(json.dumps(res, indent=1, default=float))
    # traces of one experiment per fault type, for the figure
    best = max(logo, key=lambda n: logo[n]["binary"]["f1"])
    traces = {}
    for e in exps:
        if e["fault"] in traces:
            continue
        s = logo[best]["settings"][e["group"]]
        a, sc, _ = alarms(DETECTORS[best], e["train"][SENSORS].values, e["test"][SENSORS].values, k=s["k"], w=s["w"])
        traces[e["fault"]] = dict(name=e["name"], score=sc.round(3).tolist(), alarm=a.astype(int).tolist(),
                                  anomaly=e["test"]["anomaly"].astype(int).tolist(), k=s["k"],
                                  flow=e["test"]["Volume Flow RateRMS"].round(2).tolist(),
                                  vib=e["test"]["Accelerometer1RMS"].round(4).tolist())
    (ROOT / "results/traces.json").write_text(json.dumps(dict(detector=best, traces=traces)))
    print(json.dumps({k: v for k, v in res.items() if k != "reproduction"}, indent=1, default=float))


if __name__ == "__main__":
    main()
