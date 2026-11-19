"""Figures for the README."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
BLUE, ORANGE, GREY, RED = "#1f5fa8", "#e69138", "#9aa7b5", "#c0392b"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def traces():
    t = json.loads((ROOT / "results/traces.json").read_text())
    pick = ["inlet valve closed", "rotor imbalance", "outlet valve closed"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 5.4), dpi=130, sharex="col")
    for j, f in enumerate(pick):
        d = t["traces"][f]
        x = np.arange(len(d["score"]))
        an = np.array(d["anomaly"]).astype(bool)
        for ax in axes[:, j]:
            ax.fill_between(x, 0, 1, where=an, transform=ax.get_xaxis_transform(), color=RED, alpha=0.12, lw=0)
        axes[0, j].plot(x, d["flow"], color=GREY, lw=1)
        axes[0, j].set_title(f"{f} ({d['name']})", fontsize=10); axes[0, j].set_ylabel("flow rate (L/min)")
        axes[1, j].plot(x, d["score"], color=BLUE, lw=1)
        axes[1, j].axhline(d["k"], color="k", lw=0.8, ls="--")
        al = np.array(d["alarm"]).astype(bool)
        axes[1, j].scatter(x[al], np.full(al.sum(), 0), s=4, color=ORANGE, label="alarm")
        axes[1, j].set_yscale("symlog"); axes[1, j].set_ylabel("score / limit"); axes[1, j].set_xlabel("seconds")
        for ax in axes[:, j]:
            clean(ax)
    fig.suptitle(f"Real SKAB experiments, {t['detector']}: labelled fault in red, alarms in orange, "
                 "settings chosen on the other groups of experiments", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig1_traces.png"); plt.close(fig)


def tradeoff():
    r = json.loads((ROOT / "results/metrics.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), dpi=130)
    ax = axes[0]
    names = list(r["leave_one_group_out"])
    short = ["T²", "PCA T²+Q", "Isolation forest", "Virtual sensors"]
    a = [r["leave_one_group_out"][n]["binary"]["f1"] for n in names]
    b = [r["leave_one_group_out_events"][n]["binary"]["f1"] for n in names]
    x = np.arange(len(names)); w = 0.38
    b1 = ax.bar(x - w / 2, a, w, color=GREY, label="settings chosen to maximise F1")
    b2 = ax.bar(x + w / 2, b, w, color=BLUE, label="settings chosen for ≤ 2 false alarms per hour")
    ax.bar_label(b1, labels=[f"{v:.2f}" for v in a], fontsize=8); ax.bar_label(b2, labels=[f"{v:.2f}" for v in b], fontsize=8)
    ax.axhline(r["always_alarm"]["f1"], color=RED, lw=1, ls="--")
    ax.text(3.6, r["always_alarm"]["f1"] - 0.035, f"alarm all the time: F1 = {r['always_alarm']['f1']:.2f}",
            fontsize=8, color=RED, ha="right")
    ax.axhline(0.78, color="k", lw=0.8, ls=":"); ax.text(3.6, 0.80, "best published (Conv-AE): 0.78", fontsize=8,
                                                           ha="right")
    ax.set_xlim(-0.6, 3.65)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9); ax.set_ylim(0, 1.0); ax.set_ylabel("F1 (SKAB metric)")
    ax.legend(fontsize=8, frameon=False, loc="lower right"); clean(ax)
    ax.set_title("The leaderboard metric rewards alarming", fontsize=10.5)
    ax = axes[1]
    for key, c, lab in (("leave_one_group_out", GREY, "max F1"), ("leave_one_group_out_events", BLUE, "≤ 2 false alarms/h")):
        for n, s in zip(names, short):
            e = r[key][n]["events"]
            ax.scatter(e["false_alarms_per_hour_normal"], e["detected"] / e["faults"], color=c, s=60, zorder=3)
            off = (6, -12) if (s == "PCA T²+Q" and key == "leave_one_group_out") else (6, -3)
            ax.annotate(s, (e["false_alarms_per_hour_normal"], e["detected"] / e["faults"]), fontsize=8,
                        textcoords="offset points", xytext=off)
    ax.set_xscale("log"); ax.set_xlabel("false alarm episodes per hour of normal operation (log)")
    ax.set_ylabel("faults detected (34 experiments)"); clean(ax)
    ax.set_title("What a maintenance team sees (grey: max F1, blue: alarm budget)", fontsize=10.5)
    fig.tight_layout(); fig.savefig(OUT / "fig2_tradeoff.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    traces(); tradeoff()


if __name__ == "__main__":
    main()
