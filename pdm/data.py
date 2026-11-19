"""SKAB, the Skoltech Anomaly Benchmark: a real water-circulation testbed (centrifugal pump, valves, tank) with
8 sensors sampled every second and labelled faults (Katser & Kozitsin, 2020; github.com/waico/SKAB, GPL-3.0).

Fetched at a pinned commit, not redistributed. 34 experiments: 16 with the inlet valve closed (valve1), 4 with the
outlet valve closed (valve2), 14 others (fluid leaks and additions, rotor imbalance, water level, cavitation).
Benchmark protocol (as in the SKAB repository): in every experiment the first 400 seconds are the training part
(normal operation), the rest is the test part.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/waico/SKAB.git"
COMMIT = "b2c0d46c2971dcbfe71e26087b6d231998bb91c2"     # 11 Aug 2024
DEST = ROOT / "data/SKAB"
SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple",
           "Voltage", "Volume Flow RateRMS"]
TRAIN = 400
OTHER = {"1": "fluid leak/addition", "2": "fluid leak/addition", "3": "fluid leak/addition",
         "4": "fluid leak/addition", "5": "rotor imbalance", "6": "rotor imbalance", "7": "rotor imbalance",
         "8": "rotor imbalance", "9": "rotor imbalance", "10": "water level", "11": "water level",
         "12": "water level", "13": "cavitation", "14": "cavitation"}


def fetch() -> Path:
    if not (DEST / "data").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(DEST)], check=True)
        subprocess.run(["git", "-C", str(DEST), "checkout", "-q", COMMIT], check=True)
    return DEST


def experiments() -> list[dict]:
    """One dict per experiment: name, group, fault type, train and test frames (sensors + labels)."""
    root = fetch() / "data"
    out = []
    for group in ("valve1", "valve2", "other"):
        for f in sorted((root / group).glob("*.csv"), key=lambda p: int(p.stem)):
            df = pd.read_csv(f, sep=";", index_col="datetime", parse_dates=True)
            fault = {"valve1": "inlet valve closed", "valve2": "outlet valve closed"}.get(group) or OTHER[f.stem]
            out.append(dict(name=f"{group}/{f.stem}", group=group, fault=fault,
                            train=df.iloc[:TRAIN], test=df.iloc[TRAIN:]))
    return out


def anomaly_free() -> pd.DataFrame:
    return pd.read_csv(fetch() / "data/anomaly-free/anomaly-free.csv", sep=";", index_col="datetime",
                       parse_dates=True)
