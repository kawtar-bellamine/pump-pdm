#!/usr/bin/env bash
set -e
python -m pdm.evaluate     # fetches SKAB (pinned commit), reproduces the leaderboard rows, evaluates (about 6 min)
python -m pdm.plots
