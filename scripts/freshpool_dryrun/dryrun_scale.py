"""The freshpool CPU dry run's scaled constants, read by run.py and dryrun_evaluator.py.

Pre-registered before the first dry run. Each value replaces a study constant only inside the
dry run; the shipped launcher and evaluator keep the study values.
"""

SEED = 42
LABELS = ("alpha", "bravo", "charlie", "delta")  # the fake 4-class label set

# Launcher and evaluator update counts (study: 150 and 600 accepted updates).
RL_SCREEN_UPDATES = 3
FINAL_UPDATES = 6
# E0 trajectory length (study: 4 updates, uninterrupted vs 2+2 vs 1+3): not scaled.
E0_STEPS = 4
# The per-arm cap is 1.25 x forecast, measured on launcher WALL time, process start-up
# included. A tiny arm's start-up dominates its seconds-long active time (SFT-T's cap is
# 1.25 x T_RL), so the dry run widens the factor; the cap logic itself still runs.
CAP_FACTOR = 100.0
# SFT max_steps for the SFT-E boundary template and SFT-T's cumulative step bound.
SFT_MAX_STEPS = 4000

# Not scaled, because they are cheap on the tiny model: P0 scores the first 400 scan
# candidates and samples G4 groups at seeds 42 and 314 on the first 64 (G1), and the live
# useful-mixed window stays 100 accepted updates (so it is inert at 3 and 6 updates).
