# Quantile coverage audit (NFL yardage props) -- research, data insufficient

**Status: not measured.** The repo does not contain the data needed to compute
empirical coverage of the q05-q95 / q10-q90 / q25-q75 intervals. Nothing in
this PR changes models, routes or UI.

## Question

`models/player_props.py` ships 19 quantile levels (`QUANTILES`). Does the
nominal 80% interval (q10-q90) contain the realised stat about 80% of the time,
and the 50% interval (q25-q75) about 50%? Has this ever been measured? Not in
the repo.

## What graded data exists in this repo

| Source | Has per-row quantile predictions? | Has realised stat? | Usable? |
|---|---|---|---|
| `evaluate/walk_forward.py::walk_forward_quantile` | Only q10, q50, q90 are emitted, and P(over) at a proxy line. No q05/q25/q75/q95. | Yes (`actual`) | Only if re-run. Its feature frame is not in the repo. |
| `tests/evaluate/test_quantile_walkforward.py` | Synthetic frames only | Synthetic | No |
| `data/public_snapshot.json` | No | Season aggregates only (2026, week 8). No per-game prop rows. | No |
| `data/lines/kalshi_nfl_props_*.jsonl.gz` (21,411 rows) | No model predictions stored | Settlement of a threshold contract (YES/NO at one line), not the stat itself. Settled rows: passing_yards 445, rushing_yards 1,336, receiving_yards 2,583, receptions 2,016. | No. A threshold outcome cannot test an interval without the model's quantiles at that time. |
| `output/` | Only QB agreement CSV (2024 w1-4) | No | No |

The walk-forward output and the player game-log feature frame used to build it
are not checked in. The VPS was not accessed (per the audit brief).

## What was built

* `scripts/quantile_coverage_audit.py` -- reproduces the audit from a
  walk-forward CSV that carries `market, season, week, player_id, actual` plus
  `q05,q10,q25,q50,q75,q90,q95`. Reports per market: n, coverage at 50/80/90,
  mean interval width, pinball loss at seven quantiles against a naive
  per-market unconditional-quantile baseline, coverage by projection tercile
  (q50), and coverage by season period (early/mid/late). Run with no input, it
  writes `insufficient_data` to the JSON.
* `tests/test_quantile_coverage_audit.py` -- planted example (8 of 10 outcomes
  inside a closed interval gives exactly 0.8).
* `docs/quantile-coverage-audit.json` -- the raw result of the run above,
  which is `insufficient_data`.

## To finish the audit

1. Extend `walk_forward_quantile` to emit q05, q25, q75, q95 (the audit depends
   on this; the change is to the harness output, not the model).
2. Run it season by season (2018-2025 per its default) on the feature frame
   built from the player game logs, and save the CSV.
3. Run `python scripts/quantile_coverage_audit.py <csv> docs/quantile-coverage-audit`.

The NBA-style failure mode to look for is tercile shape: intervals that
over-cover low-projection players and under-cover high-projection players.
That cannot be judged from this PR.
