# Pre-game starting QB block: negative result, flag stays off

**Verdict: FAIL. `DEFAULT_BLOCKS` stays `()`. Nothing served changes.**

## Why this was tried

The served game model is ridge (margin) + XGBoost (total) on the ten base columns in
`features/build.py::FEATURE_COLUMNS` (`models/manifest.json` `feature_cols`). It loses to the closing line on
both margin and total. The obvious thing the line knows and we do not is who starts at QB.

The QB block already in the code (`features/qb.py::add_qb_features`, block `qb`) is off at serving
(`DEFAULT_BLOCKS = ()`) because it never cleared `docs/nfl_block_eval_results.md`. It also cannot be
judged as a pre-game input: for a played game it uses the **actual** starter (most dropbacks), which
nobody knows before kickoff. That doc's "serving-realistic" run fixed the starter but read the QB's
career totals from **all** his games, including ones after the game being scored.

## What was built (code on branch `claude/nfl-qb-starter-code`, not in this PR)

Block `qb_pregame`, same four columns per side as `qb` (`qb_epa_pd`, `qb_games`, `qb_changed`, `qb_new`),
with every input knowable before kickoff:

- **Starter** = the nflverse depth chart's top QB who is not listed `Out` on that week's nflverse injury
  report (reuses `qb.expected_starters`). Legacy charts (2001-2024): the latest weekly chart at or before
  the game's week. 2025 snapshot feed: the latest snapshot timestamped before the game day (UTC midnight).
  No chart (all of 2000, a few team-weeks): **the team's starter in its previous game**, as the proxy.
- **Quality** = shrunk career EPA per dropback (`SHRINK_K = 200`, prior 0), from that QB's games dated
  strictly before this one (`merge_asof`, `allow_exact_matches=False`). Relief appearances count.
- **Changed** = starter differs from the team's previous-game starter.

Caveats on "pre-game": the legacy weekly charts are not timestamped. They look pre-game (they keep
injured starters listed, which is why agreement is 87%, not ~100%), but that is inferred, not proven.
Injury reports exist only from 2009.

Bug found and fixed on the code branch: play-by-play names OAK/SD/STL by their current codes (LV/LAC/LA),
schedules by the old ones, so every per-team QB lookup missed those clubs' pre-move games (822 team-games).
`features/qb.py::_history` now maps them. The `epa` block's `team_game_efficiency` merge has the same
mismatch and is **not** fixed here.

## Pre-registered gate (written into the tool's docstring before the run)

Repo folds (`walk_forward.prepare_folds`: train on all earlier seasons, score one season), 2000-2025,
held-out 2002-2025. Paired bootstrap (2000 resamples of games, seed 0). Ships only if the margin-MAE CI is
above zero **and** total MAE and Brier are not worse. No parameter was chosen on these folds.

## Result (`output/qb_pregame_eval.txt`)

6,499 held-out games, identical on both sides. Δ = base − block, **positive = block better**.

| Metric | Served model | + qb_pregame | Δ | 95% CI | Closing line | Gap to line closed |
|---|---|---|---|---|---|---|
| Margin MAE (ridge) | 10.5961 | 10.6134 | −0.0173 | [−0.0448, +0.0099] | 10.2492 | **−5.0%** |
| Total MAE (xgb) | 11.0179 | 11.0252 | −0.0073 | [−0.0914, +0.0759] | 10.5613 | **−1.6%** |
| Brier (home win) | 0.22132 | 0.22128 | +0.00004 | [−0.00062, +0.00072] | | |

Starter named for 13,509 of 14,034 team-game slots; it matches who actually started in 87.3% of them.

The block closes none of the gap to the line. Point estimates are slightly worse on both MAEs; neither CI
excludes zero in either direction.

A first run before the relocated-team fix gave margin Δ −0.0239 [−0.0492, +0.0016], total Δ −0.0261
[−0.1113, +0.0595]: also FAIL. Both runs are reported. Fixing the bug was a correctness change, not tuning.

### Post-hoc diagnostic (not part of the gate)

| Games | N | Served | + qb_pregame | Line | Δ margin MAE | 95% CI |
|---|---|---|---|---|---|---|
| Either side's starter changed | 1,477 | 10.674 | 10.768 | 10.312 | −0.094 | [−0.179, −0.008] |
| No change | 5,022 | 10.573 | 10.568 | 10.231 | +0.005 | [−0.019, +0.029] |

The block does worst exactly where it should help. Likely reasons, untested: one chart-named change in eight
is wrong; a new starter's rating is mostly prior (few dropbacks); and the rolling team form already holds the
old QB's level. The line has the confirmed starter and a real view on how good the backup is. We have neither.

## Raw output

`output/qb_pregame_eval_games.csv.gz`: one row per held-out game with actual margin and total, closing
`spread_line` and `total_line`, and raw (uncorrected) base and block margin, total and home-win probability.
Every number above recomputes from it.

## Reproduce (on `claude/nfl-qb-starter-code`)

```bash
PYTHONPATH=src .venv/bin/python -m nfl_predictor.tools.qb_pregame_eval   # writes the CSV, prints the table
```
