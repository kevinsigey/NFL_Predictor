"""Player-prop grading: how close each counted projection landed.

The track record already grades player props as one aggregate MAE per stat
(`tracking/store.py::_prop_markets`). This is the per-pick counterpart --
the NFL half of the prop-grading feature the NBA predictor shipped first
(NBA_Predictor PR #163), with the same semantics, adapted to this repo's
stores: every counted projection for a completed game day, laid against the
player's actual box score, and graded in units of the model's own error.

**Which pick is graded.** The COUNTED pick, exactly the one the track
record counts: the earliest recorded projection per (game, player, market),
selected by the store's own reader (`store._counted_prop_picks`). A row is
LABELLED `made_before_kickoff` from its own `snapshotted_at` against the
game's kickoff (`store._made_before_kickoff`, derived as UTC instants and
failing closed) -- the label is the honesty mechanism, not an exclusion, so
the two can never be silently conflated. (In practice the reconciler refuses
to resolve a non-forward row snapshotted at or after kickoff, so a resolved
row that fails the label is a legacy row, and it is graded and labelled like
any other.)

Only the three yardage markets are graded here -- `passing_yards`,
`rushing_yards`, `receiving_yards`. `anytime_td` is a probability, not a
counting stat, and stays with its confidence-bucket block in the track
record; `receptions`/`carries` are count-model calls with their own
line-and-side grading; forward-test rows (`fwd_*`) carry a book line as
their `predicted_value`, not a projection, and are graded by the forward
report. Grading any of those here would judge a number against a meaning it
never had.

**The grade.** A miss is measured against the model's own published
holdout error for that stat: the walk-forward MAE in
`models/manifest.json` under `quantile_yardage_v1.markets.<stat>.
walkforward_mae` -- the per-market figure the offline gate stamped on the
yardage artifacts (chronological walk-forward folds: train on seasons < Y,
score season Y; the MAE of the model's median forecast). It is the only
per-stat holdout MAE this repo publishes for the yardage markets; the point
regressors' manifest entry (`yardage_metrics`) carries no MAE at all. Bands:

* ``On the number`` -- |actual - projected| <= 0.5 x MAE
* ``Close``         -- |actual - projected| <= 1.0 x MAE
* ``Off``           -- |actual - projected| <= 2.0 x MAE
* ``Way off``       -- beyond that

A stat with no published MAE has no grade: the row is served with its delta
and `grade=None`, never judged against an invented band. The naive MAE the
walk-forward report computes (the trailing-average forecast's error) is not
persisted in any artifact, so `naive_mae` is served as None rather than
recomputed or guessed.

**Most confident.** The section ranks a day's projections by how far the
model moved from the player's own usual line -- |projected - baseline|. The
baseline is the player's causal season-to-date average for the stat: the
mean of his actuals in counted rows whose kickoff is strictly before this
game's, within the same season, reconstructed from the recorded record
itself (this backend stores no rolling-average column on a prop row, so the
recorded actuals are the only stored baseline there is). A player with no
prior counted game this season has no baseline; his rows are graded but
cannot be ranked here.

**Team and opponent.** The prop table carries no team column -- the writer
keys a prop to its game through the player's team at snapshot time and then
discards it. The team a row is served with is therefore read back from the
public snapshot's per-week prop rows (`recent_team`, the team the
projection was made for that week), matched by (game, player) through the
week the game belongs to; the opponent is the game's other side. A player
the snapshot cannot place is served with `team=None` rather than a guess.

**Where it is served.** From the tracking store at request time, like
`/track-record` -- stored data only, never the live pipeline. It is NOT
baked into the public snapshot: that artifact is built in CI, where the
tracking database does not exist (the store lives on the deployment and is
restored from its backup at startup), so a snapshot-baked grading would be
built against an empty store and published as an honest-looking blank.

**Delta direction.** ``delta = actual - projected``: positive is the player
beating the projection, negative is the player falling short. It says
nothing about whether the model was "right" to move -- that is what the
grade is for.

**Game day.** A game's day is its kickoff's date in America/New_York --
the zone the schedule's own `gameday`/`gametime` pair is defined in before
normalisation to UTC -- so a Thursday-night game lands on the Thursday its
audience watched, not on the Friday its UTC instant names.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import store

#: The yardage markets this grading judges, in box-score order. See the
#: module docstring for why the other prop markets are not here.
GRADED_STATS = ("passing_yards", "rushing_yards", "receiving_yards")

GRADE_ON_THE_NUMBER = "On the number"
GRADE_CLOSE = "Close"
GRADE_OFF = "Off"
GRADE_WAY_OFF = "Way off"

#: Grade bands as multiples of the stat's holdout MAE, tightest first. The
#: comparison carries the same epsilon discipline as the NBA grading (a miss
#: of exactly the band is inside the band).
_BANDS = ((0.5, GRADE_ON_THE_NUMBER), (1.0, GRADE_CLOSE), (2.0, GRADE_OFF))
_EPSILON = 1e-9

#: How many rows each ranked section shows.
SECTION_LIMIT = 8

#: The zone a kickoff's game day is read in (see the module docstring).
_GAME_DAY_ZONE = ZoneInfo("America/New_York")


def grade_for(abs_delta: float, mae: float | None) -> str | None:
    """The grade for one miss, or None when there is no measured error to
    judge it against (no band is ever invented)."""
    if mae is None or mae <= 0:
        return None
    ratio = abs_delta / mae
    for multiple, label in _BANDS:
        if ratio <= multiple + _EPSILON:
            return label
    return GRADE_WAY_OFF


def prop_bands(models_dir: Path) -> dict[str, dict[str, float | None]]:
    """The published holdout MAE per graded stat, from the models manifest.

    `{stat: {"holdout_mae": ..., "naive_mae": ...}}` for every graded stat,
    always all three keys: a stat the manifest does not publish a
    walk-forward MAE for gets `holdout_mae=None` (and therefore no grades),
    never a default. `naive_mae` is None for every stat -- the walk-forward
    report computes the trailing-average error but no artifact persists it,
    and a number nobody stored is not reconstructed here.
    """
    stats: dict[str, dict[str, float | None]] = {
        stat: {"holdout_mae": None, "naive_mae": None} for stat in GRADED_STATS
    }
    try:
        manifest = json.loads((Path(models_dir) / "manifest.json").read_text())
    except (OSError, ValueError):
        return stats
    markets = ((manifest.get("quantile_yardage_v1") or {}).get("markets") or {})
    for stat in GRADED_STATS:
        mae = (markets.get(stat) or {}).get("walkforward_mae")
        if isinstance(mae, (int, float)) and not isinstance(mae, bool):
            stats[stat]["holdout_mae"] = float(mae)
    return stats


def team_map_from_snapshot(snapshot: dict) -> dict[tuple[str, str], str]:
    """{(game_id, player_id): team} from the public snapshot's week sections.

    Each week's prop rows carry `recent_team` -- the team the projection was
    made for -- and each week's games carry the two teams playing, so a
    player is placed in the one game of that week his team is in. A player
    the snapshot does not carry, or whose team plays no game in the stored
    week, is simply absent from the map; the caller serves `team=None` for
    him rather than guessing.
    """
    team_map: dict[tuple[str, str], str] = {}
    for week in ((snapshot or {}).get("weeks") or {}).values():
        if not isinstance(week, dict):
            continue
        game_by_team: dict[str, str] = {}
        for game in week.get("games") or []:
            for side in ("home_team", "away_team"):
                team, game_id = game.get(side), game.get("game_id")
                if team and game_id:
                    game_by_team[team] = game_id
        for prop in week.get("player_props") or []:
            team = prop.get("recent_team")
            player_id = prop.get("player_id")
            game_id = game_by_team.get(team) if team else None
            if team and player_id and game_id:
                team_map[(game_id, player_id)] = team
    return team_map


def _empty_payload(stats: dict, now: datetime | None) -> dict:
    return {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "stats": stats,
        "days": [],
    }


def _props_tables_present(conn: sqlite3.Connection) -> bool:
    """Whether the store carries the tables the grading reads at all.

    A store created before player props existed -- or a path that is not a
    database at all -- holds no recorded props by definition. That is an
    empty grading, not an error: the endpoint must still answer, with the
    sections empty, rather than refuse over tables it never promised.
    """
    try:
        found = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('player_prop_predictions', 'game_predictions')"
            )
        }
    except sqlite3.Error:
        return False
    return found == {"player_prop_predictions", "game_predictions"}


def _season_for(game_id: str, game: dict) -> int | None:
    """The season a graded row belongs to, for the baseline's season window.

    The game row's own `season` column first; the nflverse game id's leading
    year (`2026_08_CAR_GB`) when the column is absent or unreadable.
    """
    season = game.get("season")
    if isinstance(season, (int, float)) and not isinstance(season, bool) \
            and pd.notna(season):
        return int(season)
    head = str(game_id).split("_", 1)[0]
    return int(head) if head.isdigit() else None


def _round1(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


def build_prop_grading(
    stats: dict[str, dict[str, float | None]],
    team_map: dict[tuple[str, str], str] | None = None,
    *,
    limit: int = SECTION_LIMIT,
    now: datetime | None = None,
) -> dict:
    """The graded player props for every completed game day in the record.

    Computed from stored data only: recorded predictions and their recorded
    outcomes in the tracking store, the games' kickoffs and teams, the
    manifest's published MAEs (via `prop_bands`, handed in as `stats`), and
    the snapshot's team placements (handed in as `team_map`). Nothing is
    recomputed from model files and nothing is fetched. Days are newest
    first; the window is every day holding at least one counted, resolved
    yardage prop -- the whole recorded record, not a trailing slice.
    """
    payload = _empty_payload(stats, now)
    try:
        with contextlib.closing(store._connect()) as conn:
            if not _props_tables_present(conn):
                return payload
            resolved = pd.read_sql(
                "SELECT * FROM player_prop_predictions WHERE resolved = 1", conn
            )
            games_frame = pd.read_sql(
                "SELECT game_id, home_team, away_team, commence_time, season "
                "FROM game_predictions", conn
            )
    except sqlite3.Error:
        return payload
    if resolved.empty:
        return payload

    resolved = resolved[resolved["market"].isin(GRADED_STATS)]
    resolved = resolved.dropna(subset=["predicted_value", "actual_value"])
    if resolved.empty:
        return payload
    # The counted pick per (game, player, market): the store's own reader,
    # so the grading and the track record can never disagree about which
    # pick a row is.
    counted = store._counted_prop_picks(resolved)

    games = {
        row["game_id"]: row for _, row in games_frame.iterrows()
    } if not games_frame.empty else {}

    # Causal season-to-date baselines, from the counted rows' own recorded
    # actuals: {(season, player_id, stat): [(kickoff instant, actual)]}
    # sorted by instant. A row's baseline is the mean of the entries strictly
    # before its own kickoff -- the graded game never enters its own
    # baseline, and neither does anything later.
    history: dict[tuple, list[tuple[datetime, float]]] = {}
    enriched: list[dict] = []
    for _, pick in counted.iterrows():
        game = games.get(pick["game_id"])
        if game is None:
            # No game row: no kickoff to date the pick by, no teams to label
            # it with, no proof to offer. It cannot be placed, so it is not
            # graded -- the same fail-closed rule the reconciler applies.
            continue
        kickoff = store._utc_instant(game["commence_time"])
        if kickoff is None:
            continue
        enriched.append({"pick": pick, "game": game, "kickoff": kickoff})
        key = (_season_for(pick["game_id"], game), pick["player_id"], pick["market"])
        history.setdefault(key, []).append((kickoff, float(pick["actual_value"])))
    for entries in history.values():
        entries.sort(key=lambda e: e[0])
    if not enriched:
        return payload

    def baseline_for(pick, game, kickoff) -> float | None:
        key = (_season_for(pick["game_id"], game), pick["player_id"], pick["market"])
        prior = [value for instant, value in history.get(key, []) if instant < kickoff]
        return sum(prior) / len(prior) if prior else None

    team_map = team_map or {}
    rows: list[dict] = []
    for entry in enriched:
        pick, game, kickoff = entry["pick"], entry["game"], entry["kickoff"]
        predicted = float(pick["predicted_value"])
        actual = float(pick["actual_value"])
        delta = actual - predicted
        mae = (stats.get(pick["market"]) or {}).get("holdout_mae")
        baseline = baseline_for(pick, game, kickoff)
        team = team_map.get((pick["game_id"], pick["player_id"])) or None
        if team is not None and team == game["home_team"]:
            opponent = game["away_team"]
        elif team is not None and team == game["away_team"]:
            opponent = game["home_team"]
        else:
            # A placement that is not one of the game's two teams proves
            # nothing about either side; serve neither rather than half.
            team, opponent = None, None
        rows.append({
            "player": pick["player_name"],
            "team": team,
            "opponent": opponent,
            "stat": pick["market"],
            "projected": _round1(predicted),
            "actual": _round1(actual),
            "delta": _round1(delta),
            "grade": grade_for(abs(delta), mae),
            "made_before_kickoff": store._made_before_kickoff(
                pick["snapshotted_at"], game["commence_time"]
            ),
            "kickoff": kickoff.isoformat(),
            "_game_id": pick["game_id"],
            "_label": f"{game['away_team']} @ {game['home_team']}",
            "_date": kickoff.astimezone(_GAME_DAY_ZONE).date().isoformat(),
            "_kickoff": kickoff,
            "_abs_delta": abs(delta),
            "_abs_departure": (
                abs(predicted - baseline) if baseline is not None else None
            ),
        })

    def publish(row_list: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in row_list
        ]

    stat_order = {stat: i for i, stat in enumerate(GRADED_STATS)}
    days: list[dict] = []
    for day in sorted({r["_date"] for r in rows}, reverse=True):
        day_rows = [r for r in rows if r["_date"] == day]
        by_game: dict[str, dict] = {}
        for r in sorted(day_rows, key=lambda r: (r["_kickoff"], r["_game_id"])):
            by_game.setdefault(
                r["_game_id"],
                {"game_id": r["_game_id"], "label": r["_label"], "rows": []},
            )["rows"].append(r)
        for game_entry in by_game.values():
            game_entry["rows"] = publish(sorted(
                game_entry["rows"],
                key=lambda r: (r["player"] or "", stat_order.get(r["stat"], 99)),
            ))
        ranked = [r for r in day_rows if r["_abs_departure"] is not None]
        most_confident = sorted(
            ranked,
            key=lambda r: (-r["_abs_departure"], -r["_abs_delta"], r["player"] or ""),
        )[:limit]
        surprises = sorted(
            day_rows, key=lambda r: (-r["_abs_delta"], r["player"] or "")
        )[:limit]
        days.append({
            "date": day,
            "most_confident": publish(most_confident),
            "biggest_surprises": publish(surprises),
            "games": list(by_game.values()),
        })
    payload["days"] = days
    return payload
