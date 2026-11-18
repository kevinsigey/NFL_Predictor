"""build.py — the single feature-construction entry point for game-outcome
models. Every consumer (training, walk-forward evaluation, live serving)
must call build_training_frame / build_features_for_game rather than
reimplementing feature logic inline — same discipline PL_Predictor's
features/build.py documents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from . import conditions, epa, power_ratings, qb, rest_days, rolling_form

#: Feature BLOCKS beyond the ten base columns. A block joins DEFAULT_BLOCKS only in the PR that shows it clears the
#: evaluation bar (paired-bootstrap intervals + calibration gap, on identical held-out games).
BLOCK_COLUMNS: dict[str, list[str]] = {"epa": epa.epa_columns(), "qb": list(qb.QB_COLUMNS),
                                    "qb_pregame": list(qb.QB_COLUMNS), "conditions": conditions.CONDITION_COLUMNS}
DEFAULT_BLOCKS: tuple[str, ...] = ()


def feature_columns(blocks: tuple[str, ...] = DEFAULT_BLOCKS) -> list[str]:
    unknown = [b for b in blocks if b not in BLOCK_COLUMNS]
    if unknown:
        raise ValueError(f"unknown feature blocks: {unknown}; known: {sorted(BLOCK_COLUMNS)}")
    cols = list(FEATURE_COLUMNS)
    for block in blocks:
        cols += BLOCK_COLUMNS[block]
    return cols


@dataclass
class Aux:
    """The extra inputs the blocks read. `efficiency` is pbp_agg.team_game_efficiency(), `qb_games` is
    pbp_agg.qb_games(), `upcoming_starters` maps (game_id, team) to the expected starting QB's id for games not yet played."""

    efficiency: pd.DataFrame | None = None
    qb_games: pd.DataFrame | None = None
    upcoming_starters: dict | None = None


FEATURE_COLUMNS = [
    "home_pregame_rating", "away_pregame_rating", "rating_diff",
    "home_points_scored_roll", "home_points_allowed_roll",
    "away_points_scored_roll", "away_points_allowed_roll",
    "home_rest_days", "away_rest_days",
    "div_game",
]


def _assemble_base(games_df: pd.DataFrame) -> pd.DataFrame:
    df = power_ratings.compute_pregame_ratings(games_df)
    df = rolling_form.add_rolling_form(df)
    df = rest_days.add_rest_days(df)
    df["rating_diff"] = df["home_pregame_rating"] - df["away_pregame_rating"]
    df["home_rest_days"] = df["home_rest_days"].fillna(7)
    df["away_rest_days"] = df["away_rest_days"].fillna(7)
    if "div_game" not in df.columns:
        df["div_game"] = 0
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    return df


def _assemble(games_df: pd.DataFrame, blocks: tuple[str, ...] = (), aux: Aux | None = None) -> pd.DataFrame:
    df = _assemble_base(games_df)
    if "epa" in blocks:
        if aux is None or aux.efficiency is None:
            raise ValueError("the epa block needs aux.efficiency; refusing to default it to zeros")
        df = epa.add_epa_features(df, aux.efficiency)
    if "qb" in blocks:
        if aux is None or aux.qb_games is None:
            raise ValueError("the qb block needs aux.qb_games; refusing to default it to a neutral QB")
        df = qb.add_qb_features(df, aux.qb_games, upcoming_starters=aux.upcoming_starters)
    if "qb_pregame" in blocks:
        if aux is None or aux.qb_games is None:
            raise ValueError("the qb_pregame block needs aux.qb_games; refusing to default it to a neutral QB")
        df = qb.add_pregame_qb_features(df, aux.qb_games, aux.upcoming_starters)
    if "conditions" in blocks:
        df = conditions.add_condition_features(df)
    return df


def build_training_frame(
    games_df: pd.DataFrame, blocks: tuple[str, ...] = DEFAULT_BLOCKS, aux: Aux | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    columns = feature_columns(blocks)
    df = _assemble(games_df, blocks, aux)
    played = df[df["home_score"].notna() & df["away_score"].notna()].reset_index(drop=True)
    played["margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    return played, columns


def _div_game_for(games_df: pd.DataFrame, home_team: str, away_team: str) -> int:
    """Whether this matchup is a divisional game, read from the schedule.

    `div_game` is a real modelled feature (`FEATURE_COLUMNS`) and training fills it
    from the schedule, where it toggles for roughly a third of games. It used to be
    hardcoded to 0 at serving time, so the model was fitted with the covariate
    varying and served with it permanently constant -- the fitted coefficient was
    dead weight at inference.

    Extracted from `build_features_for_game` so the season-scoping rule can be
    tested without reconstructing a full valid games frame.

    Two rules, both of which the first version of this lookup got wrong:

    - **Scope to the most recent season.** `routes._load_game_history`
      concatenates eight completed seasons and `schedules.fetch_schedules` sorts
      `["season", "week", "gameday"]` -- oldest first -- so the first home/away
      pair on record is from the *earliest* season available. Demonstrated: a frame
      where 2019's BUF-MIA was divisional and 2021's and 2026's were not served
      `div_game = 1` for a 2026 game whose true value is 0.
    - **Take the latest matchup inside that season**, not the first, since a
      division's two meetings are both in the frame.

    Falls back to 0 for any frame without the column, which is also the training
    default, so a games frame that never carried `div_game` still serves.
    """
    if "div_game" not in games_df.columns:
        return 0
    candidates = games_df
    if "season" in games_df.columns and games_df["season"].notna().any():
        candidates = games_df[games_df["season"] == games_df["season"].max()]
    matchup = candidates[
        (candidates["home_team"] == home_team) & (candidates["away_team"] == away_team)
    ]
    if matchup.empty:
        return 0
    if "week" in matchup.columns:
        matchup = matchup.sort_values("week")
    value = matchup.iloc[-1]["div_game"]
    return 0 if pd.isna(value) else int(value)


def build_features_for_game(
    home_team: str, away_team: str, games_df: pd.DataFrame, gameday: str | pd.Timestamp | None = None,
    blocks: tuple[str, ...] = DEFAULT_BLOCKS, aux: Aux | None = None, starters: dict[str, str | None] | None = None,
    game_schedule: dict | None = None,
) -> pd.Series:
    """One feature row for an upcoming home_team vs away_team game, built by the SAME code that builds training rows.

    The upcoming game is appended to the PLAYED games (no result, its own date) and the whole frame goes through
    `_assemble`, so ratings, rolling form and rest days are computed exactly as they are for a training row. This used
    to be a second, hand-written implementation, and it measured rest as the days from the last game to TODAY, not to
    the game: a game five days out was served with five fewer rest days than the model was fitted on.

    `gameday` is the game's date; None means today (an ad-hoc "if they played now" request). `starters` maps each team
    to its expected starting QB id for the `qb` block (never guessed: an unknown starter is a neutral, "new" QB).
    `game_schedule` is optional schedule data for the upcoming game (roof, temp, wind): a dict or a pandas Series.
    """
    played = games_df[games_df["home_score"].notna() & games_df["away_score"].notna()].copy()
    when = pd.Timestamp(gameday) if gameday is not None else pd.Timestamp.now().normalize()
    # Read scheduled conditions (roof/temp/wind) for the upcoming game from the schedule.
    # The schedule may carry roof/temp/wind for upcoming games; fall back to NaN if absent.
    # `game_schedule` is a dict in some callers and the schedule ROW (a pandas Series) in the live routes: test
    # None/length, never truthiness (a Series raises on `if series:`).
    if game_schedule is not None and len(game_schedule) > 0:
        roof = game_schedule.get("roof", np.nan)
        temp = game_schedule.get("temp", np.nan)
        wind = game_schedule.get("wind", np.nan)
    else:
        # Fallback: try to read from the schedule DataFrame if it contains the upcoming game.
        sched_cond = games_df[
            (games_df["home_team"] == home_team) & (games_df["away_team"] == away_team)
            & (pd.to_datetime(games_df["gameday"]) == when)
        ]
        roof = sched_cond["roof"].iloc[0] if not sched_cond.empty and "roof" in sched_cond.columns else np.nan
        temp = sched_cond["temp"].iloc[0] if not sched_cond.empty and "temp" in sched_cond.columns else np.nan
        wind = sched_cond["wind"].iloc[0] if not sched_cond.empty and "wind" in sched_cond.columns else np.nan

    upcoming = {c: np.nan for c in played.columns}
    upcoming.update({
        "game_id": "__upcoming__", "gameday": when, "home_team": home_team, "away_team": away_team,
        "home_score": np.nan, "away_score": np.nan, "div_game": _div_game_for(games_df, home_team, away_team),
        "roof": roof, "temp": temp, "wind": wind,
    })
    if "season" in played.columns and played["season"].notna().any():
        upcoming["season"] = played["season"].max()
    frame = pd.concat([played, pd.DataFrame([upcoming])], ignore_index=True)
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    if starters:
        known = dict(aux.upcoming_starters or {}) if aux is not None else {}
        known.update({("__upcoming__", team): qb_id for team, qb_id in starters.items()})
        aux = Aux(aux.efficiency if aux else None, aux.qb_games if aux else None, known)
    row = _assemble(frame, blocks, aux)
    served = row[row["game_id"] == "__upcoming__"].iloc[0]
    # What `_assemble` produced, in the declared order. A column declared in FEATURE_COLUMNS but never assembled is
    # absent here, and `manifest.load_models` compares this against what each model was fitted on and refuses the
    # mismatch: the declared list is the code's claim, and this is what the code actually builds.
    return served[[c for c in feature_columns(blocks) if c in served.index]]