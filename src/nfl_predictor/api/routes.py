"""routes.py — game/player prop/track-record endpoints. Thin HTTP layer
over data/features/models/tracking, same shape as PL_Predictor's/
F1_Predictor's own api/routes.py."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Response

from ..config import (
    CURRENT_SEASON,
    DEPTH_CHARTS_CACHE_DIR,
    PUBLIC_MODE,
    PUBLIC_SNAPSHOT_PATH,
    PUBLIC_SNAPSHOT_REFRESH_URL,
)
from ..data import depth_charts, injuries, player_stats, schedules, teams as teams_data
from ..data import team_efficiency as team_efficiency_mod
from ..data import player_season as player_season_mod
from ..data.forecast import forecast_for
from ..data.forecast_cache import FetchBudget, ForecastCache
from ..data.weather_cache import _kickoff_utc
from ..data.pbp_agg import load_pbp_agg, team_game_efficiency, qb_games
from ..features import build as feature_build
from ..features import player_usage
from ..features import power_ratings
from ..models import game_outcome, manifest, player_props, season_projection
from ..odds import value_bets
from ..tracking import store

router = APIRouter(prefix="/api")

# In-process forecast cache for upcoming games; a few hours of staleness is fine for a weather icon.
_forecast_cache = ForecastCache(ttl=timedelta(hours=3), fetch=forecast_for)

logger = logging.getLogger(__name__)

_public_snapshot_cache: dict | None = None
_public_snapshot_etag: str | None = None


def _public_snapshot() -> dict:
    """The precomputed data public_snapshot.py generates. Cold-start value
    is whatever was baked into this image at build time; a running process
    then keeps it current via refresh_public_snapshot_from_remote below,
    polled on a timer (see api/main.py's lifespan). Empty dict if none has
    been generated yet, so every PUBLIC_MODE branch below just falls
    through to a live compute instead of a 500."""
    global _public_snapshot_cache
    if _public_snapshot_cache is None:
        _public_snapshot_cache = (
            json.loads(PUBLIC_SNAPSHOT_PATH.read_text()) if PUBLIC_SNAPSHOT_PATH.exists() else {}
        )
    return _public_snapshot_cache


def refresh_public_snapshot_from_remote() -> bool:
    """Re-fetch the precomputed public snapshot from its GitHub raw URL and
    swap it in -- lets the scheduled snapshot-refresh GitHub Action reach
    this running process without a Docker rebuild+redeploy. ETag-conditional
    so an unchanged snapshot costs one small request. Any fetch/parse
    failure is swallowed and leaves the previous snapshot serving -- a
    transient network hiccup here must never blank an already-working
    public site."""
    global _public_snapshot_cache, _public_snapshot_etag
    try:
        headers = {"If-None-Match": _public_snapshot_etag} if _public_snapshot_etag else {}
        resp = requests.get(PUBLIC_SNAPSHOT_REFRESH_URL, headers=headers, timeout=30)
        if resp.status_code == 304:
            return False
        resp.raise_for_status()
        snapshot = resp.json()
    except Exception as exc:
        logger.info("public snapshot remote refresh skipped: %s", exc)
        return False
    _public_snapshot_cache = snapshot
    _public_snapshot_etag = resp.headers.get("ETag")
    return True


def _json_safe(value):
    """NaN/inf -> None, recursively.

    Snapshots written before missing lines were handled contain NaN, and Starlette refuses to
    serialize it, so the live /batch returned 500. Sanitizing at the serving edge also covers
    every other non-finite value that a stale snapshot might carry.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _snapshot_week(season: int, week: int) -> dict | None:
    snap = _public_snapshot()
    if snap.get("season") != season:
        return None
    week_snap = snap.get("weeks", {}).get(str(week))
    return None if week_snap is None else _json_safe(week_snap)


def current_season_and_week() -> tuple[int, int]:
    """Current NFL season/week, anchored to the real schedule's own week-1
    kickoff date rather than a hardcoded month/day guess. A fixed
    `date(season, 9, 1)` anchor drifts every year the season's actual
    opening week doesn't start exactly then (confirmed live: it was a full
    week ahead of the real week 1, which kicks off whenever the schedule
    says it does, not on a fixed calendar date)."""
    today = date.today()
    season = today.year if today.month >= 3 else today.year - 1
    try:
        schedule = schedules.fetch_schedules([season])
        week1_start = schedule.loc[schedule["week"] == 1, "gameday"].min()
        anchor = week1_start.date() if pd.notna(week1_start) else date(season, 9, 1)
    except Exception:
        anchor = date(season, 9, 1)
    week = max(1, min(22, ((today - anchor).days // 7) + 1))
    return season, week


@router.get("/current-week")
def get_current_week():
    season, week = current_season_and_week()
    return {"season": season, "week": week}


@router.get("/snapshot-meta")
def get_snapshot_meta() -> dict:
    """When the data this site serves was actually produced.

    The snapshot carries generated_at at its top level, but every route that
    reads it returns a nested block (a week, a standings table) and drops it,
    so no other endpoint can answer "how old are these numbers?". The hub
    needs to say that on the card, and this is also the honest answer for
    anyone tracking how fresh a deployment is.

    Empty dict when no snapshot has been generated yet -- a public deploy
    before its first snapshot -- so this reports source "live" rather than
    raising. That is the real state, not a failure.
    """
    snap = _public_snapshot()
    return {
        "generated_at": snap.get("generated_at"),
        "source": "public_snapshot" if snap else "live",
    }


@lru_cache(maxsize=1)
def _load_models_cached() -> dict:
    return manifest.load_models()


@lru_cache(maxsize=1)
def _load_completed_aux_cached() -> feature_build.Aux:
    """Aux over completed seasons only. Completed seasons never change, so freezing them in
    memory is safe; the current season is layered on fresh per call in `_load_aux_cached`."""
    seasons = schedules.default_completed_seasons(n=8)
    eff_frames, qb_frames = [], []
    for s in seasons:
        try:
            pbp = load_pbp_agg(s)
        except Exception:
            logger.warning("pbp unavailable for completed season=%s; skipping in aux", s)
            continue
        eff_frames.append(team_game_efficiency(pbp))
        qb_frames.append(qb_games(pbp))
    efficiency = pd.concat(eff_frames, ignore_index=True) if eff_frames else pd.DataFrame()
    qb_games_df = pd.concat(qb_frames, ignore_index=True) if qb_frames else pd.DataFrame()
    return feature_build.Aux(efficiency=efficiency, qb_games=qb_games_df, upcoming_starters={})


def _load_aux_cached(season: int) -> feature_build.Aux:
    """Aux for feature blocks: completed-season history (cached) plus the requested season (fresh).

    The completed seasons are frozen in `_load_completed_aux_cached` — safe, because a finished
    season's play-by-play never changes. The requested season is ALWAYS loaded fresh here, never
    from the in-memory cache: an `lru_cache` on this function froze the current season's partial
    aggregate on first request and served it all season. History needs prior seasons too, not just
    the current one: a QB's rating in week 1 comes almost entirely from last season.
    """
    base = _load_completed_aux_cached()
    completed = set(schedules.default_completed_seasons(n=8))
    if season in completed:
        return base
    try:
        pbp = load_pbp_agg(season)
    except Exception:
        logger.warning("pbp unavailable for season=%s; serving completed-season history only", season)
        return base
    eff_frames = [base.efficiency, team_game_efficiency(pbp)] if not base.efficiency.empty else [team_game_efficiency(pbp)]
    qb_frames = [base.qb_games, qb_games(pbp)] if not base.qb_games.empty else [qb_games(pbp)]
    return feature_build.Aux(
        efficiency=pd.concat(eff_frames, ignore_index=True),
        qb_games=pd.concat(qb_frames, ignore_index=True),
        upcoming_starters={},
    )


def _expected_starters_for_week(season: int, week: int) -> dict[str, str | None]:
    """{team: expected starting QB gsis_id} for a scheduled week, fail-open.

    Depth chart's first QB who is not reported Out. Any failure (no chart, no injury
    report, network) returns {} — an unknown starter is served as a neutral new QB downstream,
    never a guess.
    """
    from ..features import qb as qb_features

    try:
        chart_full = depth_charts.load_depth_charts(season, DEPTH_CHARTS_CACHE_DIR)
    except Exception:
        logger.warning("depth chart unavailable for season=%s; no expected starters", season)
        return {}
    if chart_full is None or chart_full.empty:
        return {}
    try:
        chart = depth_charts.resolve_chart(chart_full, season, week)
    except Exception:
        logger.warning("depth chart unresolvable for season=%s week=%s", season, week)
        return {}
    out_ids: set[str] = set()
    try:
        frame = injuries.fetch_injuries(
            [season], max_age_seconds=INJURY_REPORT_MAX_AGE_SECONDS if season == CURRENT_SEASON else None
        )
        if frame is not None and not frame.empty:
            status_by_player = injuries.current_status_by_player(frame, season=season, week=week)
            out_ids = {pid for pid, st in status_by_player.items() if _is_gating_status(st)}
    except Exception:
        logger.warning("injury report unavailable for season=%s week=%s; chart without outs", season, week)
    try:
        return qb_features.expected_starters(chart, out_ids=out_ids)
    except Exception:
        logger.warning("expected starters uncomputable for season=%s week=%s", season, week)
        return {}


def _predict_game_from_models(
    models: dict, home: str, away: str, games_df: pd.DataFrame,
    spread_line: float | None = None, total_line: float | None = None,
    gameday=None,
    blocks: tuple[str, ...] = (), aux: feature_build.Aux | None = None,
    starters: dict[str, str | None] | None = None,
    game_schedule: dict | None = None,
) -> dict:
    feature_row = feature_build.build_features_for_game(home, away, games_df, gameday=gameday, blocks=blocks, aux=aux, starters=starters, game_schedule=game_schedule)
    feature_cols = models["feature_cols"]
    X = feature_row.reindex(feature_cols).fillna(0)

    if models["chosen_candidate"] == "elo":
        predicted_margin = game_outcome.predict_margin_elo(
            {"points_per_rating_point": game_outcome.ELO_POINTS_PER_RATING_POINT},
            feature_row["rating_diff"], feature_row["home_rest_days"], feature_row["away_rest_days"],
        )
    else:
        predicted_margin = float(models["game_outcome_model"].predict(X.to_numpy().reshape(1, -1))[0])

    predicted_total = float(models["total_model"].predict(X.to_numpy().reshape(1, -1))[0])

    result = game_outcome.margin_to_probabilities(
        predicted_margin, models["sigma"],
        spread_line=spread_line, total_line=total_line,
        predicted_total=predicted_total, total_sigma=models["total_sigma"],
    )
    # Not part of the public API response shape, just handed straight to
    # whichever caller wants it (season_projection.py sums this across a
    # team's remaining games for a projected point differential) -- extra
    # dict keys are harmless since this project doesn't validate the
    # /prediction endpoint's response against a schema.
    result["predicted_margin"] = predicted_margin
    result["predicted_total"] = predicted_total
    result["sigma"] = models["sigma"]
    result["total_sigma"] = models["total_sigma"]
    # Which model produced this number, so the frozen snapshot carries its own provenance.
    result["model_version"] = models.get("model_version")
    return result


def _load_game_history(season: int) -> pd.DataFrame:
    """Historical + requested-season game data for feature building.

    Completed seasons are safe to read through load_training_data()'s
    per-season parquet cache — a finished season's results never change.
    But when `season` is CURRENT_SEASON (a season still being played), that
    same call would cache the season's games-so-far on first request and
    then return that SAME STALE snapshot on every later request that
    season, even after more games are played. schedules.py already has a
    dedicated fetch_current_season_partial() for exactly this case
    (refetched every call, no per-season cache) — use it instead of
    folding the current season into load_training_data().
    """
    history_seasons = schedules.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = schedules.load_training_data(history_seasons)
        current_df = schedules.fetch_current_season_partial()
        return pd.concat([historical_df, current_df], ignore_index=True)
    return schedules.load_training_data(history_seasons + [season])


def _load_player_history(season: int) -> pd.DataFrame:
    """Historical + requested-season player stats for feature building.

    Same staleness concern as _load_game_history, but player_stats.py has
    no dedicated "current season, always refetch" helper (only a single
    fetch_weekly_player_stats(seasons, force_refresh) applying to the
    whole list). Rather than extend that module (out of scope for this
    task), fetch the current season on its own with force_refresh=True
    (cheap — one season/week of data per nfl_data_py call) and merge it
    with normally-cached historical seasons.
    """
    history_seasons = schedules.default_completed_seasons(n=8)
    if season == CURRENT_SEASON:
        historical_df = player_stats.fetch_weekly_player_stats(history_seasons)
        try:
            current_df = player_stats.fetch_weekly_player_stats([season], force_refresh=True)
        except Exception:
            # nflverse doesn't publish a weekly-stats file for the current
            # season until its first games are played (404 before then) —
            # historical-only is the correct fallback, not a hard failure.
            logger.info("No weekly player stats available yet for season=%s", season)
            current_df = historical_df.iloc[0:0]
        return pd.concat([historical_df, current_df], ignore_index=True)
    return player_stats.fetch_weekly_player_stats(history_seasons + [season])


@router.get("/games")
def get_games(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["games"]
    return _get_games_live(season, week)


def _get_games_live(season: int, week: int):
    games = schedules.fetch_week_games(season, week)
    # Unplayed games (the entire point of this endpoint) have home_score/
    # away_score as pandas NaN, not None. Starlette's default JSONResponse
    # uses json.dumps(..., allow_nan=False), so an unconverted NaN raises
    # ValueError -> unhandled 500. Swap NaN for None before returning.
    # NOTE: .where(cond, None) alone is NOT enough here -- on a float64
    # column pandas re-casts the replacement None right back into NaN to
    # preserve the column's dtype. astype(object) first forces the frame
    # into per-cell Python objects so None actually sticks (and other
    # values, e.g. gameday's pd.Timestamp, pass through unchanged).
    games = games.astype(object).where(pd.notna(games), None)
    records = games.to_dict("records")
    # Attach kickoff-hour conditions to upcoming games only (no scores yet), in
    # kickoff order and under a per-request fetch budget: the nearest games get
    # their chip first, and a cold process makes at most a handful of Open-Meteo
    # calls instead of one per game (cached games never spend the budget).
    budget = FetchBudget(max_fetches=6)
    upcoming = sorted(
        (g for g in records if g.get("home_score") is None and g.get("away_score") is None),
        key=_kickoff_utc,
    )
    for game in upcoming:
        conditions = _forecast_cache.get(game.get("stadium"), _kickoff_utc(game), budget=budget)
        if conditions is not None:
            game["conditions"] = conditions
    return records


@router.get("/teams/{team}/form")
def get_team_form(team: str, season: int = CURRENT_SEASON, n: int = 5):
    history = _load_game_history(season)
    return {"team": team, "recent_form": _team_recent_form(team, history, n, season)}


def _team_recent_form(team: str, history: pd.DataFrame, n: int, season: int) -> list[dict]:
    # history spans multiple seasons (loaded for power-rating continuity) --
    # recent form should only ever reflect the current season, not bleed in
    # last season's results.
    played = history[
        (history["season"] == season)
        & history["home_score"].notna() & history["away_score"].notna()
        & ((history["home_team"] == team) | (history["away_team"] == team))
    ].sort_values("gameday")
    recent = played.tail(n)

    entries = []
    for _, g in recent.iterrows():
        is_home = g["home_team"] == team
        team_score = g["home_score"] if is_home else g["away_score"]
        opponent_score = g["away_score"] if is_home else g["home_score"]
        opponent = g["away_team"] if is_home else g["home_team"]
        if team_score > opponent_score:
            result = "W"
        elif team_score < opponent_score:
            result = "L"
        else:
            result = "T"
        entries.append({
            "game_id": g["game_id"],
            "opponent": opponent,
            "is_home": bool(is_home),
            "result": result,
            "team_score": int(team_score),
            "opponent_score": int(opponent_score),
            "gameday": g["gameday"],
        })
    return entries


@router.get("/games/{game_id}/head-to-head")
def get_head_to_head(game_id: str, season: int = CURRENT_SEASON, week: int = 1, n_seasons: int = 8):
    teams = _resolve_game_teams(game_id, season, week)
    if teams is None:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    home_team, away_team = teams

    history = _load_game_history(season)
    cutoff_season = season - n_seasons
    history = history[history["season"] >= cutoff_season]
    played = history[history["home_score"].notna() & history["away_score"].notna()]
    meetings = played[
        ((played["home_team"] == home_team) & (played["away_team"] == away_team))
        | ((played["home_team"] == away_team) & (played["away_team"] == home_team))
    ].sort_values("gameday", ascending=False)

    return {
        "game_id": game_id,
        "meetings": [
            {
                "game_id": g["game_id"],
                "season": int(g["season"]),
                "gameday": g["gameday"],
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "home_score": int(g["home_score"]),
                "away_score": int(g["away_score"]),
            }
            for _, g in meetings.iterrows()
        ],
    }


def _resolve_game_teams(game_id: str, season: int, week: int) -> tuple[str, str] | None:
    games = schedules.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        return None
    game = matches.iloc[0]
    return game["home_team"], game["away_team"]


@router.get("/games/{season}/{week}/{game_id}/prediction")
def get_game_prediction(season: int, week: int, game_id: str):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            pred = snap["predictions"].get(game_id)
            if pred is not None:
                return pred
            # Snapshot covers this week but not this specific game's
            # prediction (a build-time failure, or a genuinely unknown
            # game_id) -- fall through to a live compute/404 rather than
            # treating "missing from a partial snapshot" as fatal.
    return _get_game_prediction_live(season, week, game_id)


def _get_game_prediction_live(season: int, week: int, game_id: str):
    games = schedules.fetch_week_games(season, week)
    matches = games[games["game_id"] == game_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Unknown game_id: {game_id}")
    game = matches.iloc[0]

    models = _load_models_cached()
    # Exclude the game's own row from its feature history -- fetch_week_games
    # (unlike the old fetch_upcoming_games) makes already-finished games
    # reachable here too, and without this exclusion a finished game's
    # prediction would leak its own result into its own features. Mirrors
    # the same exclusion background_tracking_tick's backfill path already
    # does for exactly this reason.
    history = _load_game_history(season)
    history = history[history["game_id"] != game_id]
    blocks = tuple(models.get("feature_blocks", []))
    aux = _load_aux_cached(season) if blocks else None
    starters_map = _expected_starters_for_week(season, week) if blocks and "qb" in blocks else {}
    starters = {game["home_team"]: starters_map.get(game["home_team"]), game["away_team"]: starters_map.get(game["away_team"])} if starters_map else None
    prediction = _predict_game_from_models(
        models, game["home_team"], game["away_team"], history,
        spread_line=game.get("spread_line"), total_line=game.get("total_line"),
        gameday=game.get("gameday"), blocks=blocks, aux=aux,
        game_schedule=game, starters=starters,
    )
    return prediction


@router.get("/predictions/{season}/{week}/batch")
def get_predictions_batch(season: int, week: int):
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap["predictions"]
    return _get_predictions_batch_live(season, week)


def _get_predictions_batch_live(season: int, week: int) -> dict:
    games = schedules.fetch_week_games(season, week)
    models = _load_models_cached()
    blocks = tuple(models.get("feature_blocks", []))
    aux = _load_aux_cached(season) if blocks else None
    starters_map = _expected_starters_for_week(season, week) if blocks and "qb" in blocks else {}
    history = _load_game_history(season)
    predictions: dict[str, dict] = {}
    for _, game in games.iterrows():
        try:
            # Exclude the game's own row from its feature history -- same
            # discipline as _get_game_prediction_live: fetch_week_games can
            # return already-finished games, and without this a finished
            # game's prediction would leak its own result into its own
            # features.
            game_history = history[history["game_id"] != game["game_id"]]
            starters = {game["home_team"]: starters_map.get(game["home_team"]), game["away_team"]: starters_map.get(game["away_team"])} if starters_map else None
            predictions[game["game_id"]] = _predict_game_from_models(
                models, game["home_team"], game["away_team"], game_history,
                spread_line=game.get("spread_line"), total_line=game.get("total_line"),
                gameday=game.get("gameday"), blocks=blocks, aux=aux,
                game_schedule=game, starters=starters,
            )
        except Exception:
            logger.exception("batch prediction failed for game_id=%s", game.get("game_id"))
            continue
    return predictions


class PlayerPropsUnavailable(RuntimeError):
    """Player props for a week could not be produced, as opposed to there being
    none to produce.

    These are different facts and the difference is the whole point. A reader
    shown "no props for this game" when the pipeline actually broke has been told
    something false, and there is no way to tell the two apart from the response
    alone -- the old `except Exception: return []` made a 404 on the season's
    stats, a dead upstream host, and a genuine empty week all arrive as `200 []`.

    Raised rather than logged-and-returned so the HTTP layer can answer 503 and
    the snapshot builder can tell a failed rebuild from a week with nothing in
    it. The live path's other callers (background_tracking_tick, _build_week)
    already wrap this call in their own try/except, so they keep their behaviour.

    The message on this exception is for the LOG, never for a reader: `str()` on a
    urllib failure carries the upstream host and port, and the sentence a reader
    gets is a fixed constant (`PROPS_UNAVAILABLE_DETAIL`) that cannot contradict
    itself the way an appended cause can.
    """


#: What a reader is told when props could not be produced. Fixed text, on purpose.
#: Interpolating the cause leaked `HTTPSConnectionPool(host='github.com',
#: port=443): Read timed out` into a public response body, and bolting "could not
#: be loaded" onto a message that already said "could not be loaded" read as two
#: separate failures. The log carries the detail; the body does not.
#: "this week", not "this game". The route is keyed on a week and renders one
#: props table for it, and this string is rendered verbatim by the frontend
#: (`client.ts` puts `body.detail` in the Error, the component puts `err.message`
#: in the alert), so "for this game" put the same unit mismatch in the error
#: message that the empty state was fixed for.
PROPS_UNAVAILABLE_DETAIL = (
    "Player props are temporarily unavailable for this week. The projections could "
    "not be produced. This is a failure, not a week without props."
)

#: Set on a 200 whose rows came from a previous snapshot build because this
#: week's rebuild failed. Carrying the last good rows forward is worth it -- they
#: are pregame projections for the same week, which is all the accuracy rule asks
#: -- but a reader must be able to tell they are not from this build.
PROPS_STALE_HEADER = "X-Player-Props-Stale"


def snapshot_props_unavailable(snap: dict) -> str | None:
    """Why this snapshot week cannot serve player props, or None if it can.

    One implementation, called by BOTH readers of the snapshot -- the props route
    and `facts._props`. While those were separate, the props page said "could not
    be loaded" for a week while the facts panel beside it said nothing at all, for
    the same week and the same reader. Two panels disagreeing about one fact is
    the defect; sharing the rule is the fix.

    Order matters, and the `games` check is deliberately last:

    1. rows present -> serve them, whatever the status says. A build that recorded
       "stale" still carried real rows, and blanking those would discard exactly
       what the carry-forward exists to preserve.
    2. `player_props_status == "unavailable"` -> the build itself said it could not
       produce rows. This is the reason the build writes the key.
    3. no status key (or an unrecognised one) and the week has games -> a pre-key
       snapshot. Every NFL game has players in it, so a full slate with zero props
       is only reachable by something breaking. This is what covers the currently
       committed file, whose weeks 2-7 predate the key.
    4. no rows and no games -> an honest empty. Weeks 19-22 look exactly like it.
    """
    if not snap:
        return None
    if snap.get("player_props"):
        return None
    if snap.get("player_props_status") == "unavailable":
        return "the last build recorded that props could not be produced for it"
    if snap.get("games"):
        return f"the snapshot has {len(snap['games'])} game(s) and no player props for it"
    return None


# --- the availability gate --------------------------------------------------
#
# The official weekly injury report, as published by nflverse, is the one feed
# that says a player is not playing. `data/injuries.py` existed for the whole
# life of the props endpoint and nothing called it: its only references in the
# tree were itself, its test and a plan document. So the spec's claim that NFL
# injuries "gate props today" was false, and this is the code that makes it
# true rather than a feature being switched on.

#: The ONLY report status that removes a player. `Doubtful` and `Questionable`
#: are deliberately excluded: a player reported questionable very often plays,
#: and dropping him from a ranked list asserts that he definitely will not.
#:
#: Compared as a normalised whole token, so an unreviewed future value like
#: "Out (Ankle)" fails to match instead of matching loosely. A gate that widens
#: itself on a string nobody looked at is a gate nobody reviewed, and the cost of
#: a wrong match here is a real player deleted from a list.
GATING_INJURY_STATUS = "out"

# How long the CURRENT season's injury report may be reused before it is refetched.
INJURY_REPORT_MAX_AGE_SECONDS = 3600

#: What an out entry is, in words. The frontend renders this verbatim, so a bare
#: player name with no source and no date is not a claim anybody can check.
INJURY_SOURCE = "nflverse official weekly injury report"


def _is_gating_status(status) -> bool:
    """True only for a report status that means the player is not playing."""
    if status is None or not isinstance(status, str):
        return False
    return status.strip().lower() == GATING_INJURY_STATUS


def _out_players_for(season: int, week: int) -> list[dict]:
    """Every player the official report lists Out for this season and week.

    **Never raises, and an absence is always `[]`.** Four different absences are
    all expected, not exceptional: the fetch failing, an empty release, a
    release that carries no row for the requested week (the per-season cache is
    written once and never expires, so a week-1 cache read for week 12 is the
    normal case late in a season), and a feed that hands back something that is
    not a frame at all.

    Every one of them resolves to "gate nobody", and that is the asymmetry this
    whole function is built around. A false removal deletes a real player from a
    ranked list a reader is about to act on; a missed removal shows one player
    who does not play. The first is the worse error by a wide margin, so an
    unknown feed is never allowed to remove anything -- it can only ever fail
    open.
    """
    try:
        # The current season's report changes through the week, so its cache expires;
        # a finished season's never does. Without this the gate read one frozen file.
        frame = injuries.fetch_injuries(
            [season], max_age_seconds=INJURY_REPORT_MAX_AGE_SECONDS if season == CURRENT_SEASON else None
        )
    except Exception as injury_err:  # noqa: BLE001 - absence must gate nobody
        logger.warning("injury report unavailable for %s wk%s: %s", season, week, injury_err)
        return []
    if frame is None or frame.empty:
        return []

    try:
        status_by_player = injuries.current_status_by_player(frame, season=season, week=week)
    except Exception as injury_err:  # noqa: BLE001 - absence must gate nobody
        logger.warning("injury report unreadable for %s wk%s: %s", season, week, injury_err)
        return []
    if not status_by_player:
        return []

    # The name and team come out of the SAME report frame that called the player
    # out, so an entry cannot cite a source and disagree with it. `player_id` on
    # the props path is the GSIS id (`00-…`), the same key the report carries,
    # so this join needs no name matching -- which is the failure mode NBA's
    # equivalent of this gate has.
    week_rows = frame[frame["week"] == week]
    by_id = {str(row.get("gsis_id")): row for _, row in week_rows.iterrows()}

    entries = []
    for player_id, status in status_by_player.items():
        if not _is_gating_status(status):
            continue
        row = by_id.get(str(player_id))
        entries.append({
            "player_id": player_id,
            "player_name": (row.get("full_name") if row is not None else None) or str(player_id),
            "recent_team": (row.get("team") if row is not None else None) or "",
            "report_status": status,
            "report_season": season,
            "report_week": week,
            "source": INJURY_SOURCE,
        })
    return entries


@router.get("/players/{season}/{week}/out")
def get_out_players(season: int, week: int):
    """The players the official report lists Out for this week.

    Separate from `/props` rather than a field on it. That body is a bare JSON
    array and three things depend on it staying one: `public_snapshot` stores it,
    `facts._props` reads it, and the frontend client types it as
    `PlayerPropPrediction[]`. A picks list is not a reason to break it.

    Always 200 with a list, including when the feed is unavailable. Not 503:
    "we could not check" and "nobody is out" are different facts, and the only
    thing a caller can honestly render for both is nothing -- a 503 here would
    take down a picks page over a transient upstream blip, which is strictly
    worse than showing the ranking ungated.

    There is deliberately no `except Exception` backstop, for the reason
    `get_player_props` gives for not having one either: `_out_players_for`
    already swallows every failure and returns `[]`, so a second net could only
    be unreachable code. The contract that makes it unreachable is pinned by
    `TestAbsenceRemovesNobody`, which reaches the real function rather than
    trusting this comment.
    """
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap.get("player_props_out") or []
    return _out_players_for(season, week)


@router.get("/players/{season}/{week}/market-lines")
def get_market_lines(season: int, week: int):
    """What Kalshi is pricing for each player's yards / receptions ladder right now (free public API, cached 10 minutes).

    `p_over` is the market's probability of clearing `line` (324.5 for a '325+' rung). Always 200: when Kalshi is
    unreachable or has nothing listed the answer is an empty list, never an error, because it only decorates the line
    checker. Players are matched by team + first initial + last name (see `kalshi_lines.parse_player_token`).

    For WR/TE receptions and QB passing TDs only, the ladder also carries
    `model_p` (the model's P(over) at that rung's line) and `generated_at`
    (when the probability was computed). These are the two fields the paper
    ledger needs to log a prop candidate without importing models. Yards markets
    do NOT carry `model_p`: their probability is a quantile interpolation that
    the paper ledger does not yet consume (PR 1b scope). Fail-soft: if the prop
    pipeline is unavailable, the rungs simply omit both fields.
    """
    from ..data import kalshi_lines

    data = kalshi_lines.current_ladders()
    try:
        _attach_prop_model_p(data, season, week)
    except Exception as exc:  # noqa: BLE001 - enhancement; never break the line checker
        logger.warning("market-lines model_p attach failed for %s wk%s: %s", season, week, exc)
    return data


def _prop_name_key(name: str) -> tuple[str, str] | None:
    """(initial, last) for a prop row's player_name, in Kalshi-token form.

    Prop rows carry either a full name ("Patrick Mahomes") or nflverse's
    abbreviated form ("P.Mahomes") -- the served rows are the abbreviated
    ones. Kalshi's token is first initial + last name, letters only: the
    same rule the frontend's `marketLadderFor` matches ladders by. The
    abbreviated form used to fall through with an empty last name, so no
    rung ever matched and `model_p` silently never attached.
    """
    import re

    n = (name or "").strip()
    if not n:
        return None
    dot = n.find(".")
    if dot > 0:
        first, rest = n[:dot], n[dot + 1:]
    else:
        parts = n.split()
        first, rest = parts[0], "".join(parts[1:])
    initial = re.sub(r"[^A-Za-z]", "", first).upper()[:1]
    last = re.sub(r"[^A-Za-z]", "", rest).upper()
    return (initial, last) if initial and last else None


def _props_for_model_p(season: int, week: int) -> list:
    """The prop rows `model_p` is read from.

    In PUBLIC_MODE these are the snapshot's stored rows -- the same rows
    `/props` serves -- never the live pipeline. The public deployment's own
    rule (see config.py) is that no request runs the live feature-building
    pipeline, and running it here made every market-lines call pay a full
    props computation (~17s) even when the ladders themselves were cached.
    Outside PUBLIC_MODE there is no snapshot to read, so the live rows are
    the source, as before.
    """
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            return snap.get("player_props") or []
    return _get_player_props_live(season, week)


def _attach_prop_model_p(data: dict, season: int, week: int) -> None:
    """Attach `model_p` and `generated_at` to receptions/passing_tds rungs only.

    Fail-soft: any failure leaves the rungs as they were. No raw quantiles are
    ever exposed.
    """
    from datetime import datetime, timezone

    try:
        props = _props_for_model_p(season, week)
    except Exception:
        return
    if not props:
        return
    # Build a lookup from (team, initial, last) -> (receptions_p_over | passing_td_over_prob, generated_at).
    lookup: dict[tuple, tuple[float, str]] = {}
    generated_at = None
    for p in props:
        name = p.get("player_name") or ""
        team = p.get("recent_team")
        if not team or not name:
            continue
        parsed = _prop_name_key(name)
        if parsed is None:
            continue
        initial, last = parsed
        key = (team, initial, last)
        # receptions: receptions_p_over; passing TDs: passing_td_over_prob
        # generated_at comes from the props row (when the pipeline computed it);
        # fall back to now so a stamped-in-time snapshot still lands.
        gen = p.get("generated_at")
        if gen is None:
            gen = datetime.now(timezone.utc).isoformat()
        if p.get("receptions_p_over") is not None:
            lookup.setdefault(key, {})["receptions"] = (float(p["receptions_p_over"]), gen)
        if p.get("passing_td_over_prob") is not None:
            lookup.setdefault(key, {})["passing_tds"] = (float(p["passing_td_over_prob"]), gen)

    for ladder in (data.get("lines") or []):
        who = ladder  # already has team/initial/last from parse_player_token
        team = ladder.get("team")
        initial = ladder.get("initial")
        last = ladder.get("last")
        stat = ladder.get("stat")
        if stat not in ("receptions", "passing_tds"):
            continue
        key = (team, initial, last)
        info = (lookup.get(key) or {}).get(stat)
        if info is None:
            continue
        model_p, gen = info
        for rung in (ladder.get("rungs") or []):
            rung["model_p"] = model_p
            rung["generated_at"] = gen


@router.get("/players/{season}/{week}/props")
def get_player_props(season: int, week: int, response: Response = None):
    """The HTTP endpoint.

    `response` is FastAPI's per-request object, used only to set the stale
    header, and it defaults to None for one reason: **`facts._props` calls this
    function directly, off-HTTP, in the non-PUBLIC_MODE branch only**
    (`facts.py`: `routes.get_player_props(season, week)`). FastAPI injects a real
    Response for a served request and ignores the default.

    The narrower the reason, the better here. A previous version of this comment
    claimed the call was to "reuse the PUBLIC_MODE snapshot rule" -- false, since
    `_props` calls `routes.snapshot_props_unavailable` directly in PUBLIC_MODE and
    never reaches this function. That mattered beyond the prose: reintroducing a
    PUBLIC_MODE call from `_props` would pass `response=None`, and the stale-header
    branch would raise `AttributeError` on it, outside the `try` below -- Critical
    1's exact 500, reintroduced through a comment. The PUBLIC_MODE branch must
    call `snapshot_props_unavailable`, not this function.
    """
    if PUBLIC_MODE:
        snap = _snapshot_week(season, week)
        if snap is not None:
            reason = snapshot_props_unavailable(snap)
            if reason is not None:
                logger.error("player props unavailable for season=%s week=%s: %s", season, week, reason)
                raise HTTPException(status_code=503, detail=PROPS_UNAVAILABLE_DETAIL)
            if snap.get("player_props_status") == "stale":
                # Served, but labelled: a stale pregame projection is still a
                # pregame projection, and presenting it as this build's is not.
                response.headers[PROPS_STALE_HEADER] = "true"
            return snap.get("player_props") or []
    try:
        return _get_player_props_live(season, week)
    except PlayerPropsUnavailable as exc:
        # No `except Exception` backstop here, and that is deliberate.
        # `_get_player_props_live` wraps its whole body, so every Exception it can
        # raise has already become a PlayerPropsUnavailable before control gets
        # back here; a second net could only be unreachable code. The contract
        # that makes it unreachable is pinned by
        # `test_the_live_path_never_leaks_a_raw_exception` rather than by a
        # branch nobody can reach -- so a mutation deleting THIS handler has to be
        # caught by that test, not by anything here.
        logger.error("player props unavailable for season=%s week=%s: %s", season, week, exc)
        raise HTTPException(status_code=503, detail=PROPS_UNAVAILABLE_DETAIL) from exc


def _get_player_props_live(season: int, week: int, out_players: list | None = None):
    """The live prop rows for this week, with listed-out players removed.

    `out_players` is an optional collector rather than a second return value
    because this function is called from five places with two different
    expectations -- `get_player_props` wants only rows, and `public_snapshot`'s
    `_build_week` wants both the rows and the out entries to store in the
    artifact production actually serves. Returning a tuple would force every one
    of those call sites, including the ones that have no use for the out list,
    to unpack it and drop half.

    The gate runs at the very END, after the total-failure check below. Applying
    it earlier would let a slate where every player is listed Out trip the
    "produced nothing at all" rule and answer 503 -- which claims the props
    pipeline broke, on a week where it worked perfectly and the honest answer is
    an empty ranking plus an out list that explains it.
    """
    try:
        models = _load_models_cached()
        player_history = _load_player_history(season)

        # 1. Fetch upcoming games for this specific week to find active teams
        upcoming_games = schedules.fetch_upcoming_games(season, week)
        if upcoming_games.empty:
            # Fallback to completed partial games if none are upcoming
            upcoming_games = schedules.fetch_current_season_partial()
            upcoming_games = upcoming_games[upcoming_games["week"] == week]

        if upcoming_games.empty:
            # No games in this week at all. Nothing to project and nothing broken --
            # this is the one empty result that is a true answer, and it is the
            # only one allowed to reach a reader as an empty list.
            return []

        active_teams = set(upcoming_games["home_team"]).union(set(upcoming_games["away_team"]))

        # 2. Filter players only belonging to the active teams playing this week,
        #    and only at a position that has a market. `predict_props` emits
        #    `anytime_td_prob` for *every* position and yardage markets only for
        #    QB/RB/WR/TE, so an unfiltered player list turns every defender and
        #    lineman into an anytime-TD prop. That was nearly free while the
        #    weekly stats came from nflverse's `player_stats` release, which
        #    publishes a row only for players with an offensive stat: 50 of 612
        #    players in 2024. The `stats_player` release publishes a row for
        #    every player who appeared in a game, which takes it to 972 of 1,422
        #    so far in 2026 -- and `build_features_for_player` returns a row for
        #    each of them, so they would all be snapshotted and graded.
        #
        #    The roster fallback below has always applied this filter; this
        #    branch not applying it was an inconsistency that only showed up
        #    once the stats source stopped being offensive-only.
        latest_players = (
            player_history[
                (player_history["season"] == season) &
                (player_history["recent_team"].isin(active_teams)) &
                (player_history["position"].isin(player_props.POSITION_MARKETS))
            ]
            [["player_id", "player_name", "position", "recent_team"]]
            .drop_duplicates("player_id")
        )

        # The season being predicted has no player stats of its own. This is a
        # real, currently-live condition: nflverse 404s player_stats_2026.parquet
        # (see docs/player-prop-accuracy-blocker.md), so `_load_player_history`
        # returns other seasons only. Every player below would then be skipped for
        # want of a pregame feature row and the route answered `200 []` -- an
        # empty state indistinguishable from a week with no props. Say so instead.
        if latest_players.empty and (player_history["season"] == season).sum() == 0:
            raise PlayerPropsUnavailable(
                f"no player stats are available for season {season}, so no props can be "
                f"projected for week {week} of it (upstream player_stats_{season}.parquet is "
                "not published); this is an upstream data gap, not a week without props"
            )

        # 3. Fallback: a team with no current-season stats yet (week 1, or a
        # bye-to-opener gap) has no rows above even though its roster exists —
        # pull the season roster for just those teams so props aren't empty.
        # `found_teams` is computed from the unfiltered history below, so a team
        # whose only current-season rows belong to non-market positions still
        # counts as found and does not trigger this.
        found_teams = set(
            player_history[
                (player_history["season"] == season) &
                (player_history["recent_team"].isin(active_teams))
            ]["recent_team"].unique()
        )
        missing_teams = active_teams - found_teams
        if missing_teams:
            try:
                roster = player_stats.fetch_seasonal_roster(season)
                fallback = roster[
                    roster["recent_team"].isin(missing_teams) & roster["position"].isin(player_props.POSITION_MARKETS)
                ]
                latest_players = pd.concat([latest_players, fallback], ignore_index=True).drop_duplicates("player_id")
            except Exception as roster_err:
                logger.warning("Failed to fetch season roster fallback for teams=%s: %s", missing_teams, roster_err)

        # Starter flags from the depth chart, when it is available. The chart is
        # keyed on full_name + club_code, and `recent_team` is the same club
        # abbreviation, so the join needs no id mapping.
        #
        # An unavailable chart is NOT an error and must not raise: it leaves
        # `is_starter` as None, which the UI renders as the visible "Projected
        # order" state. `False` would be a different and much worse claim --
        # it would say this player is known to be on the bench, which is an
        # assertion about depth-chart data we do not have.
        try:
            chart_flags = depth_charts.flags_for_season_week(season, week, DEPTH_CHARTS_CACHE_DIR)
        except Exception as chart_err:  # noqa: BLE001 - never fatal to props
            logger.warning("depth charts unavailable for %s wk%s: %s", season, week, chart_err)
            chart_flags = {}

        results = []
        # `skipped` and `failed` are different facts and the error message has to
        # keep them apart. "every player failed (0 of 1 failed)" -- which is what
        # the first cut of this said -- is self-contradictory, and it is what you
        # get when the roster fallback supplies a name and `build_features_for_player`
        # returns None for them all. One is a prediction error; the other is a
        # missing pregame feature row.
        failed = []
        skipped = []
        for _, player in latest_players.iterrows():
            try:
                feature_row = player_usage.build_features_for_player(
                    player["player_id"], player_history, season=season, week=week
                )
                if feature_row is None:
                    # No usage history in the season being predicted — a true rookie,
                    # or a player the roster fallback pulled in. Skip rather than
                    # fabricating a zero row: the model is never trained on the
                    # all-zero region, so scoring it returns the origin intercept
                    # rather than a prediction. Measured 2026-09-27, that path made
                    # 430 of 880 live rows (48.9%) bit-identical, serving 62.592
                    # passing yards to real quarterbacks. This matches the sibling
                    # CFB route, which already skipped.
                    skipped.append(player.get("player_id"))
                    continue
                props = player_props.predict_props(models["player_models"], feature_row, position=player["position"])
                flag = chart_flags.get(player["player_id"]) or chart_flags.get(player["player_name"])
                results.append({
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                    "recent_team": player["recent_team"],
                    "position": player["position"],
                    # None when there is no depth-chart data. See above: this is
                    # "unknown", not "not a starter".
                    "is_starter": flag["is_starter"] if flag else None,
                    "depth_slot": flag["depth_slot"] if flag else None,
                    # His own plain last-5-games mean per stat (empty if none). Position-invariant key on
                    # purpose: public_snapshot rebuilds reused weeks whose rows lack it.
                    "recent_avg": player_usage.recent_averages(player["player_id"], player_history, season, week),
                    **props,
                })
            except Exception as player_err:
                failed.append(player.get("player_id"))
                logger.warning("Failed to predict props for player_id=%s: %s", player.get("player_id"), player_err)
                continue
        # Producing nothing at all is an outage, not a week without props. One bad
        # player is a tolerable gap and is still a partial result, so only the
        # total-failure case is promoted to an error. Returning `[]` here is what
        # made a broken prediction path look like an empty slate.
        #
        # The `not results` half is load-bearing and was one token from being lost:
        # with `and failed:` bolted on, a week where the roster fallback supplied
        # every name and `build_features_for_player` returned None for all of them
        # (`failed` empty, `skipped` full) would fall straight through to `200 []`.
        # The two causes get separate sentences because they are different bugs
        # upstream -- a dead model backend versus a missing pregame feature row.
        if latest_players.shape[0] and not results:
            considered = latest_players.shape[0]
            if skipped and not failed:
                raise PlayerPropsUnavailable(
                    f"no props for season {season} week {week}: all {considered} player(s) were "
                    "skipped for want of a pregame usage history in the season being predicted, "
                    "so none could be scored"
                )
            raise PlayerPropsUnavailable(
                f"no props for season {season} week {week}: {len(failed)} of {considered} "
                f"player(s) failed to predict (first={failed[:3]}) and {len(skipped)} were "
                "skipped for want of a pregame usage history"
            )

        # The availability gate, and it is LAST on purpose -- see the docstring.
        # Everything above measures what the pipeline produced; the gate removes
        # from that result without ever being able to turn it into an error.
        gated = _out_players_for(season, week)
        if gated:
            ranked_ids = {str(row["player_id"]) for row in results}
            gated_ids = {str(entry["player_id"]) for entry in gated}
            kept = [row for row in results if str(row["player_id"]) not in gated_ids]
            # An entry for a player who was not in this week's ranking is not an
            # out player of it, and the frontend renders whatever it is handed as
            # the reason a name is missing from the lists. Reporting one would be
            # attributing a removal that never happened.
            entries = [entry for entry in gated if str(entry["player_id"]) in ranked_ids]
            if out_players is not None:
                out_players.extend(entries)
            if len(kept) != len(results):
                logger.info(
                    "injury gate removed %d of %d prop row(s) for %s wk%s",
                    len(results) - len(kept), len(results), season, week,
                )
            results = kept

        # Raw quantile dicts are an internal handoff, not an API field: strip
        # them before the rows leave. (The directionality patch consumes
        # `{market}_quantiles` to compute `{market}_p_over` + line and strips
        # them itself; it runs before this point, so this is a no-op safety
        # net once it lands.)
        for row in results:
            for key in [k for k in row if k.endswith("_quantiles")]:
                del row[key]

        return results
    except PlayerPropsUnavailable:
        # Already carries the reason. Logging it again as a traceback here would
        # bury the message that explains the 503 a reader is about to get.
        raise
    except Exception as e:
        # The old `return []`. A 404 on the season's stats, a dead upstream host
        # and a crash in feature building all used to land here and reach a reader
        # as an empty props table.
        logger.exception("Failed to load player props for season=%s week=%s", season, week)
        raise PlayerPropsUnavailable(f"unhandled error in the props pipeline: {e!r}") from e


def _full_track_record() -> dict:
    # The season and week come from the schedule, not from the data: `weekly` has to
    # list every elapsed week, and only the calendar knows which of them are elapsed.
    # Without them the store falls back to the newest week it happens to hold, so a
    # week with no tracked picks at the tail of the record would silently drop off.
    season, week = current_season_and_week()
    result = store.get_track_record(current_week=week, season=season)
    result["calibration"] = store.get_calibration()
    return result


def _drop_per_pick_rows(result: dict) -> dict:
    """Strip the row-level pick lists from a track-record payload, in place.

    Only `games.per_pick` and `player_props.per_pick` go; every summary field
    stays exactly what the full payload carries. Those rows are the bulk of
    the payload (the CFB sibling's ran to ~5.8 MB of prop rows no surface
    renders) and are served from `/track-record/picks` for the one reader
    action that needs them -- opening the every-pick fold.
    """
    for section in ("games", "player_props"):
        block = result.get(section)
        if isinstance(block, dict):
            block.pop("per_pick", None)
    return result


@router.get("/track-record")
def get_track_record(picks: str | None = None):
    result = _full_track_record()
    if picks == "deferred":
        # Payload-split contract with the Sports frontend (Sports_Predictor
        # PR #81): the summary without the row-level picks. Any other value --
        # or none -- keeps the historical full payload, so a consumer that
        # predates the split sees no change at all.
        _drop_per_pick_rows(result)
    return result


@router.get("/track-record/picks")
def get_track_record_picks():
    """The row-level game picks the full `/track-record` payload inlines as
    `games.per_pick`, served on their own so the summary can defer them."""
    games = _full_track_record().get("games") or {}
    return {"per_pick": games.get("per_pick") or []}


@router.get("/kalshi-feed")
def get_kalshi_feed():
    """Read-only feed for the Algo Trade Hub: frozen pre-game snapshots for games that have not
    kicked off, plus reliability buckets over graded pre-game snapshots.

    Served from the tracking database only, so it never recomputes a prediction: a live forecast
    is a different number from the snapshot the hub graded, and recomputing would swap the series
    out from under the calibration check with no error anywhere.
    """
    return {
        "sport": "nfl",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lead_hours": SNAPSHOT_LEAD_HOURS,
        "games": store.get_feed_predictions(),
        "calibration": store.get_calibration(),
    }


@router.get("/games/{game_id}/verdict")
def get_game_verdict(game_id: str):
    verdict = store.get_game_verdict(game_id)
    if verdict is None:
        raise HTTPException(status_code=404, detail="Game not tracked or not yet resolved")
    return verdict


@router.get("/predictions/{season}/{week}")
def get_predictions_for_week(season: int, week: int):
    # A union, not an if/else fallback: a week in progress has both
    # already-final games and still-upcoming ones, and the old if/else
    # dropped every finished game (and its verdict) whenever any game in
    # the week was still unplayed (see this plan's final review, finding B3).
    upcoming = schedules.fetch_upcoming_games(season, week)
    finished = schedules.load_training_data(seasons=[season])
    finished = finished[finished["week"] == week]
    games = pd.concat([finished, upcoming], ignore_index=True).drop_duplicates(subset="game_id")
    return store.get_predictions_for_week(season, week, games)


@router.get("/standings")
def get_standings(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and "standings" in snap:
            return snap["standings"]
    return _get_standings_live(season)


def _get_standings_live(season: int):
    schedule = schedules.fetch_schedules([season], force_refresh=(season == CURRENT_SEASON))
    played = schedule[schedule["home_score"].notna() & schedule["away_score"].notna()]
    remaining = schedule[schedule["home_score"].isna()]

    current_records = season_projection.compute_current_records(played)
    team_conferences = teams_data.fetch_team_conferences()

    models = _load_models_cached()
    blocks = tuple(models.get("feature_blocks", []))
    aux = _load_aux_cached(season) if blocks else None
    history = _load_game_history(season)

    def predict_fn(home: str, away: str) -> dict:
        return _predict_game_from_models(models, home, away, history, blocks=blocks, aux=aux, starters=None)

    return season_projection.project_standings(remaining, current_records, team_conferences, predict_fn)


@router.get("/power-rankings")
def get_power_rankings(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and "power_rankings" in snap:
            return snap["power_rankings"]
    return _get_power_rankings_live(season)


def _get_power_rankings_live(season: int) -> dict:
    history = _load_game_history(season)
    ratings = power_ratings.final_ratings(history)

    season_games = history[history["season"] == season]
    played = season_games[season_games["home_score"].notna() & season_games["away_score"].notna()]
    records = season_projection.compute_current_records(played)

    team_conferences = teams_data.fetch_team_conferences()
    division_by_team = {row["team"]: row["division"] for _, row in team_conferences.iterrows()}
    conference_by_team = {row["team"]: row["conference"] for _, row in team_conferences.iterrows()}

    ranked_teams = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    rankings = []
    for rank, (team, rating) in enumerate(ranked_teams, start=1):
        record = records.get(team, {"wins": 0, "losses": 0, "ties": 0})
        rankings.append({
            "team": team,
            "rating": round(float(rating), 1),
            "rank": rank,
            "wins": record["wins"],
            "losses": record["losses"],
            "ties": record["ties"],
            "conference": conference_by_team.get(team),
            "division": division_by_team.get(team),
        })
    return {"season": season, "rankings": rankings}


@router.get("/hub/teams")
def get_hub_teams(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and snap.get("hub_teams"):
            return snap["hub_teams"]
    return _get_hub_teams_live(season)


def _get_hub_teams_live(season: int) -> dict:
    games = _load_game_history(season)
    pbp = team_efficiency_mod.load_pbp(season)
    return {"season": season, "teams": team_efficiency_mod.team_efficiency(pbp, games, season)}


@router.get("/hub/players")
def get_hub_players(season: int = CURRENT_SEASON):
    if PUBLIC_MODE:
        snap = _public_snapshot()
        if snap.get("season") == season and snap.get("hub_players"):
            return snap["hub_players"]
    return _get_hub_players_live(season)


def _get_hub_players_live(season: int) -> dict:
    weekly = player_season_mod.load_hub_weekly(season)
    return {"season": season, **player_season_mod.player_season(weekly, season)}


@router.post("/retrain")
def retrain():
    if PUBLIC_MODE:
        raise HTTPException(status_code=403, detail="Retraining is disabled in public mode")
    result = manifest.train_all()
    _load_models_cached.cache_clear()
    return {"trained_at": result["trained_at"], "chosen_candidate": result["chosen_candidate"]}


def _attach_game_id(stats_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Join weekly player stats to the game_id of the game each row's
    player actually played in, matched by season/week/team -- weekly
    stats carry a team and week but no game_id of their own."""
    if stats_df.empty or games_df.empty:
        return stats_df.iloc[0:0]
    home = games_df[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "recent_team"})
    away = games_df[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "recent_team"})
    team_game = pd.concat([home, away], ignore_index=True)
    return stats_df.merge(team_game, on=["season", "week", "recent_team"], how="inner")


# How close to kickoff a game's prediction is frozen. The first snapshot inside this window is
# kept forever (INSERT OR IGNORE), so it is the one the track record grades and the Kalshi feed
# serves. 48h lands after the prior week's Monday night game for every slot (Thu/Sat/Sun/Mon).
SNAPSHOT_LEAD_HOURS = float(os.getenv("SNAPSHOT_LEAD_HOURS", "48"))


def _games_to_snapshot(season: int, week: int, now: datetime, lead_hours: float | None = None) -> pd.DataFrame:
    """Upcoming games from this week AND next whose kickoff is within the lead window.

    Next week is included because `current_season_and_week()` rolls over on the UTC date of week
    1's first kickoff (a Friday for NFL, a Saturday for CFB). Relying on the current week left
    Thursday-night games unsnapshotted until after their kickoff, at which point
    `record_game_predictions` rejects them -- so they were never tracked at all. It also froze
    next week's games on the previous Saturday, before that day's results.

    The window filters on the UPPER bound only: past games still reach `record_game_predictions`,
    which rejects them itself, and the tick's own test depends on that.
    """
    lead = timedelta(hours=SNAPSHOT_LEAD_HOURS if lead_hours is None else lead_hours)
    frames = [f for f in (schedules.fetch_upcoming_games(season, wk) for wk in (week, week + 1)) if not f.empty]
    if not frames:
        return pd.DataFrame()
    games = pd.concat(frames, ignore_index=True)
    kickoff = pd.to_datetime(games["gameday"], utc=True, errors="coerce")
    horizon = pd.Timestamp(now + lead)
    # A NaT kickoff cannot be placed relative to the window, and snapshotting it would freeze a
    # prediction at an unknown distance from kickoff.
    return games[kickoff.notna() & (kickoff <= horizon)].drop_duplicates("game_id").reset_index(drop=True)


def _passing_td_prop_row(prop: dict, game_id: str) -> dict | None:
    """The `passing_tds` pick row for one QB prop, or None when there is no call.

    `predict_props` has already done the work: a QB whose passing-TD model ran
    carries `passing_td_*` on the same row the yardage markets use, and a QB whose
    did not (no `qb_passing_td` in the artefact, or no mu to project from) carries
    none of them. So this is a shape adapter, not a second prediction -- the line
    recorded here is the line that was served, which is the whole point: a
    re-derived one could differ from what a reader acted on.

    `predicted_value` is the probability of the side called, which is what it means
    for `anytime_td` too, so `_prop_markets` can grade both markets' confidence the
    same way. The line is NOT `predicted_value`; it is an over/under, so it travels
    in its own column beside its `line_source`.

    Subscript, not `.get`, for every field after the `passing_td_line` presence
    check. A payload carrying a line but no side is a broken contract, and this
    raises rather than quietly storing half a call that could not be graded. The
    tick catches it per player, so one malformed payload costs that QB its pick and
    nothing else -- see the comment at the call site.
    """
    if prop["position"] != "QB" or prop.get("passing_td_line") is None:
        return None
    return {
        "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
        "position": prop["position"],
        "market": player_props.qb_passing_td.PASSING_TD_MARKET,
        "predicted_value": float(prop["passing_td_prob"]),
        # Provenance travels with the number, in the row itself. The line is
        # derived from the model's own projection, so it is a model line and NOT a
        # sportsbook price -- there is no book here to have an edge against.
        "line": float(prop["passing_td_line"]),
        "line_source": prop["passing_td_line_source"],
        "side": prop["passing_td_side"],
        "mu": float(prop["passing_td_mu"]),
        "call_prob": float(prop["passing_td_prob"]),
    }


def background_tracking_tick(season: int, week: int) -> None:
    """Snapshot this week's upcoming-game (and player-prop) predictions,
    then reconcile anything now resolved. Called on a timer from
    api/main.py's lifespan the same way PL_Predictor's own
    background_tracking_tick is.

    **This is the only writer of `player_prop_predictions`, and it writes the QB
    passing-TD market alongside the others**, through `_passing_td_prop_row`, with
    the report on the other side in `store._prop_markets`. It is written HERE, in
    the tick, and nowhere else: for a season the market is a prediction a reader can
    make a decision on and cannot be graded, which is worse than not shipping it.
    The prop table's primary key is `(game_id, player_id, market)`, so
    `market="passing_tds"` needs no migration and no weakening of that key."""
    games = _games_to_snapshot(season, week, datetime.now(timezone.utc))
    if not games.empty:
        models = _load_models_cached()
        blocks = tuple(models.get("feature_blocks", []))
        aux = _load_aux_cached(season) if blocks else None
        starters_map = _expected_starters_for_week(season, week) if blocks and "qb" in blocks else {}
        history = _load_game_history(season)
        predictions = []
        for _, game in games.iterrows():
            try:
                starters = {game["home_team"]: starters_map.get(game["home_team"]), game["away_team"]: starters_map.get(game["away_team"])} if starters_map else None
                pred = _predict_game_from_models(
                    models, game["home_team"], game["away_team"], history,
                    spread_line=game.get("spread_line"), total_line=game.get("total_line"),
                    gameday=game.get("gameday"), blocks=blocks, aux=aux,
                    game_schedule=game, starters=starters,
                )
                predictions.append(
                    {
                        "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                        "commence_time": str(game["gameday"]), "season": season,
                        # This tick also snapshots NEXT week's games, so each row must carry its
                        # own week rather than the tick's.
                        "week": int(game["week"]) if pd.notna(game.get("week")) else week,
                        "home_spread_line": game.get("spread_line"), "total_line": game.get("total_line"),
                        **pred,
                    }
                )
            except Exception:
                logger.exception("prediction failed for game_id=%s", game.get("game_id"))
                continue
        store.record_game_predictions(predictions)

        try:
            # THIS week's games only. `_get_player_props_live(season, week)` is this week's prop feed
            # (it can fall back to the current week), while `games` also holds next week's -- so a
            # team playing in both had its prop stored under next week's game_id, and
            # `record_player_prop_predictions` is INSERT OR IGNORE, which would freeze that wrong
            # snapshot forever.
            team_to_game = {}
            for _, g in games[games["week"] == week].iterrows():
                team_to_game[g["home_team"]] = g["game_id"]
                team_to_game[g["away_team"]] = g["game_id"]
            prop_rows = []
            # QBs whose passing-TD call could not be turned into a row. Counted and
            # logged rather than dropped: an absent row is indistinguishable from a
            # QB nobody picked, and this is the one gap in the prop record that no
            # reader of the track record could otherwise see.
            uncalled_qbs: list[str] = []
            for prop in _get_player_props_live(season, week):
                game_id = team_to_game.get(prop["recent_team"])
                if game_id is None:
                    continue
                prop_rows.append({
                    "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                    "position": prop["position"],
                    "market": "anytime_td", "predicted_value": prop["anytime_td_prob"],
                })
                for market in player_props.POSITION_MARKETS.get(prop["position"], []):
                    value = prop.get(market)
                    if value is not None:
                        prop_rows.append({
                            "game_id": game_id, "player_id": prop["player_id"], "player_name": prop["player_name"],
                            "position": prop["position"],
                            "market": market, "predicted_value": value,
                        })
                if prop["position"] != "QB":
                    continue
                try:
                    call_row = _passing_td_prop_row(prop, game_id)
                except Exception:  # noqa: BLE001 - one QB must not cost the whole prop record
                    # A payload with a line and no side, or a side that is not one.
                    # `_passing_td_prop_row` raises on that rather than storing half
                    # a call, and catching it HERE rather than at the block's own
                    # `except` is what keeps the blast radius one QB: without it a
                    # malformed payload would take down the anytime-TD and yardage
                    # rows too, which have nothing to do with this market.
                    logger.exception(
                        "malformed passing-TD payload for player_id=%s in season=%s "
                        "week=%s; this pick is not recorded and the others are",
                        prop.get("player_id"), season, week,
                    )
                    call_row = None
                if call_row is None:
                    uncalled_qbs.append(str(prop["player_id"]))
                    continue
                prop_rows.append(call_row)
            store.record_player_prop_predictions(prop_rows)
            if uncalled_qbs:
                # WARNING, not INFO, and named: the same reasoning as the empty
                # `actual_stats` block lower down in this tick. The likely cause is
                # an artefact directory with no `qb_passing_td` model in it, which
                # makes `predict_props` omit every `passing_td_*` key -- so the
                # market would otherwise record nothing at all and read on the
                # track record as "no picks yet" forever.
                logger.warning(
                    "no passing-TD call for %d QB prop(s) in season=%s week=%s "
                    "(first=%s); their passing_tds picks were NOT recorded, so the "
                    "passing-TD record is short by that many -- is "
                    "models/manifest.json missing qb_passing_td?",
                    len(uncalled_qbs), season, week, uncalled_qbs[:3],
                )
        except Exception:
            logger.exception("player prop snapshot failed for season=%s week=%s", season, week)

    completed = schedules.fetch_current_season_partial()
    store.reconcile_game_predictions(completed[["game_id", "home_score", "away_score"]])

    # Backfill games that finished before this tracker ever ran once
    # (e.g. background_tracking_tick wasn't wired up / wasn't running
    # yet) -- reconcile only ever updates an EXISTING snapshot, so a game
    # with no snapshot at all would otherwise never get a verdict.
    # Excludes each game from its own history so the prediction still
    # reflects strictly pre-game information, not this game's own result.
    try:
        if not completed.empty:
            untracked_ids = store.get_untracked_game_ids(list(completed["game_id"]))
            if untracked_ids:
                models = _load_models_cached()
                blocks = tuple(models.get("feature_blocks", []))
                aux = _load_aux_cached(season) if blocks else None
                starters_map = _expected_starters_for_week(season, week) if blocks and "qb" in blocks else {}
                history_all = _load_game_history(season)
                backfill_games = []
                for _, game in completed[completed["game_id"].isin(untracked_ids)].iterrows():
                    try:
                        history_excl = history_all[history_all["game_id"] != game["game_id"]]
                        starters = {game["home_team"]: starters_map.get(game["home_team"]), game["away_team"]: starters_map.get(game["away_team"])} if starters_map else None
                        pred = _predict_game_from_models(
                            models, game["home_team"], game["away_team"], history_excl,
                            spread_line=game.get("spread_line"), total_line=game.get("total_line"),
                            gameday=game.get("gameday"), blocks=blocks, aux=aux,
                            game_schedule=game, starters=starters,
                        )
                        backfill_games.append({
                            "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                            "commence_time": str(game["gameday"]), "season": season,
                            "week": int(game["week"]) if pd.notna(game.get("week")) else None,
                            "home_spread_line": game.get("spread_line"), "total_line": game.get("total_line"),
                            "actual_home_score": game["home_score"], "actual_away_score": game["away_score"],
                            **pred,
                        })
                    except Exception:
                        logger.exception("backfill prediction failed for game_id=%s", game.get("game_id"))
                        continue
                store.record_resolved_game_predictions(backfill_games)
    except Exception:
        logger.exception("backfill of untracked finished games failed")

    try:
        if not completed.empty:
            actual_stats = player_stats.fetch_weekly_player_stats([season])
            if actual_stats.empty:
                # `hub_cache.cached_frame` catches the upstream 404 and returns an
                # empty frame on purpose, so the site shows dashes instead of
                # erroring, and it logs at INFO -- below the default threshold, and
                # per season per tick, so raising it there would flood. The
                # consequence lands here, so it is reported here: once per tick, at a
                # level that is actually visible.
                #
                # `n_resolved: 0` on the track record with snapshots still being
                # written looks exactly like a tracking bug. It is this. Verified
                # URLs, and why a prop backfill would be the wrong fix:
                # docs/player-prop-accuracy-blocker.md
                #
                # Note the `else`, not a `return`. The game backfill below is a
                # separate `try` block in the same tick, and returning would let a
                # missing *player* stats file silently stop *game* reconciliation --
                # which is the part that works, and would have regressed from 30 of
                # 33 resolved to 0 with no error anywhere.
                logger.warning(
                    "player prop reconciliation skipped: no nflverse player stats for season %s "
                    "(player_stats_%s.parquet is 404 upstream); n_resolved will stay 0 until "
                    "the file is published -- see docs/player-prop-accuracy-blocker.md",
                    season, season,
                )
            else:
                store.reconcile_player_prop_predictions(_attach_game_id(actual_stats, completed))
    except Exception:
        logger.exception("player prop reconciliation failed")

    try:
        store.backfill_unresolved_games(schedules)
    except Exception:
        logger.exception("backfill_unresolved_games failed")


def warm_caches() -> None:
    """Pre-fetch schedules/player stats so the first real request after
    startup isn't slow — best-effort, never raises. Deliberately doesn't
    touch odds_api: NFL's own spread_line/total_line already come straight
    off nfl_data_py's own schedule data (see _predict_game_from_models'
    every caller passing game.get("spread_line")/game.get("total_line")),
    so odds_api.fetch_game_odds() was pure dead weight here -- a real
    network call every warm-up, against a quota shared with other
    projects that genuinely need it, for a value nothing ever read."""
    try:
        schedules.fetch_schedules(schedules.default_completed_seasons(n=8))
        player_stats.fetch_weekly_player_stats(schedules.default_completed_seasons(n=8))
    except Exception:
        pass
