"""The starting quarterback as a feature: his career EPA per dropback (shrunk), experience, and whether he changed.

Training uses the QB who actually started (the passer with the most dropbacks). Serving has no result, so it uses the
EXPECTED starter: the depth chart's first QB who is not reported Out. That skew is real and is measured by
`tools/qb_starter_agreement.py`, not assumed away. A QB's own past games are the only data his rating reads.
"""
from __future__ import annotations

import pandas as pd

SHRINK_K = 200.0        # dropbacks of prior worth: a QB with 200 dropbacks is half his own record, half the prior
PRIOR_EPA_PER_DROPBACK = 0.0
NEW_QB_GAMES = 3        # fewer career games than this in the data = a new / unproven starter
RELOCATED = {"LV": "OAK", "LAC": "SD", "LA": "STL"}  # play-by-play code -> pre-move schedule code
QB_COLUMNS = [f"{side}_{c}" for c in ("qb_epa_pd", "qb_games", "qb_changed", "qb_new") for side in ("home", "away")]


def _history(qb_games_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    dates = pd.concat([
        games_df[["game_id", "gameday"]],
    ]).drop_duplicates("game_id")
    h = qb_games_df.merge(dates, on="game_id", how="left")
    h["gameday"] = pd.to_datetime(h["gameday"])
    # Play-by-play names relocated clubs by their CURRENT code (LV/LAC/LA) while schedules keep the historical one
    # (OAK/SD/STL), so a per-team lookup missed every pre-move game. Rename to the schedule's code.
    game_teams = games_df.set_index("game_id")[["home_team", "away_team"]]
    sides = game_teams.reindex(h["game_id"])
    old = h["team"].map(RELOCATED).to_numpy()
    use_old = (sides["home_team"].to_numpy() == old) | (sides["away_team"].to_numpy() == old)
    h["team"] = h["team"].where(~use_old, old)
    return h.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def add_qb_features(
    games_df: pd.DataFrame, qb_games_df: pd.DataFrame,
    upcoming_starters: dict[tuple[str, str], str | None] | None = None,
    k: float = SHRINK_K, prior: float = PRIOR_EPA_PER_DROPBACK, new_games: int = NEW_QB_GAMES,
) -> pd.DataFrame:
    """Add `{home,away}_qb_epa_pd / qb_games / qb_changed / qb_new` for every game.

    `qb_games_df` holds one row per (game, team, qb) — every passer, not just the starter.
    The starter for a completed game is the passer with the most dropbacks (lowest id on a tie);
    his rating and experience accumulate ALL his prior dropbacks, including relief appearances
    in games he did not start. A game with no QB row (upcoming) uses `upcoming_starters[(game_id, team)]`,
    and a game with neither gets the prior, zero games and "new": the honest "we do not know who starts".
    """
    upcoming_starters = upcoming_starters or {}
    hist = _history(qb_games_df, games_df)
    hist["cum_db"] = hist.groupby("qb_id")["dropbacks"].cumsum() - hist["dropbacks"]
    hist["cum_epa"] = hist.groupby("qb_id")["epa_sum"].cumsum() - hist["epa_sum"]
    hist["cum_games"] = hist.groupby("qb_id").cumcount()
    # Starter per (game, team): most dropbacks, lowest id on a tie. Pick within each
    # game-team group; order the starters chronologically for prev_qb.
    ranked = hist.sort_values(["game_id", "team", "dropbacks", "qb_id"], ascending=[True, True, False, True])
    starter_rows = ranked.drop_duplicates(["game_id", "team"], keep="first").sort_values(["gameday", "game_id"])
    starter_rows = starter_rows.copy()
    starter_rows["prev_qb"] = starter_rows.groupby("team")["qb_id"].shift(1)
    by_starter = starter_rows.set_index(["game_id", "team"])
    # What each QB has accumulated through his LAST game: the starting point for a game that has not been played.
    totals = hist.groupby("qb_id").agg(db=("dropbacks", "sum"), epa=("epa_sum", "sum"), games=("dropbacks", "size"))
    last_qb_of_team = starter_rows.groupby("team")["qb_id"].last()

    out = games_df.copy()
    for side in ("home", "away"):
        epa_pd, n_games, changed, new = [], [], [], []
        for game_id, team in zip(out["game_id"], out[f"{side}_team"]):
            key = (game_id, team)
            if key in by_starter.index:
                r = by_starter.loc[key]
                epa_pd.append((r["cum_epa"] + k * prior) / (r["cum_db"] + k))
                n_games.append(float(r["cum_games"]))
                changed.append(float(isinstance(r["prev_qb"], str) and r["prev_qb"] != r["qb_id"]))
            else:
                qb = upcoming_starters.get(key)
                if qb is not None and qb in totals.index:
                    t = totals.loc[qb]
                    epa_pd.append((t["epa"] + k * prior) / (t["db"] + k))
                    n_games.append(float(t["games"]))
                    changed.append(float(team in last_qb_of_team.index and last_qb_of_team[team] != qb))
                else:
                    epa_pd.append(prior)
                    n_games.append(0.0)
                    changed.append(0.0)
            new.append(float(n_games[-1] < new_games))
        out[f"{side}_qb_epa_pd"], out[f"{side}_qb_games"] = epa_pd, n_games
        out[f"{side}_qb_changed"], out[f"{side}_qb_new"] = changed, new
    return out


def add_pregame_qb_features(
    games_df: pd.DataFrame, qb_games_df: pd.DataFrame, starters: dict[tuple[str, str], str | None] | None = None,
    k: float = SHRINK_K, prior: float = PRIOR_EPA_PER_DROPBACK, new_games: int = NEW_QB_GAMES,
) -> pd.DataFrame:
    """The same four QB_COLUMNS, but every input is knowable before kickoff, for completed games too.

    Starter = `starters[(game_id, team)]` (the depth-chart / injury-report expected starter), else the team's starter
    in its previous game. Rating, games and "changed" read only games dated strictly before this one, so a historical
    row sees exactly what serving would have seen. `add_qb_features` instead names the ACTUAL starter for a played
    game, which a pre-game model cannot know.
    """
    starters = starters or {}
    hist = _history(qb_games_df, games_df).dropna(subset=["gameday"])
    actual = (hist.sort_values(["game_id", "team", "dropbacks", "qb_id"], ascending=[True, True, False, True])
              .drop_duplicates(["game_id", "team"])[["team", "gameday", "qb_id"]]
              .rename(columns={"qb_id": "prev_qb"}).sort_values("gameday"))
    cum = (hist.groupby(["qb_id", "gameday"]).agg(db=("dropbacks", "sum"), epa=("epa_sum", "sum"), n=("dropbacks", "size"))
           .groupby(level=0).cumsum().reset_index().sort_values("gameday"))
    out = games_df.copy()
    out["_gd"] = pd.to_datetime(out["gameday"])
    for side in ("home", "away"):
        keys = out[["game_id", f"{side}_team", "_gd"]].rename(columns={f"{side}_team": "team", "_gd": "gameday"})
        keys["_row"] = range(len(keys))
        keys = pd.merge_asof(keys.sort_values("gameday"), actual, on="gameday", by="team", allow_exact_matches=False)
        keys["qb_id"] = [starters.get((g, t)) or p for g, t, p in zip(keys["game_id"], keys["team"], keys["prev_qb"])]
        keys["qb_id"] = keys["qb_id"].where(keys["qb_id"].map(lambda q: isinstance(q, str)), "__none__")
        keys = pd.merge_asof(keys, cum, on="gameday", by="qb_id", allow_exact_matches=False).sort_values("_row")
        db, epa, n = (keys[c].fillna(0.0).to_numpy() for c in ("db", "epa", "n"))
        known = (keys["qb_id"] != "__none__").to_numpy()
        out[f"{side}_qb_epa_pd"] = (epa + k * prior) / (db + k)
        out[f"{side}_qb_games"] = n
        out[f"{side}_qb_changed"] = (known & keys["prev_qb"].notna().to_numpy()
                                     & (keys["qb_id"] != keys["prev_qb"]).to_numpy()).astype(float)
        out[f"{side}_qb_new"] = (n < new_games).astype(float)
    return out.drop(columns="_gd")


def expected_starters(depth_chart: pd.DataFrame, out_ids: set[str] | None = None) -> dict[str, str | None]:
    """{team: gsis_id} of each team's expected starting QB: the first QB on its depth chart who is not reported Out.

    `depth_chart` is the nflverse chart for the week (columns club_code, position, depth_team (a STRING '1','2',...),
    gsis_id). A team with no usable QB maps to None, never to a guess.
    """
    out_ids = set(out_ids or ())
    if depth_chart is None or depth_chart.empty:
        return {}
    qbs = depth_chart[depth_chart["position"] == "QB"].copy()
    qbs["slot"] = pd.to_numeric(qbs["depth_team"], errors="coerce")
    result: dict[str, str | None] = {}
    for team, grp in qbs.groupby("club_code"):
        ordered = grp.sort_values("slot")
        pick = next((g for g in ordered["gsis_id"] if isinstance(g, str) and g not in out_ids), None)
        result[team] = pick
    return result