"""Gate the pre-game starting-QB block (`qb_pregame`) against the served model, with the closing line beside it.

Served model = ridge margin + XGBoost total on the ten base columns. Same repo folds both sides (train on every
earlier season, score one season), so held-out games are identical. Starter per (game, team) is the depth chart's
first QB not reported Out on that week's injury report; with no chart, the team's previous-game starter.

Pre-registered before the run: primary = margin MAE, ships only if its paired-bootstrap 95% CI is above zero
(base - block) AND total MAE and Brier are not worse. Range 2000-2025. No tuning on these folds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from nfl_predictor.config import DEPTH_CHARTS_CACHE_DIR
from nfl_predictor.data import depth_charts as dc
from nfl_predictor.data import injuries
from nfl_predictor.evaluate import walk_forward as wf
from nfl_predictor.features import build as feature_build
from nfl_predictor.features import qb
from nfl_predictor.tools.block_eval import load_recent_nfl_data, paired_bootstrap


def _out_ids_by_week(seasons) -> dict:
    seasons = [int(s) for s in seasons if s >= 2009]  # nflverse injury reports start in 2009
    if not seasons:
        return {}
    inj = injuries.fetch_injuries(seasons)
    inj = inj[inj["report_status"] == "Out"]
    return inj.groupby(["season", "week"])["gsis_id"].agg(set).to_dict()


def pregame_starters(games_df: pd.DataFrame, cache_dir=DEPTH_CHARTS_CACHE_DIR) -> dict:
    """(game_id, team) -> expected starting QB, using only what was published before kickoff."""
    out_ids = _out_ids_by_week(games_df["season"].unique())
    result: dict = {}
    for season, sg in games_df.groupby("season"):
        season = int(season)
        if dc.load_depth_charts(season, cache_dir) is None:
            continue
        raw = pd.read_parquet(dc._cache_path(cache_dir, season))
        snapshots = "dt" in raw.columns
        if snapshots:  # 2025+ feed: many timestamped snapshots; take the latest one before the game day
            raw = raw[raw["pos_abb"] == "QB"].rename(columns={"team": "club_code", "pos_rank": "depth_team"})
            raw = raw.assign(position="QB", _stamp=pd.to_datetime(raw["dt"], utc=True).dt.tz_localize(None))
        else:
            raw = raw[raw["position"] == "QB"].assign(_wk=lambda f: pd.to_numeric(f["week"], errors="coerce"))
        group_key = "gameday" if snapshots else "week"
        for key, gg in sg.groupby(group_key):
            if snapshots:
                snap = raw[raw["_stamp"] < pd.Timestamp(key)]
                chart = snap[snap["_stamp"] == snap["_stamp"].max()]
            else:
                snap = raw[raw["_wk"] <= key]
                chart = snap[snap["_wk"] == snap["_wk"].max()]
            if chart.empty:
                continue
            outs = set().union(*(out_ids.get((season, w), set()) for w in gg["week"].unique()))
            expected = qb.expected_starters(chart, outs)
            for g in gg.itertuples():
                for team in (g.home_team, g.away_team):
                    if expected.get(team):
                        result[(g.game_id, team)] = expected[team]
    return result


def run(games_df, aux) -> pd.DataFrame:
    """One row per held-out game: actuals, closing lines, base and block predictions (raw, uncorrected)."""
    base = wf.prepare_folds(games_df)
    block = wf.prepare_folds(games_df, blocks=("qb_pregame",), aux=aux)
    rows = []
    for fb, fq in zip(base, block, strict=True):
        assert fb["val_season"] == fq["val_season"]
        assert fb["val_df"]["game_id"].tolist() == fq["val_df"]["game_id"].tolist(), "folds diverge"
        out = fb["val_df"][["game_id", "season", "margin", "total_points", "spread_line", "total_line"]].copy()
        for tag, f in (("base", fb), ("qb", fq)):
            m = wf._fit_predict("ridge", f["train_df"], f["val_df"], f["feature_cols"], "margin")
            sigma = wf._honest_sigma("ridge", f["train_df"], f["feature_cols"], "margin")
            out[f"margin_{tag}"] = m
            out[f"p_home_{tag}"] = [wf.game_outcome.margin_to_probabilities(x, sigma)["home_win_prob"] for x in m]
            out[f"total_{tag}"] = wf._fit_predict("xgb", f["train_df"], f["val_df"], f["feature_cols"], "total_points")
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def report(df: pd.DataFrame) -> dict:
    y_win = (df["margin"] > 0).astype(float)
    err = {
        "margin": {t: (df[f"margin_{t}"] - df["margin"]).abs() for t in ("base", "qb")},
        "total": {t: (df[f"total_{t}"] - df["total_points"]).abs() for t in ("base", "qb")},
        "brier": {t: (df[f"p_home_{t}"] - y_win) ** 2 for t in ("base", "qb")},
    }
    line = {"margin": (df["spread_line"] - df["margin"]).abs().mean(),
            "total": (df["total_line"] - df["total_points"]).abs().mean()}
    res = {"n": len(df)}
    for k, e in err.items():
        b, q = e["base"].mean(), e["qb"].mean()
        res[k] = {"base": b, "qb": q, "delta": b - q, "ci": paired_bootstrap(e["base"], e["qb"])}
        if k in line:
            res[k]["line"] = line[k]
            res[k]["gap_closed"] = (b - q) / (b - line[k]) if b != line[k] else float("nan")
    res["passes"] = bool(res["margin"]["ci"][0] > 0 and res["total"]["delta"] >= 0 and res["brier"]["delta"] >= 0)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start-year", type=int, default=2000)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--output", default="output/qb_pregame_eval_games.csv.gz")
    args = ap.parse_args()

    games_df, aux = load_recent_nfl_data(args.start_year, args.end_year)
    starters = pregame_starters(games_df)
    aux = feature_build.Aux(aux.efficiency, aux.qb_games, starters)
    qg = aux.qb_games.copy()
    old = qg["team"].map(qb.RELOCATED)
    qg["team"] = [o if isinstance(o, str) and f"_{o}" in g else t for t, o, g in zip(qg["team"], old, qg["game_id"])]
    actual = (qg.sort_values(["game_id", "team", "dropbacks", "qb_id"], ascending=[True, True, False, True])
              .drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"])["qb_id"])
    named = pd.Series(starters)
    common = named.index.intersection(actual.index)
    print(f"team-game slots: {2 * len(games_df)}; chart/injury starter named: {len(named)}; "
          f"agrees with actual starter: {(named[common] == actual[common]).mean():.3f} (n={len(common)})")

    df = run(games_df, aux)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    r = report(df)
    print(f"\nheld-out games: {r['n']}  (positive delta = qb_pregame better; CI = paired bootstrap 95%)")
    for k in ("margin", "total"):
        m = r[k]
        print(f"{k:<7} MAE base {m['base']:.4f}  qb {m['qb']:.4f}  delta {m['delta']:+.4f} "
              f"[{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]  | closing line {m['line']:.4f}  gap closed {m['gap_closed']:.1%}")
    b = r["brier"]
    print(f"brier   base {b['base']:.5f}  qb {b['qb']:.5f}  delta {b['delta']:+.5f} [{b['ci'][0]:+.5f}, {b['ci'][1]:+.5f}]")
    print(f"\nGATE: {'PASS' if r['passes'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
