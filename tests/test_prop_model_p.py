"""Regression test for PR 1b: model_p on live ladder rungs.

Covers exactly which keys change and which stay byte-identical, using mocked
inputs so no network / nfl_data_py is required:

  * receptions / passing_tds rungs: `model_p` + `generated_at` added
  * passing_yards / rushing_yards / receiving_yards rungs: byte-identical
    (no model_p, no generated_at — their probability is a quantile
    interpolation the hub does not yet consume)
  * fail-soft: a broken props pipeline omits both fields rather than crashing
    the endpoint
"""
from __future__ import annotations

import pytest


def _one_ladder(ticker, stat, team="KC", initial="P", last="MAHOMES"):
    """One Kalshi ladder with rungs for `stat`, matching current_ladders shape."""
    return {
        "team": team,
        "initial": initial,
        "last": last,
        "stat": stat,
        "rungs": [
            {"line": 250.5, "volume": 1200.0},
            {"line": 275.5, "volume": 900.0},
        ],
    }


def test_receptions_rungs_gain_model_p_and_generated_at(monkeypatch):
    data = {"lines": [_one_ladder("t1", "receptions"), _one_ladder("t2", "passing_tds", team="KC", initial="J", last="BURROW")]}
    props = {
        ("KC", "P", "MAHOMES"): {
            "receptions": (0.42, "2026-10-18T05:00:00+00:00"),
            "passing_tds": (0.61, "2026-10-18T05:00:00+00:00"),
        },
        ("KC", "J", "BURROW"): {"passing_tds": (0.39, "2026-10-18T05:00:00+00:00")},
    }
    from nfl_predictor.api.routes import _attach_prop_model_p

    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [
        {
            "player_name": "Patrick Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.42, "passing_td_over_prob": 0.61,
            "generated_at": "2026-10-18T05:00:00+00:00",
        },
        {
            "player_name": "Joe Burrow", "recent_team": "KC",
            "passing_td_over_prob": 0.39,
            "generated_at": "2026-10-18T05:00:00+00:00",
        },
    ],
    )
    _attach_prop_model_p(data, 2026, 3)

    by_ticker = {l["stat"]: l for l in data["lines"]}
    rec_rungs = by_ticker["receptions"]["rungs"]
    td_rungs = by_ticker["passing_tds"]["rungs"]
    assert all(r["model_p"] == 0.42 for r in rec_rungs)
    assert all(r["generated_at"] == "2026-10-18T05:00:00+00:00" for r in rec_rungs)
    assert all(r["model_p"] == 0.39 for r in td_rungs)
    assert all(r["generated_at"] == "2026-10-18T05:00:00+00:00" for r in td_rungs)


def test_yardage_rungs_byte_identical_no_model_p(monkeypatch):
    data = {
        "lines": [
            _one_ladder("t1", "passing_yards"),
            _one_ladder("t2", "rushing_yards"),
            _one_ladder("t3", "receiving_yards"),
        ]
    }
    from nfl_predictor.api.routes import _attach_prop_model_p

    # Props exist for the player, but yardage markets must still not gain model_p.
    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [{
            "player_name": "Patrick Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.42, "passing_td_over_prob": 0.61,
            "generated_at": "2026-10-18T05:00:00+00:00",
        }],
    )
    _attach_prop_model_p(data, 2026, 3)

    by_stat = {l["stat"]: l for l in data["lines"]}
    for stat in ("passing_yards", "rushing_yards", "receiving_yards"):
        assert "model_p" not in by_stat[stat]
        assert "generated_at" not in by_stat[stat]
        # byte-identical: only team/initial/last/stat/rungs, nothing added
        assert set(by_stat[stat]) == {"team", "initial", "last", "stat", "rungs"}


def test_fail_soft_leaves_rungs_untouched(monkeypatch):
    data = {"lines": [_one_ladder("t1", "receptions")]}
    props = {}
    from nfl_predictor.api.routes import _attach_prop_model_p
    # Simulate a broken props pipeline that returns an unusable row set.
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr("nfl_predictor.api.routes._get_player_props_live", boom)
    _attach_prop_model_p(data, 2026, 3)  # does not raise
    by_stat = {l["stat"]: l for l in data["lines"]}
    assert "model_p" not in by_stat["receptions"]
    assert "generated_at" not in by_stat["receptions"]
    assert set(by_stat["receptions"]) == {"team", "initial", "last", "stat", "rungs"}


def test_selected_player_only_receives_model_p(monkeypatch):
    """One QB with receptions_p_over, another QB without — only the first rung
    carries model_p; the second stays byte-identical."""
    data = {
        "lines": [
            _one_ladder("t1", "receptions", team="KC", initial="P", last="MAHOMES"),
            _one_ladder("t2", "receptions", team="KC", initial="J", last="BURROW"),
        ]
    }
    from nfl_predictor.api.routes import _attach_prop_model_p

    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [
        {
            "player_name": "Patrick Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.55, "generated_at": "2026-10-18T05:00:00+00:00",
        },
    ],
    )
    _attach_prop_model_p(data, 2026, 3)

    lines = data["lines"]
    mahomes = next(l for l in lines if l["last"] == "MAHOMES")
    burrow = next(l for l in lines if l["last"] == "BURROW")
    assert mahomes["rungs"][0]["model_p"] == 0.55
    assert mahomes["rungs"][0]["generated_at"] == "2026-10-18T05:00:00+00:00"
    assert "model_p" not in burrow["rungs"][0]
    assert "generated_at" not in burrow["rungs"][0]


def test_public_mode_reads_model_p_from_the_snapshot_not_the_live_pipeline(monkeypatch):
    """PUBLIC_MODE serves /props from the committed snapshot; the market-lines
    attach must read those same stored rows. Running the live pipeline here
    made every market-lines request pay a full props computation (~17s) even
    when the ladders themselves were cached."""
    from nfl_predictor.api import routes

    data = {"lines": [_one_ladder("t1", "receptions")]}
    snap = {"player_props": [{
        "player_name": "Patrick Mahomes", "recent_team": "KC",
        "receptions_p_over": 0.42, "generated_at": "2026-10-18T05:00:00+00:00",
    }]}
    monkeypatch.setattr(routes, "PUBLIC_MODE", True)
    monkeypatch.setattr(routes, "_snapshot_week", lambda season, week: snap)

    def _live_must_not_run(*a, **k):
        raise AssertionError("the live props pipeline must not run in PUBLIC_MODE")

    monkeypatch.setattr(routes, "_get_player_props_live", _live_must_not_run)
    routes._attach_prop_model_p(data, 2026, 3)

    rung = data["lines"][0]["rungs"][0]
    assert rung["model_p"] == 0.42
    assert rung["generated_at"] == "2026-10-18T05:00:00+00:00"


def test_abbreviated_nflverse_names_attach_model_p(monkeypatch):
    """The rows production actually serves name players "P.Mahomes", not
    "Patrick Mahomes". The matcher must resolve the abbreviated form to the
    same (initial, last) key the Kalshi ladder carries, or model_p never
    attaches -- which is what silently happened before."""
    from nfl_predictor.api import routes

    data = {"lines": [_one_ladder("t1", "receptions")]}
    monkeypatch.setattr(routes, "PUBLIC_MODE", False)
    monkeypatch.setattr(
        "nfl_predictor.api.routes._get_player_props_live",
        lambda *a, **k: [{
            "player_name": "P.Mahomes", "recent_team": "KC",
            "receptions_p_over": 0.42, "generated_at": "2026-10-18T05:00:00+00:00",
        }],
    )
    routes._attach_prop_model_p(data, 2026, 3)

    rung = data["lines"][0]["rungs"][0]
    assert rung["model_p"] == 0.42
    assert rung["generated_at"] == "2026-10-18T05:00:00+00:00"
