"""/games carries the Kalshi game-winner price as `market_home_win_probability`: the latest snapshot taken BEFORE kickoff,
matched by ET game date + both teams, None when there is none (never defaulted)."""
import gzip
import json

import pandas as pd

from nfl_predictor.api import routes
from nfl_predictor.data import kalshi_lines as kl, schedules

# PHI @ JAX, 13:00 ET Sun 2026-10-11 = 17:00Z (EDT)
KICK = "2026-10-11T17:00:00+00:00"


def _row(ts, event, side, bid, ask):
    return {"series": "KXNFLGAME", "snapshot_ts": ts, "ticker": f"{event}-{side}", "event_ticker": event,
            "yes_bid": bid, "yes_ask": ask, "last": None, "status": "active"}


def _write(tmp_path, rows):
    with gzip.open(tmp_path / "kalshi_nfl_lines_2026-10-11.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


def _games(monkeypatch, tmp_path):
    raw = pd.DataFrame([{"game_id": "2026_05_PHI_JAX", "season": 2026, "week": 5, "gameday": "2026-10-11", "gametime": "13:00",
                         "home_team": "JAX", "away_team": "PHI", "home_score": None, "away_score": None, "home_rest": 7,
                         "away_rest": 7, "div_game": 0, "roof": "outdoors", "surface": "grass", "temp": None, "wind": None,
                         "spread_line": 7.5, "total_line": 41.5, "stadium": "S"}])
    monkeypatch.setattr(schedules, "SCHEDULES_CACHE_DIR", tmp_path)
    monkeypatch.setattr(schedules, "_import_schedules", lambda years: raw)
    monkeypatch.setattr(routes, "DATA_DIR", tmp_path)  # lines live in <DATA_DIR>/lines
    monkeypatch.setattr(routes._forecast_cache, "_fetch", lambda *a: None)
    monkeypatch.setattr(routes._forecast_cache, "_store", {})
    return lambda: routes._get_games_live(2026, 5)[0]["market_home_win_probability"]


def _lines(tmp_path):
    d = tmp_path / "lines"
    d.mkdir(exist_ok=True)
    return d


def test_latest_pre_kickoff_price_is_used_and_post_kickoff_ignored(monkeypatch, tmp_path):
    get = _games(monkeypatch, tmp_path)
    d = _lines(tmp_path)
    ev = "KXNFLGAME-26OCT11PHIJAC"  # Kalshi says JAC for JAX
    _write(d, [
        _row("2026-10-10T12:00:00+00:00", ev, "JAC", 0.28, 0.32), _row("2026-10-10T12:00:00+00:00", ev, "PHI", 0.68, 0.72),
        _row("2026-10-11T16:00:00+00:00", ev, "JAC", 0.38, 0.42), _row("2026-10-11T16:00:00+00:00", ev, "PHI", 0.58, 0.62),
        _row("2026-10-11T18:00:00+00:00", ev, "JAC", 0.98, 1.0), _row("2026-10-11T18:00:00+00:00", ev, "PHI", 0.0, 0.02),
    ])
    assert get() == 0.4   # 0.40 / (0.40 + 0.60): the 16:00Z snapshot, normalised; the post-kickoff 18:00Z one is ignored


def test_only_post_kickoff_prices_or_no_rows_is_none(monkeypatch, tmp_path):
    get = _games(monkeypatch, tmp_path)
    d = _lines(tmp_path)
    assert get() is None  # no lines at all
    ev = "KXNFLGAME-26OCT11PHIJAC"
    _write(d, [_row("2026-10-11T18:00:00+00:00", ev, "JAC", 0.9, 0.92), _row("2026-10-11T18:00:00+00:00", ev, "PHI", 0.08, 0.1)])
    kl._moneyline_snapshots.cache_clear()
    assert get() is None


def test_one_priced_side_uses_its_price_and_a_different_game_never_matches(monkeypatch, tmp_path):
    get = _games(monkeypatch, tmp_path)
    d = _lines(tmp_path)
    _write(d, [_row("2026-10-10T12:00:00+00:00", "KXNFLGAME-26OCT11PHIJAC", "PHI", 0.68, 0.72),
               _row("2026-10-10T12:00:00+00:00", "KXNFLGAME-26OCT18PHIJAC", "JAC", 0.9, 0.9)])  # same teams, other week
    kl._moneyline_snapshots.cache_clear()
    assert get() == 0.3


def test_a_wide_quote_is_no_price(monkeypatch, tmp_path):
    get = _games(monkeypatch, tmp_path)
    ev = "KXNFLGAME-26OCT11PHIJAC"
    _write(_lines(tmp_path), [_row("2026-10-10T12:00:00+00:00", ev, "JAC", 0.07, 0.79), _row("2026-10-10T12:00:00+00:00", ev, "PHI", 0.1, 0.8)])
    assert get() is None
