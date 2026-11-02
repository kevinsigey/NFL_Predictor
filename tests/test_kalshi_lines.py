import gzip
import json
from datetime import datetime, timezone

from nfl_predictor.data import kalshi_lines as kl

HURTS = {
    "ticker": "KXNFLPASSYDS-26OCT18CARPHI-PHIJHURTS1-325", "event_ticker": "KXNFLPASSYDS-26OCT18CARPHI",
    "title": "Jalen Hurts: 325+ passing yards", "floor_strike": 324.5, "status": "active", "result": "",
    "yes_bid_dollars": "0.0500", "yes_ask_dollars": "0.0600", "no_bid_dollars": "0.9400", "no_ask_dollars": "0.9500",
    "last_price_dollars": "0.0000", "volume_fp": "12.00", "open_interest_fp": "3.00",
    "occurrence_datetime": "2026-10-18T20:00:00Z", "close_time": "2026-10-20T17:00:00Z", "expiration_value": "",
}


def test_normalize_keeps_the_real_threshold_and_prices_as_probabilities():
    row = kl.normalize(HURTS, "KXNFLPASSYDS", "2026-10-15T12:00:00+00:00")
    assert row["player"] == "Jalen Hurts" and row["stat"] == "passing_yards"
    assert row["line"] == 324.5                       # "325+" is the over of 324.5, not of 325
    assert (row["yes_bid"], row["yes_ask"]) == (0.05, 0.06)
    assert row["result"] is None and row["status"] == "active"


def test_a_settled_market_carries_its_outcome():
    row = kl.normalize({**HURTS, "status": "settled", "result": "yes", "expiration_value": "331"}, "KXNFLPASSYDS", "t")
    assert row["result"] == "yes" and row["settlement_value"] == "331"


def test_missing_prices_are_none_not_zero():
    row = kl.normalize({"title": "A B: 5+ receptions", "floor_strike": 4.5}, "KXNFLREC", "t")
    assert row["yes_bid"] is None and row["yes_ask"] is None and row["player"] == "A B"


class _Resp:
    def __init__(self, body, status=200):
        self._b, self.status_code = body, status

    def json(self):
        return self._b

    def raise_for_status(self):
        assert self.status_code < 400


class _Session:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        if params.get("cursor") == "p2":
            return _Resp({"markets": [HURTS], "cursor": ""})
        return _Resp({"markets": [HURTS], "cursor": "p2" if params["series_ticker"] == "KXNFLPASSYDS" and params["status"] == "open" else ""})


def test_collect_pages_through_cursors_and_asks_for_open_and_recently_settled():
    s = _Session()
    rows = kl.collect(s, now=datetime(2026, 10, 15, 12, tzinfo=timezone.utc))
    statuses = {(c["series_ticker"], c["status"]) for c in s.calls}
    assert ("KXNFLPASSYDS", "open") in statuses and ("KXNFLREC", "settled") in statuses
    assert any(c.get("cursor") == "p2" for c in s.calls)
    assert len(rows) == 6 * 2 + 1                       # 6 series x (open, settled) + the extra page
    assert all(r["snapshot_ts"].startswith("2026-10-15") for r in rows)


def test_write_snapshot_appends_one_gzip_file_per_utc_day(tmp_path):
    rows = [kl.normalize(HURTS, "KXNFLPASSYDS", "t")]
    now = datetime(2026, 10, 15, 12, tzinfo=timezone.utc)
    p1 = kl.write_snapshot(rows, tmp_path, now)
    p2 = kl.write_snapshot(rows, tmp_path, now)
    assert p1 == p2 and p1.name == "kalshi_nfl_lines_2026-10-15.jsonl.gz"
    lines = gzip.open(p1, "rt").read().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["line"] == 324.5
    assert kl.write_snapshot([], tmp_path, now) is None


def test_player_tokens_resolve_team_initial_and_last_name_even_with_glued_event_teams():
    f = kl.parse_player_token
    assert f("KXNFLPASSYDS-26OCT18CARPHI-PHIJHURTS1-325", "KXNFLPASSYDS-26OCT18CARPHI") == {"team": "PHI", "initial": "J", "last": "HURTS"}
    assert f("KXNFLRSHYDS-26OCT18LACKC-LACKVIDAL28-60", "KXNFLRSHYDS-26OCT18LACKC") == {"team": "LAC", "initial": "K", "last": "VIDAL"}
    assert f("KXNFLRECYDS-26OCT18LACKC-KCXWORTHY1-90", "KXNFLRECYDS-26OCT18LACKC") == {"team": "KC", "initial": "X", "last": "WORTHY"}
    assert f("KXNFLREC-26OCT18HOUJAC-JACPWASHINGTON11-9", "KXNFLREC-26OCT18HOUJAC") == {"team": "JAX", "initial": "P", "last": "WASHINGTON"}   # JAC -> JAX
    assert f("garbage", "garbage") is None


def test_implied_probability_is_the_midpoint_else_last_trade_else_none():
    assert kl.implied_probability({"yes_bid": 0.40, "yes_ask": 0.44, "last": 0.9}) == 0.42
    assert kl.implied_probability({"yes_bid": None, "yes_ask": None, "last": 0.31}) == 0.31
    assert kl.implied_probability({"yes_bid": None, "yes_ask": None, "last": 0.0}) is None


def test_build_ladders_groups_open_rungs_by_player_and_stat_sorted_by_line_and_skips_unquoted():
    def rung(n, bid, ask, status="active"):
        return {**kl.normalize({**HURTS, "ticker": f"KXNFLPASSYDS-26OCT18CARPHI-PHIJHURTS1-{n}", "floor_strike": n - 0.5,
                                "yes_bid_dollars": bid, "yes_ask_dollars": ask, "status": status}, "KXNFLPASSYDS", "t")}
    rows = [rung(300, "0.20", "0.24"), rung(250, "0.60", "0.64"), rung(350, None, None), rung(275, "0.4", "0.44", status="settled")]
    (ladder,) = kl.build_ladders(rows)
    assert (ladder["team"], ladder["initial"], ladder["last"], ladder["stat"]) == ("PHI", "J", "HURTS", "passing_yards")
    assert [r["line"] for r in ladder["rungs"]] == [249.5, 299.5] and ladder["rungs"][0]["p_over"] == 0.62


def test_current_ladders_caches_and_fails_soft(monkeypatch):
    calls = []

    def boom(session, series, status, min_close_ts=None, max_close_ts=None):
        calls.append(series)
        raise RuntimeError("kalshi down")

    monkeypatch.setattr(kl, "fetch_series", boom)
    monkeypatch.setattr(kl, "_ladder_cache", None)
    assert kl.current_ladders(now=1000.0) == {"fetched_at": None, "lines": []}       # no cache + failure -> empty, no raise
    good = {"fetched_at": "x", "lines": [{"team": "PHI"}]}
    monkeypatch.setattr(kl, "_ladder_cache", (1000.0, good))
    assert kl.current_ladders(now=1100.0) is good                                      # within TTL: served from cache
    n = len(calls)
    assert kl.current_ladders(now=1000.0 + kl.LADDER_TTL_SECONDS + 1) is good          # expired + failure -> last good value
    assert len(calls) > n


def test_the_serving_fetch_is_bounded_to_the_current_weeks_close_window(monkeypatch):
    # The line checker's fetch must not page through every open market Kalshi
    # lists: it asks only for markets closing inside the current week's
    # window. (The collector's own scan stays unbounded -- it wants history.)
    seen = []

    def fake_fetch(session, series, status, min_close_ts=None, max_close_ts=None):
        seen.append((series, status, max_close_ts))
        return []

    monkeypatch.setattr(kl, "fetch_series", fake_fetch)
    out = kl._fetch_ladders(object(), now=1000.0)
    assert out["lines"] == []
    assert seen and all(status == "open" for _, status, _ in seen)
    assert all(max_close == int(1000.0 + kl.LADDER_CLOSE_WINDOW_DAYS * 86400) for _, _, max_close in seen)


def test_current_ladders_warms_from_the_disk_cache(monkeypatch, tmp_path):
    # A cold process (no memory cache) holding a fresh on-disk payload serves
    # it without touching Kalshi. This app scales to zero between requests,
    # so the disk copy under CACHE_DIR is the cache most requests meet.
    import json
    from datetime import datetime, timezone

    payload = {
        "fetched_at": datetime.fromtimestamp(1000.0, timezone.utc).isoformat(timespec="seconds"),
        "lines": [{"team": "PHI"}],
    }
    path = tmp_path / "kalshi_ladders.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(kl, "LADDER_CACHE_PATH", path)
    monkeypatch.setattr(kl, "_ladder_cache", None)

    def boom(*a, **k):
        raise AssertionError("must not fetch: the disk cache is fresh")

    monkeypatch.setattr(kl, "fetch_series", boom)
    assert kl.current_ladders(now=1100.0) == payload


def test_market_lines_route_returns_the_cached_ladders(monkeypatch):
    from nfl_predictor.api import routes

    payload = {"fetched_at": "2026-10-15T00:00:00+00:00", "lines": [{"team": "PHI", "initial": "J", "last": "HURTS", "stat": "passing_yards", "rungs": []}]}
    monkeypatch.setattr(kl, "current_ladders", lambda *a, **k: payload)
    # The route also tries to attach prop probabilities, which runs the live props pipeline (nfl_data_py -> github.com,
    # habitatring.com). This test is about the cached ladders, so that pipeline is stubbed, not the socket.
    monkeypatch.setattr(routes, "_get_player_props_live", lambda *a, **k: [])
    assert routes.get_market_lines(2026, 6) == payload
