"""Player-prop grading: the delta, the grade bands, the section ranking.

The grading judges the COUNTED pick (the earliest recorded projection per
game, player and market -- the pick the track record counts, selected by the
store's own reader) against the recorded actual, in units of the model's own
published holdout error for the stat: the walk-forward MAE the yardage
artifacts carry in `models/manifest.json`. These tests pin the arithmetic
and the honesty rules: no published MAE means no grade, a baseline comes
only from games recorded BEFORE the graded one in the same season, a pick
that cannot prove it was made before kickoff is graded and labelled, never
dropped and never passed off as pre-kickoff, and a store with no props in it
grades empty rather than crashing.
"""
import contextlib
import json

import pytest

from nfl_predictor.tracking import prop_grading, store


@pytest.fixture(autouse=True)
def _isolated_db(monkeypatch, tmp_path):
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    yield


def _insert_game(*, game_id, commence_time, home="GB", away="CAR", season=2026):
    """An unresolved game row carrying the kickoff and the two teams. It is
    `resolved = 0` deliberately: this fixture is about the prop record, and
    a resolved game row would also land in the games headline."""
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "INSERT OR IGNORE INTO game_predictions (game_id, home_team, away_team, "
            "commence_time, snapshotted_at, home_win_prob, away_win_prob, resolved, season) "
            "VALUES (?, ?, ?, ?, ?, 0.5, 0.5, 0, ?)",
            (game_id, home, away, commence_time, commence_time, season),
        )


def _insert_prop(*, game_id, player_id, market, predicted_value, actual_value,
                 snapshotted_at, player_name="Player", position="QB"):
    """One resolved prop row, with the timestamps written exactly as given --
    the rule under test is entirely about which timestamps a pick carries,
    and the public writers stamp their own."""
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            """
            INSERT INTO player_prop_predictions
                (game_id, player_id, player_name, market, predicted_value, snapshotted_at,
                 resolved, actual_value, position)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (game_id, player_id, player_name, market, predicted_value, snapshotted_at,
             actual_value, position),
        )


def _stats():
    """The real bands, read from the repo's own manifest the way the route
    reads them -- so the build tests grade against the published MAEs
    whatever a retrain moves them to."""
    from nfl_predictor.config import MODELS_DIR

    return prop_grading.prop_bands(MODELS_DIR)


def _no_stats():
    return {stat: {"holdout_mae": None, "naive_mae": None}
            for stat in prop_grading.GRADED_STATS}


def _day(payload, date):
    return next(d for d in payload["days"] if d["date"] == date)


def _rows(payload, date):
    return [row for game in _day(payload, date)["games"] for row in game["rows"]]


# ---------------------------------------------------------------------------
# The grade bands
# ---------------------------------------------------------------------------


def test_grade_bands_are_multiples_of_the_stats_own_mae():
    from nfl_predictor.tracking.prop_grading import grade_for

    # MAE 100: half is 50, the error itself 100, twice it 200.
    assert grade_for(0.0, 100.0) == "On the number"
    assert grade_for(50.0, 100.0) == "On the number"   # exactly half: inside the band
    assert grade_for(50.1, 100.0) == "Close"
    assert grade_for(100.0, 100.0) == "Close"          # exactly the MAE: inside the band
    assert grade_for(100.1, 100.0) == "Off"
    assert grade_for(200.0, 100.0) == "Off"            # exactly twice: inside the band
    assert grade_for(200.1, 100.0) == "Way off"


def test_grade_is_none_without_a_measured_error():
    from nfl_predictor.tracking.prop_grading import grade_for

    assert grade_for(3.0, None) is None
    assert grade_for(3.0, 0.0) is None


def test_prop_bands_reads_the_manifests_walkforward_mae(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "quantile_yardage_v1": {"markets": {
            "passing_yards": {"walkforward_mae": 70.5},
            "rushing_yards": {"walkforward_mae": None},
            "receiving_yards": {"walkforward_mae": 20.9},
        }},
    }))
    bands = prop_grading.prop_bands(tmp_path)
    assert bands["passing_yards"] == {"holdout_mae": 70.5, "naive_mae": None}
    assert bands["rushing_yards"] == {"holdout_mae": None, "naive_mae": None}
    assert bands["receiving_yards"]["holdout_mae"] == 20.9
    # A missing manifest publishes no bands at all -- and no defaults.
    assert prop_grading.prop_bands(tmp_path / "missing") == _no_stats()


def test_prop_bands_match_the_repo_manifests_own_figures():
    """The wiring test: the bands the route serves are the figures the
    committed manifest itself carries, read back out of the artifact rather
    than restated here (a retrain moves both together)."""
    from nfl_predictor.config import MODELS_DIR

    manifest = json.loads((MODELS_DIR / "manifest.json").read_text())
    markets = manifest["quantile_yardage_v1"]["markets"]
    bands = _stats()
    for stat in prop_grading.GRADED_STATS:
        assert bands[stat]["holdout_mae"] == markets[stat]["walkforward_mae"]
        assert bands[stat]["naive_mae"] is None


# ---------------------------------------------------------------------------
# The counted pick and the kickoff label
# ---------------------------------------------------------------------------


def test_the_counted_pick_is_the_earliest_recorded():
    """A rerun must not displace the counted pick: the writer is INSERT OR
    IGNORE, so the first value recorded stands, and the grading grades that
    one -- never the later, better-informed rerun."""
    _insert_game(game_id="2026_09_CAR_GB", commence_time="2026-09-27T17:00:00+00:00")
    prop = {"game_id": "2026_09_CAR_GB", "player_id": "qb1", "player_name": "Q B",
            "position": "QB", "market": "passing_yards", "predicted_value": 300.0}
    assert store.record_player_prop_predictions([prop]) == 1
    assert store.record_player_prop_predictions(
        [{**prop, "predicted_value": 999.0}]) == 0
    with contextlib.closing(store._connect()) as conn, conn:
        conn.execute(
            "UPDATE player_prop_predictions SET resolved = 1, actual_value = 310.0 "
            "WHERE game_id = '2026_09_CAR_GB'")

    rows = _rows(prop_grading.build_prop_grading(_stats()), "2026-09-27")
    assert len(rows) == 1
    assert rows[0]["projected"] == 300.0
    assert rows[0]["delta"] == 10.0


def test_a_late_pick_is_graded_and_labelled_not_before_kickoff():
    """A resolved row whose own timestamp is after kickoff (a legacy row --
    the reconciler will not resolve one today) is counted and graded like
    any other, with the label withheld. Dropping it would empty the record;
    labelling it pre-kickoff would be a claim its timestamps refute."""
    _insert_game(game_id="g1", commence_time="2026-09-27T17:00:00+00:00")
    _insert_prop(game_id="g1", player_id="early", market="rushing_yards",
                 predicted_value=50.0, actual_value=55.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00",
                 player_name="Early", position="RB")
    _insert_prop(game_id="g1", player_id="late", market="rushing_yards",
                 predicted_value=50.0, actual_value=55.0,
                 snapshotted_at="2026-09-27T18:00:00+00:00",
                 player_name="Late", position="RB")

    rows = {r["player"]: r for r in _rows(
        prop_grading.build_prop_grading(_stats()), "2026-09-27")}
    assert rows["Early"]["made_before_kickoff"] is True
    assert rows["Late"]["made_before_kickoff"] is False
    assert rows["Late"]["grade"] is not None


def test_rows_are_graded_against_their_own_stats_mae():
    _insert_game(game_id="g1", commence_time="2026-09-27T17:00:00+00:00")
    # Passing MAE ~70.5: a 30-yard miss is within half the model's own error.
    _insert_prop(game_id="g1", player_id="qb", market="passing_yards",
                 predicted_value=300.0, actual_value=330.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00", player_name="Qb")
    # Rushing MAE ~21.5: a 50-yard miss is beyond twice it.
    _insert_prop(game_id="g1", player_id="rb", market="rushing_yards",
                 predicted_value=60.0, actual_value=10.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00",
                 player_name="Rb", position="RB")
    # Receiving MAE ~20.9: a 25-yard miss is beyond the error, within twice.
    _insert_prop(game_id="g1", player_id="wr", market="receiving_yards",
                 predicted_value=80.0, actual_value=105.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00",
                 player_name="Wr", position="WR")

    rows = {r["player"]: r for r in _rows(
        prop_grading.build_prop_grading(_stats()), "2026-09-27")}
    assert rows["Qb"]["grade"] == "On the number"
    assert rows["Rb"]["grade"] == "Way off"
    assert rows["Rb"]["delta"] == -50.0  # the player fell short: negative
    assert rows["Wr"]["grade"] == "Off"


def test_no_bands_means_rows_are_served_ungraded():
    _insert_game(game_id="g1", commence_time="2026-09-27T17:00:00+00:00")
    _insert_prop(game_id="g1", player_id="qb", market="passing_yards",
                 predicted_value=300.0, actual_value=390.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00")

    rows = _rows(prop_grading.build_prop_grading(_no_stats()), "2026-09-27")
    assert rows[0]["delta"] == 90.0
    assert rows[0]["grade"] is None


def test_other_markets_are_not_graded_here():
    """anytime_td is a probability (it keeps its bucket block), the forward
    rows carry a book line rather than a projection, and receptions is a
    count-model call -- none of them is a yardage projection to grade."""
    _insert_game(game_id="g1", commence_time="2026-09-27T17:00:00+00:00")
    for market, value in (("anytime_td", 0.7), ("fwd_passing_yards", 275.5),
                          ("receptions", 6.0), ("passing_tds", 2.0)):
        _insert_prop(game_id="g1", player_id=f"p-{market}", market=market,
                     predicted_value=value, actual_value=1.0,
                     snapshotted_at="2026-09-27T12:00:00+00:00")
    _insert_prop(game_id="g1", player_id="qb", market="passing_yards",
                 predicted_value=300.0, actual_value=310.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00")

    payload = prop_grading.build_prop_grading(_stats())
    rows = _rows(payload, "2026-09-27")
    assert [r["stat"] for r in rows] == ["passing_yards"]


# ---------------------------------------------------------------------------
# The sections
# ---------------------------------------------------------------------------


def test_most_confident_ranks_by_departure_from_the_season_to_date_average():
    """The baseline is the mean of the player's own actuals in EARLIER games
    of the same season. The QB's causal baseline on Nov 1 is (200+300)/2 =
    250, so a 330 projection departs by 80 and outranks the WR's 65. Either
    leak -- the graded game's own 340, or the following week's 500 --
    would shrink the QB's departure below the WR's and flip the order, so
    the ranking pins causality in both directions."""
    games = [
        ("w6", "2026-10-18T17:00:00+00:00", "2026-10-18"),
        ("w7", "2026-10-25T17:00:00+00:00", "2026-10-25"),
        ("w8", "2026-11-01T18:00:00+00:00", "2026-11-01"),
        ("w9", "2026-11-08T18:00:00+00:00", "2026-11-08"),
    ]
    for game_id, kickoff, _ in games:
        _insert_game(game_id=game_id, commence_time=kickoff)
    # QB history, then the graded game, then a later game that must not leak back.
    _insert_prop(game_id="w6", player_id="qb", market="passing_yards",
                 predicted_value=210.0, actual_value=200.0,
                 snapshotted_at="2026-10-18T12:00:00+00:00", player_name="Qb")
    _insert_prop(game_id="w7", player_id="qb", market="passing_yards",
                 predicted_value=290.0, actual_value=300.0,
                 snapshotted_at="2026-10-25T12:00:00+00:00", player_name="Qb")
    _insert_prop(game_id="w8", player_id="qb", market="passing_yards",
                 predicted_value=330.0, actual_value=340.0,
                 snapshotted_at="2026-11-01T12:00:00+00:00", player_name="Qb")
    _insert_prop(game_id="w9", player_id="qb", market="passing_yards",
                 predicted_value=480.0, actual_value=500.0,
                 snapshotted_at="2026-11-08T12:00:00+00:00", player_name="Qb")
    # WR: one prior game of 100 receiving yards, projected 165 on Nov 1.
    _insert_prop(game_id="w6", player_id="wr", market="receiving_yards",
                 predicted_value=95.0, actual_value=100.0,
                 snapshotted_at="2026-10-18T12:00:00+00:00",
                 player_name="Wr", position="WR")
    _insert_prop(game_id="w8", player_id="wr", market="receiving_yards",
                 predicted_value=165.0, actual_value=150.0,
                 snapshotted_at="2026-11-01T12:00:00+00:00",
                 player_name="Wr", position="WR")

    day = _day(prop_grading.build_prop_grading(_stats()), "2026-11-01")
    assert [r["player"] for r in day["most_confident"]] == ["Qb", "Wr"]


def test_biggest_surprises_carry_both_directions():
    _insert_game(game_id="g1", commence_time="2026-09-27T17:00:00+00:00")
    _insert_prop(game_id="g1", player_id="over", market="passing_yards",
                 predicted_value=200.0, actual_value=360.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00", player_name="Over")
    _insert_prop(game_id="g1", player_id="under", market="passing_yards",
                 predicted_value=300.0, actual_value=150.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00", player_name="Under")
    _insert_prop(game_id="g1", player_id="close", market="passing_yards",
                 predicted_value=280.0, actual_value=285.0,
                 snapshotted_at="2026-09-27T12:00:00+00:00", player_name="Close")

    day = _day(prop_grading.build_prop_grading(_stats()), "2026-09-27")
    surprises = day["biggest_surprises"]
    assert [r["player"] for r in surprises[:2]] == ["Over", "Under"]
    assert surprises[0]["delta"] == 160.0
    assert surprises[1]["delta"] == -150.0


def test_days_are_newest_first_and_grouped_by_eastern_date():
    """A Thursday-night kickoff at 00:15 UTC is the Thursday's game (20:15
    Eastern), not Friday's; a Sunday-night kickoff at 01:20 UTC Monday is
    Sunday's. The day is read in America/New_York, the zone the schedule's
    own gameday is defined in."""
    _insert_game(game_id="2026_08_CAR_GB", commence_time="2026-10-30T00:15:00+00:00")
    _insert_prop(game_id="2026_08_CAR_GB", player_id="qb", market="passing_yards",
                 predicted_value=300.0, actual_value=310.0,
                 snapshotted_at="2026-10-29T12:00:00+00:00")
    _insert_game(game_id="2026_09_CAR_GB", commence_time="2026-11-02T01:20:00+00:00")
    _insert_prop(game_id="2026_09_CAR_GB", player_id="qb", market="passing_yards",
                 predicted_value=300.0, actual_value=290.0,
                 snapshotted_at="2026-11-01T12:00:00+00:00")

    payload = prop_grading.build_prop_grading(_stats())
    assert [d["date"] for d in payload["days"]] == ["2026-11-01", "2026-10-29"]
    thursday = _day(payload, "2026-10-29")
    assert thursday["games"][0]["label"] == "CAR @ GB"
    assert thursday["games"][0]["rows"][0]["kickoff"] == "2026-10-30T00:15:00+00:00"


# ---------------------------------------------------------------------------
# Team placement
# ---------------------------------------------------------------------------


def test_team_map_from_snapshot_places_players_in_their_game():
    snapshot = {"weeks": {"8": {
        "games": [{"game_id": "2026_08_CAR_GB", "home_team": "GB", "away_team": "CAR"},
                  {"game_id": "2026_08_DAL_PHI", "home_team": "PHI", "away_team": "DAL"}],
        "player_props": [
            {"player_id": "qb1", "recent_team": "GB"},
            {"player_id": "wr1", "recent_team": "DAL"},
            {"player_id": "ghost", "recent_team": "NYJ"},  # no game this week
            {"player_id": "noteam", "recent_team": ""},
        ],
    }}}
    team_map = prop_grading.team_map_from_snapshot(snapshot)
    assert team_map == {("2026_08_CAR_GB", "qb1"): "GB",
                        ("2026_08_DAL_PHI", "wr1"): "DAL"}
    assert prop_grading.team_map_from_snapshot({}) == {}
    assert prop_grading.team_map_from_snapshot(None) == {}


def test_rows_carry_team_and_opponent_from_the_map_and_null_without_one():
    _insert_game(game_id="2026_08_CAR_GB", commence_time="2026-10-30T00:15:00+00:00")
    _insert_prop(game_id="2026_08_CAR_GB", player_id="qb1", market="passing_yards",
                 predicted_value=300.0, actual_value=310.0,
                 snapshotted_at="2026-10-29T12:00:00+00:00", player_name="Placed")
    _insert_prop(game_id="2026_08_CAR_GB", player_id="qb2", market="passing_yards",
                 predicted_value=280.0, actual_value=270.0,
                 snapshotted_at="2026-10-29T12:00:00+00:00", player_name="Unplaced")

    payload = prop_grading.build_prop_grading(
        _stats(), {("2026_08_CAR_GB", "qb1"): "GB"})
    rows = {r["player"]: r for r in _rows(payload, "2026-10-29")}
    assert rows["Placed"]["team"] == "GB"
    assert rows["Placed"]["opponent"] == "CAR"
    assert rows["Unplaced"]["team"] is None
    assert rows["Unplaced"]["opponent"] is None


# ---------------------------------------------------------------------------
# Empty and broken stores
# ---------------------------------------------------------------------------


def test_an_empty_store_grades_nothing():
    payload = prop_grading.build_prop_grading(_stats())
    assert payload["days"] == []
    assert payload["stats"] == _stats()
    assert payload["generated_at"]


def test_a_store_that_is_not_a_database_grades_nothing(tmp_path, monkeypatch):
    """NBA needed this fix after shipping without it: a path whose store was
    never initialised -- here, a file that is not a database at all -- holds
    no recorded props by definition, and building the grading must serve
    the empty payload instead of raising."""
    from nfl_predictor import config

    db_path = tmp_path / "tracking.db"
    db_path.write_bytes(b"this is not a sqlite database")
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)

    payload = prop_grading.build_prop_grading(_stats())
    assert payload["days"] == []


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from nfl_predictor import config
    from nfl_predictor.api.main import app

    db_path = tmp_path / "tracking.db"
    monkeypatch.setattr(config, "TRACKING_DB_PATH", db_path)
    monkeypatch.setattr(store, "TRACKING_DB_PATH", db_path)
    return TestClient(app)


def test_prop_grading_route_serves_the_contract_shape(client):
    _insert_game(game_id="2026_08_CAR_GB", commence_time="2026-10-30T00:15:00+00:00")
    _insert_prop(game_id="2026_08_CAR_GB", player_id="qb", market="passing_yards",
                 predicted_value=300.0, actual_value=330.0,
                 snapshotted_at="2026-10-29T12:00:00+00:00", player_name="Qb")

    response = client.get("/api/prop-grading")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"generated_at", "stats", "days"}
    assert set(body["stats"]) == set(prop_grading.GRADED_STATS)
    (day,) = body["days"]
    assert set(day) == {"date", "most_confident", "biggest_surprises", "games"}
    assert day["date"] == "2026-10-29"
    (game,) = day["games"]
    assert game["game_id"] == "2026_08_CAR_GB"
    assert game["label"] == "CAR @ GB"
    (row,) = game["rows"]
    assert set(row) == {"player", "team", "opponent", "stat", "projected",
                        "actual", "delta", "grade", "made_before_kickoff", "kickoff"}
    assert row["player"] == "Qb"
    assert row["delta"] == 30.0
    assert row["grade"] == "On the number"
    assert row["made_before_kickoff"] is True


def test_prop_grading_route_is_empty_but_valid_with_no_props(client):
    response = client.get("/api/prop-grading")
    assert response.status_code == 200
    assert response.json()["days"] == []
