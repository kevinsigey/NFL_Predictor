import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nfl_predictor.api.main import app
from nfl_predictor.api import routes


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_upcoming_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": seasons[0], "week": 1, "gameday": "2025-08-01",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(
        routes, "_load_models_cached",
        lambda: {
            "game_outcome_model": None, "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
            "total_model": None, "player_models": {"feature_cols": [], "anytime_td": None},
            "feature_cols": [], "player_feature_cols": [],
        },
    )
    monkeypatch.setattr(
        routes, "_predict_game_from_models",
        lambda models, home, away, games_df, spread_line=None, total_line=None, gameday=None, blocks=None, aux=None, starters=None, game_schedule=None: {
            "home_win_prob": 0.6, "away_win_prob": 0.4, "home_cover_prob": 0.55, "away_cover_prob": 0.45,
            "over_prob": 0.52, "under_prob": 0.48,
        },
    )
    # Accepts **kwargs because /track-record now threads the season and week through
    # (the `weekly` rows have to span every elapsed week, and only the calendar knows
    # which those are). `test_get_track_record_passes_the_calendar_to_the_store` below
    # pins that the two really arrive.
    monkeypatch.setattr(
        routes.store, "get_track_record",
        lambda **kwargs: {"n_resolved_games": 0, "pct_moneyline_correct": None},
    )
    return TestClient(app)


def test_get_games_returns_week_slate(client):
    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["game_id"] == "2025_01_BAL_KC"


def test_upcoming_game_gets_conditions(client, monkeypatch):
    """An upcoming game (no scores) carries conditions from the forecast cache."""
    monkeypatch.setattr(
        routes, "_forecast_cache",
        type("FakeCache", (), {"get": lambda self, stadium, kickoff, now=None, budget=None: {"kind": "rain", "temp_f": 50, "source": "open-meteo"}})(),
    )
    body = client.get("/api/games?season=2025&week=1").json()
    assert body[0]["conditions"] == {"kind": "rain", "temp_f": 50, "source": "open-meteo"}


def test_schedule_shaped_upcoming_game_gets_conditions(client, monkeypatch):
    """The weekly schedule hands `gameday` as the UTC instant and no `gametime`;
    the games route must read that instant as-is (not re-apply the ET offset,
    which would push it eight hours -- and a late kickoff a full day -- off)."""
    weekday_records = [{
        "game_id": "2025_01_BAL_KC", "season": 2025, "week": 1,
        "gameday": pd.Timestamp("2025-09-05 00:20:00"), "home_team": "BAL", "away_team": "KC",
        "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5,
    }]
    monkeypatch.setattr(routes.schedules, "fetch_week_games",
                        lambda season, week: pd.DataFrame(weekday_records))
    seen = {}

    class _CapturingCache:
        def get(self, stadium, kickoff_iso, now=None, budget=None):
            seen["kickoff"] = kickoff_iso
            return {"kind": "rain", "temp_f": 50, "source": "open-meteo"}

    monkeypatch.setattr(routes, "_forecast_cache", _CapturingCache())
    body = client.get("/api/games?season=2025&week=1").json()
    assert body[0]["conditions"] == {"kind": "rain", "temp_f": 50, "source": "open-meteo"}
    assert seen["kickoff"] == "2025-09-05T00:20:00Z", \
        "an already-UTC gameday must pass through untouched, not get the ET offset"


def test_final_game_has_no_conditions(client, monkeypatch):
    """A final game (scores present) never gets conditions."""
    monkeypatch.setattr(
        routes, "_forecast_cache",
        type("FakeCache", (), {"get": lambda self, stadium, kickoff, now=None, budget=None: {"kind": "rain"}})(),
    )
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": 24, "away_score": 17, "spread_line": -2.5, "total_line": 46.5}]
        ),
    )
    body = client.get("/api/games?season=2025&week=1").json()
    assert "conditions" not in body[0]


def test_none_conditions_leaves_no_key(client, monkeypatch):
    """A None result from the cache leaves no conditions key on the game."""
    monkeypatch.setattr(
        routes, "_forecast_cache",
        type("FakeCache", (), {"get": lambda self, stadium, kickoff, now=None, budget=None: None})(),
    )
    body = client.get("/api/games?season=2025&week=1").json()
    assert "conditions" not in body[0]


def test_get_game_prediction(client):
    response = client.get("/api/games/2025/1/2025_01_BAL_KC/prediction")

    assert response.status_code == 200
    body = response.json()
    assert body["home_win_prob"] == 0.6


def test_predict_game_from_models_includes_sigma_and_total_sigma(monkeypatch):
    class _FakeTotalModel:
        def predict(self, _X):
            return [45.0]

    monkeypatch.setattr(
        routes.feature_build, "build_features_for_game",
        lambda home, away, games_df, gameday=None, blocks=None, aux=None, starters=None, game_schedule=None: pd.Series({"rating_diff": 50.0, "home_rest_days": 7.0, "away_rest_days": 7.0}),
    )
    models = {
        "feature_cols": ["rating_diff", "home_rest_days", "away_rest_days"],
        "chosen_candidate": "elo", "sigma": 12.0, "total_sigma": 10.0,
        "total_model": _FakeTotalModel(),
    }

    result = routes._predict_game_from_models(models, "KC", "BAL", pd.DataFrame())

    assert result["sigma"] == 12.0
    assert result["total_sigma"] == 10.0
    assert "predicted_margin" in result
    assert "predicted_total" in result


def test_get_game_prediction_404s_for_unknown_game(client):
    response = client.get("/api/games/2025/1/nonexistent/prediction")

    assert response.status_code == 404


def test_get_track_record(client, monkeypatch):
    # `current_season_and_week` reads the real schedule to anchor week 1, and the
    # store below ignores the value, so it is stubbed rather than fetched: this
    # route is not what is under test, and the suite-wide live-upstream guard
    # (tests/live_upstream.py) fails any run that reaches for it.
    monkeypatch.setattr(routes, "current_season_and_week", lambda: (2025, 1))
    response = client.get("/api/track-record")

    assert response.status_code == 200
    assert response.json()["n_resolved_games"] == 0


def test_get_track_record_passes_the_calendar_to_the_store(monkeypatch):
    """The store can only list every elapsed week if it is told which week it is.

    Without these two it falls back to the newest week it happens to hold, and a week
    at the tail of the record that nobody picked in drops off the end of the chart --
    the exact silent absence this payload exists to remove. `current_season_and_week`
    reads the real schedule, so it is stubbed rather than recomputed.
    """
    seen = {}

    def fake_get_track_record(**kwargs):
        seen.update(kwargs)
        return {"games": {}, "player_props": {}}

    monkeypatch.setattr(routes, "current_season_and_week", lambda: (2026, 7))
    monkeypatch.setattr(routes.store, "get_track_record", fake_get_track_record)

    routes.get_track_record()

    assert seen == {"current_week": 7, "season": 2026}


def test_get_games_handles_nan_scores_for_unplayed_games(client, monkeypatch):
    # Real upcoming-game data (unplayed games) carries home_score/away_score
    # as genuine pandas/numpy float NaN, not Python None -- reproduce that
    # exactly (float("nan") forces a float64 column, matching the real
    # schedules data) rather than the `client` fixture's None, which pandas
    # keeps as an object-dtype None and never actually reproduces the bug.
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_01_BAL_KC", "season": season, "week": week,
              "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
              "home_score": float("nan"), "away_score": float("nan"),
              "spread_line": -2.5, "total_line": 46.5}]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["home_score"] is None
    assert body[0]["away_score"] is None


def test_get_games_includes_finished_games(client, monkeypatch):
    # A week with a mix of finished and still-upcoming games must return
    # both -- fetch_upcoming_games alone would silently drop the finished
    # one, which is exactly the bug this route must not have.
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [
                {"game_id": "2025_01_BAL_KC", "season": season, "week": week,
                 "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                 "home_score": 27, "away_score": 20, "spread_line": -2.5, "total_line": 46.5},
                {"game_id": "2025_01_PHI_GB", "season": season, "week": week,
                 "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                 "home_score": None, "away_score": None, "spread_line": 1.5, "total_line": 45.0},
            ]
        ),
    )

    response = client.get("/api/games?season=2025&week=1")

    assert response.status_code == 200
    game_ids = {g["game_id"] for g in response.json()}
    assert game_ids == {"2025_01_BAL_KC", "2025_01_PHI_GB"}


def test_get_player_props_includes_recent_team_and_position(client, monkeypatch):
    # The roster fallback below is reached because the stub history above has no
    # row for either team in the game, so `missing_teams` is non-empty and
    # `player_stats.fetch_seasonal_roster` runs. Unstubbed that reads nflverse,
    # which is the pre-existing tenth connection the offline guard now names.
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team"]),
    )
    monkeypatch.setattr(
        routes, "_load_player_history",
        lambda season: pd.DataFrame(
            [{"player_id": "00-001", "player_name": "Pat Mahomes", "position": "QB",
              "recent_team": "KC", "season": season}]
        ),
    )
    # `season` and `week` are both passed as keywords -- `season` so a 2026 roster is
    # not scored on 2024 stat lines, `week` so the feature is the pregame view of the
    # gameweek being predicted rather than every played week. Recorded so the
    # assertions below can check the call site actually passes them; a
    # default-accepting lambda that ignores them let both be deleted silently.
    seen_kwargs: list[dict] = []

    def _stub(player_id, history, season=None, week=None, window=5):
        seen_kwargs.append({"player_id": player_id, "season": season, "week": week, "window": window})
        return pd.Series({"dummy_feature": 1.0})

    monkeypatch.setattr(routes.player_usage, "build_features_for_player", _stub)
    monkeypatch.setattr(
        routes.player_props, "predict_props",
        lambda player_models, feature_row, position: {"anytime_td_prob": 0.42, "passing_yards": 275.0},
    )
    # Two fetches `_get_player_props_live` makes that this test is not about: the
    # season-roster fallback for a team with no current-season stats yet (the
    # fixture's slate carries BAL, and only KC is in the history above), and the
    # depth chart. Both are network reads, so the suite-wide guard
    # (tests/live_upstream.py) fails a run that reaches them, and an empty answer
    # is what a team with no roster data and no chart already produces live.
    monkeypatch.setattr(
        routes.player_stats, "fetch_seasonal_roster",
        lambda season: pd.DataFrame(columns=["player_id", "player_name", "position", "recent_team", "season"]),
    )
    monkeypatch.setattr(routes.depth_charts, "flags_for_season_week", lambda *a, **k: {})

    response = client.get("/api/players/2025/1/props")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["recent_team"] == "KC"
    assert body[0]["position"] == "QB"

    # The call site must pass the season AND the gameweek being predicted. Deleting
    # either keyword left this suite green, because the stub above accepted
    # defaults for both and nothing looked at what it was handed.
    assert seen_kwargs, "build_features_for_player was never called"
    for call in seen_kwargs:
        assert call["season"] == 2025, f"season not forwarded: {call}"
        assert call["week"] == 1, f"week not forwarded: {call}"


def test_get_game_verdict_404s_for_untracked_game(client, monkeypatch):
    monkeypatch.setattr(routes.store, "get_game_verdict", lambda game_id: None)

    response = client.get("/api/games/nope/verdict")

    assert response.status_code == 404


def test_get_predictions_for_week_returns_prediction_status(client, monkeypatch):
    predictions = [
        {"game_id": "2025_01_BAL_KC", "status": "pending", "home_win_prob": 0.6, "away_win_prob": 0.4, "verdict": None},
    ]
    monkeypatch.setattr(routes.store, "get_predictions_for_week", lambda season, week, games: predictions)

    response = client.get("/api/predictions/2025/1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["game_id"] == "2025_01_BAL_KC"
    assert body[0]["status"] == "pending"
    assert body[0]["home_win_prob"] == 0.6


def test_get_predictions_for_week_includes_both_resolved_and_pending_games(client, monkeypatch):
    """A week in progress has both finished games (from load_training_data)
    and still-upcoming ones (from fetch_upcoming_games). The old if/else
    fallback dropped the finished game whenever any game that week was
    still unplayed (see this plan's final review, finding B3); the route
    must union both sources instead. The client fixture's default mocks
    already return one pending game ("2025_01_BAL_KC", week=1) from
    fetch_upcoming_games and one finished game ("g0", week=1) from
    load_training_data."""
    captured = {}

    def fake_get_predictions_for_week(season, week, games):
        captured["game_ids"] = set(games["game_id"])
        return [
            {"game_id": gid, "status": "resolved" if gid == "g0" else "pending", "verdict": None}
            for gid in games["game_id"]
        ]

    monkeypatch.setattr(routes.store, "get_predictions_for_week", fake_get_predictions_for_week)

    response = client.get("/api/predictions/2025/1")

    assert response.status_code == 200
    assert captured["game_ids"] == {"2025_01_BAL_KC", "g0"}
    body = response.json()
    statuses = {row["game_id"]: row["status"] for row in body}
    assert statuses["2025_01_BAL_KC"] == "pending"
    assert statuses["g0"] == "resolved"


def test_get_power_rankings_sorts_descending_by_rating_and_includes_division(client, monkeypatch):
    monkeypatch.setattr(
        routes.power_ratings, "final_ratings",
        lambda history: {"BAL": 1550.0, "KC": 1480.0},
    )
    monkeypatch.setattr(
        routes.teams_data, "fetch_team_conferences",
        lambda: pd.DataFrame(
            [
                {"team": "BAL", "conference": "AFC", "division": "AFC North"},
                {"team": "KC", "conference": "AFC", "division": "AFC West"},
            ]
        ),
    )

    response = client.get("/api/power-rankings?season=2025")

    assert response.status_code == 200
    body = response.json()
    assert body["season"] == 2025
    rankings = body["rankings"]
    assert [r["team"] for r in rankings] == ["BAL", "KC"]
    assert rankings[0]["rank"] == 1
    assert rankings[1]["rank"] == 2
    assert rankings[0]["division"] == "AFC North"
    assert rankings[1]["division"] == "AFC West"


def test_get_power_rankings_includes_win_loss_record(client, monkeypatch):
    # Override load_training_data to return a played game for the exact
    # requested season -- BAL beat KC 24-17, so BAL should show 1 win.
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [{"game_id": "g0", "season": 2025, "week": 1, "gameday": "2025-09-04",
              "home_team": "BAL", "away_team": "KC", "home_score": 24, "away_score": 17}]
        ),
    )
    monkeypatch.setattr(
        routes.power_ratings, "final_ratings",
        lambda history: {"BAL": 1550.0, "KC": 1480.0},
    )
    monkeypatch.setattr(
        routes.teams_data, "fetch_team_conferences",
        lambda: pd.DataFrame(
            [
                {"team": "BAL", "conference": "AFC", "division": "AFC North"},
                {"team": "KC", "conference": "AFC", "division": "AFC West"},
            ]
        ),
    )

    response = client.get("/api/power-rankings?season=2025")

    body = response.json()
    by_team = {r["team"]: r for r in body["rankings"]}
    assert by_team["BAL"]["wins"] == 1
    assert by_team["BAL"]["losses"] == 0
    assert by_team["KC"]["wins"] == 0
    assert by_team["KC"]["losses"] == 1


def test_get_predictions_batch_is_keyed_by_game_id(client):
    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"2025_01_BAL_KC"}
    assert body["2025_01_BAL_KC"]["home_win_prob"] == 0.6


def test_get_predictions_batch_skips_one_failing_game_without_failing_the_rest(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [
                {"game_id": "bad_game", "season": season, "week": week,
                 "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                 "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5},
                {"game_id": "good_game", "season": season, "week": week,
                 "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                 "home_score": None, "away_score": None, "spread_line": 1.5, "total_line": 45.0},
            ]
        ),
    )

    def flaky_predict(models, home, away, games_df, spread_line=None, total_line=None, gameday=None, blocks=None, aux=None, starters=None, game_schedule=None):
        if home == "BAL":
            raise ValueError("feature build blew up")
        return {"home_win_prob": 0.6, "away_win_prob": 0.4}

    monkeypatch.setattr(routes, "_predict_game_from_models", flaky_predict)

    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"good_game"}


def test_get_predictions_batch_loads_history_once_for_the_whole_week(client, monkeypatch):
    """The plan requires the live batch fallback to load games/models/history
    once and call _predict_game_from_models per game -- not re-fetch and
    re-load history inside the loop for every game."""
    load_calls = []
    real_load_game_history = routes._load_game_history

    def counting_load(season):
        load_calls.append(season)
        return real_load_game_history(season)

    monkeypatch.setattr(routes, "_load_game_history", counting_load)
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [
                {"game_id": "g1", "season": season, "week": week,
                 "gameday": "2025-09-04", "home_team": "BAL", "away_team": "KC",
                 "home_score": None, "away_score": None, "spread_line": -2.5, "total_line": 46.5},
                {"game_id": "g2", "season": season, "week": week,
                 "gameday": "2025-09-05", "home_team": "GB", "away_team": "PHI",
                 "home_score": None, "away_score": None, "spread_line": 1.5, "total_line": 45.0},
            ]
        ),
    )

    response = client.get("/api/predictions/2025/1/batch")

    assert response.status_code == 200
    assert set(response.json().keys()) == {"g1", "g2"}
    assert load_calls == [2025]


def test_get_team_form_returns_last_n_games_with_result(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": "g1", "season": 2025, "week": 1, "gameday": "2025-09-04",
                 "home_team": "BAL", "away_team": "KC", "home_score": 27, "away_score": 20},
                {"game_id": "g2", "season": 2025, "week": 2, "gameday": "2025-09-11",
                 "home_team": "KC", "away_team": "BAL", "home_score": 24, "away_score": 10},
            ]
        ),
    )

    response = client.get("/api/teams/BAL/form?season=2025&n=5")

    assert response.status_code == 200
    body = response.json()
    assert body["team"] == "BAL"
    form = body["recent_form"]
    assert len(form) == 2
    # g1: BAL home, won 27-20
    assert form[0]["opponent"] == "KC"
    assert form[0]["is_home"] is True
    assert form[0]["result"] == "W"
    assert form[0]["team_score"] == 27
    # g2: BAL away, lost 10-24
    assert form[1]["opponent"] == "KC"
    assert form[1]["is_home"] is False
    assert form[1]["result"] == "L"
    assert form[1]["team_score"] == 10


def test_get_team_form_respects_n_limit(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": f"g{i}", "season": 2025, "week": i, "gameday": f"2025-09-{i:02d}",
                 "home_team": "BAL", "away_team": "KC", "home_score": 20 + i, "away_score": 10}
                for i in range(1, 8)
            ]
        ),
    )

    response = client.get("/api/teams/BAL/form?season=2025&n=3")

    assert response.status_code == 200
    body = response.json()
    assert len(body["recent_form"]) == 3
    # Should be the last 3 chronologically (g5, g6, g7)
    assert [f["game_id"] for f in body["recent_form"]] == ["g5", "g6", "g7"]


def test_get_team_form_excludes_games_from_other_seasons(client, monkeypatch):
    """_load_game_history spans multiple seasons for power-rating
    continuity -- recent form must not bleed in last season's results."""
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": "g_old", "season": 2024, "week": 18, "gameday": "2025-01-05",
                 "home_team": "BAL", "away_team": "PIT", "home_score": 34, "away_score": 17},
                {"game_id": "g_new", "season": 2025, "week": 1, "gameday": "2025-09-04",
                 "home_team": "BAL", "away_team": "KC", "home_score": 27, "away_score": 20},
            ]
        ),
    )

    response = client.get("/api/teams/BAL/form?season=2025&n=5")

    assert response.status_code == 200
    form = response.json()["recent_form"]
    assert len(form) == 1
    assert form[0]["game_id"] == "g_new"


def test_get_head_to_head_returns_past_meetings(client, monkeypatch):
    monkeypatch.setattr(
        routes.schedules, "fetch_week_games",
        lambda season, week: pd.DataFrame(
            [{"game_id": "2025_09_LV_KC", "season": season, "week": week,
              "gameday": "2025-11-02", "home_team": "KC", "away_team": "LV",
              "home_score": None, "away_score": None}]
        ),
    )
    monkeypatch.setattr(
        routes.schedules, "load_training_data",
        lambda seasons: pd.DataFrame(
            [
                {"game_id": "g_old", "season": 2024, "week": 15, "gameday": "2024-12-14",
                 "home_team": "LV", "away_team": "KC", "home_score": 17, "away_score": 24},
                {"game_id": "g_unrelated", "season": 2024, "week": 15, "gameday": "2024-12-14",
                 "home_team": "DEN", "away_team": "LAC", "home_score": 10, "away_score": 3},
            ]
        ),
    )

    response = client.get("/api/games/2025_09_LV_KC/head-to-head?season=2025&week=9&n_seasons=8")

    assert response.status_code == 200
    body = response.json()
    assert body["game_id"] == "2025_09_LV_KC"
    meetings = body["meetings"]
    assert len(meetings) == 1
    assert meetings[0]["game_id"] == "g_old"
    assert meetings[0]["home_team"] == "LV"
    assert meetings[0]["away_team"] == "KC"


def test_get_head_to_head_404s_for_unknown_game(client, monkeypatch):
    monkeypatch.setattr(routes.schedules, "fetch_week_games", lambda season, week: pd.DataFrame(columns=["game_id", "home_team", "away_team"]))

    response = client.get("/api/games/nope/head-to-head?season=2025&week=9")

    assert response.status_code == 404


# --- Track-record payload split (Sports_Predictor PR #81 contract) ---------
# `?picks=deferred` returns the summary with NO row-level picks; the game rows
# move to `/track-record/picks`; a plain request keeps the historical payload.

_SPLIT_RECORD = {
    "games": {
        "n_resolved": 2,
        "pct_moneyline_correct": 0.5,
        "per_pick": [
            {"game_id": "2025_01_BAL_KC", "pick": "KC"},
            {"game_id": "2025_01_BUF_MIA", "pick": "BUF"},
        ],
    },
    "player_props": {
        "n_resolved": 3,
        "per_pick": [{"player_id": "00-0036355", "market": "passing_yards"}],
    },
    "n_resolved_games": 2,
}


@pytest.fixture
def split_client(client, monkeypatch):
    """The shared `client` fixture's store stub carries no pick rows; this one
    serves a record that does, so the split has something to split. Deep-copied
    per call because the deferred path strips keys off the dict it is handed."""
    monkeypatch.setattr(
        routes.store, "get_track_record",
        lambda **kwargs: copy.deepcopy(_SPLIT_RECORD),
    )
    monkeypatch.setattr(routes.store, "get_calibration", lambda: {"winner": []})
    monkeypatch.setattr(routes, "current_season_and_week", lambda: (2025, 1))
    return client


def test_track_record_default_still_inlines_per_pick_rows(split_client):
    response = split_client.get("/api/track-record")

    assert response.status_code == 200
    body = response.json()
    assert body["games"]["per_pick"] == _SPLIT_RECORD["games"]["per_pick"]
    assert body["player_props"]["per_pick"] == _SPLIT_RECORD["player_props"]["per_pick"]
    assert body["games"]["n_resolved"] == 2
    assert body["calibration"] == {"winner": []}


def test_track_record_deferred_omits_rows_and_keeps_summary_identical(split_client):
    response = split_client.get("/api/track-record?picks=deferred")

    assert response.status_code == 200
    body = response.json()
    assert "per_pick" not in body["games"]
    assert "per_pick" not in body["player_props"]
    expected = copy.deepcopy(_SPLIT_RECORD)
    del expected["games"]["per_pick"]
    del expected["player_props"]["per_pick"]
    expected["calibration"] = {"winner": []}
    assert body == expected


def test_track_record_picks_endpoint_returns_the_game_rows(split_client):
    response = split_client.get("/api/track-record/picks")

    assert response.status_code == 200
    assert response.json() == {"per_pick": _SPLIT_RECORD["games"]["per_pick"]}


def test_track_record_unknown_picks_value_keeps_full_payload(split_client):
    body = split_client.get("/api/track-record?picks=inline").json()

    assert body["games"]["per_pick"] == _SPLIT_RECORD["games"]["per_pick"]
    assert body["player_props"]["per_pick"] == _SPLIT_RECORD["player_props"]["per_pick"]
