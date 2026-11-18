import pandas as pd

from nfl_predictor.features import qb


def test_pregame_qb_reads_only_prior_games_and_falls_back_to_last_starter():
    games = pd.DataFrame({
        "game_id": ["g1", "g2", "g3"], "gameday": ["2020-09-01", "2020-09-08", "2020-09-15"],
        "home_team": ["OAK", "OAK", "OAK"], "away_team": ["X", "Y", "Z"],
    })
    # Play-by-play uses the relocated code (LV); schedules use OAK. A starts g1 and g2 (big EPA in g2), B relieves.
    qbg = pd.DataFrame({
        "game_id": ["g1", "g2", "g2", "g3"], "team": ["LV", "LV", "LV", "LV"], "qb_id": ["A", "A", "B", "B"],
        "qb_name": ["a", "a", "b", "b"], "dropbacks": [30, 30, 5, 40], "epa_sum": [3.0, 30.0, -1.0, 99.0],
    })
    out = qb.add_pregame_qb_features(games, qbg, starters={("g3", "OAK"): "B"}).set_index("game_id")
    # g1: no prior game for anyone -> no starter known, prior rating, "new".
    assert out.loc["g1", "home_qb_games"] == 0 and out.loc["g1", "home_qb_new"] == 1
    # g2: no chart -> last game's starter A, rated on g1 only (never on g2's own 30 EPA).
    assert out.loc["g2", "home_qb_games"] == 1
    assert out.loc["g2", "home_qb_epa_pd"] == 3.0 / (30 + qb.SHRINK_K)
    assert out.loc["g2", "home_qb_changed"] == 0
    # g3: chart names B, a change from A; B's rating is his g2 relief only, not g3's 99 EPA.
    assert out.loc["g3", "home_qb_changed"] == 1
    assert out.loc["g3", "home_qb_epa_pd"] == -1.0 / (5 + qb.SHRINK_K)
