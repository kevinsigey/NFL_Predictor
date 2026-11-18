export interface GameSummary {
  game_id: string;
  season: number;
  week: number;
  gameday: string;
  home_team: string;
  away_team: string;
  home_score: number | null;
  away_score: number | null;
  spread_line: number | null;
  total_line: number | null;
  // Kalshi game-winner price (home win prob), pre-kickoff only; null when none was recorded.
  market_home_win_probability?: number | null;
}

export interface GamePrediction {
  home_win_prob: number;
  away_win_prob: number;
  home_cover_prob: number | null;
  away_cover_prob: number | null;
  over_prob: number | null;
  under_prob: number | null;
}

export interface PlayerPropPrediction {
  player_id: string;
  player_name: string;
  anytime_td_prob: number;
  passing_yards?: number;
  rushing_yards?: number;
  receiving_yards?: number;
}

export interface TrackRecord {
  n_resolved_games: number;
  pct_moneyline_correct: number | null;
}

export interface RetrainResponse {
  trained_at: string;
  chosen_candidate: string;
}
