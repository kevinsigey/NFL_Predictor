import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { GamePrediction, GameSummary } from "../types";

export function GamesPage() {
  const [season, setSeason] = useState(2026);
  const [week, setWeek] = useState(1);
  const [games, setGames] = useState<GameSummary[]>([]);
  const [predictions, setPredictions] = useState<Record<string, GamePrediction>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .games(season, week)
      .then(async (fetchedGames) => {
        setGames(fetchedGames);
        const entries = await Promise.all(
          fetchedGames.map(async (g) => {
            try {
              const prediction = await api.gamePrediction(season, week, g.game_id);
              return [g.game_id, prediction] as const;
            } catch {
              return null;
            }
          }),
        );
        setPredictions(Object.fromEntries(entries.filter((e): e is [string, GamePrediction] => e !== null)));
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [season, week]);

  return (
    <div>
      <h1>Week {week} Games</h1>
      <div>
        <label>
          Season:{" "}
          <input type="number" value={season} onChange={(e) => setSeason(Number(e.target.value))} />
        </label>
        <label>
          Week:{" "}
          <input type="number" min={1} max={22} value={week} onChange={(e) => setWeek(Number(e.target.value))} />
        </label>
      </div>
      {loading && <p>Loading…</p>}
      {error && <p role="alert">{error}</p>}
      <ul>
        {games.map((game) => {
          const prediction = predictions[game.game_id];
          return (
            <li key={game.game_id}>
              <strong>{game.away_team} @ {game.home_team}</strong> — {new Date(game.gameday).toLocaleDateString()}
              {/* The market leads: Kalshi's game-winner price. The model's number sits beside it, smaller. The spread/total
                  stay as secondary text. No price, no market figure: it is never defaulted. */}
              {game.market_home_win_probability != null && (
                <span style={{ fontSize: "1.15em", fontWeight: 600 }} data-testid="market-headline">
                  {" "}— Market: {game.market_home_win_probability >= 0.5 ? game.home_team : game.away_team}{" · "}
                  {Math.round(Math.max(game.market_home_win_probability, 1 - game.market_home_win_probability) * 100)}%
                </span>
              )}
              {game.spread_line != null && (
                <small data-testid="market-line">
                  {" "}({game.spread_line >= 0 ? game.home_team : game.away_team} {game.spread_line === 0 ? "pick'em" : `-${Math.abs(game.spread_line)}`}
                  {game.total_line != null && ` · total ${game.total_line}`})
                </small>
              )}
              {prediction && (
                <small>
                  {" "}— Model: {game.market_home_win_probability == null || game.market_home_win_probability >= 0.5
                    ? `${game.home_team} ${Math.round(prediction.home_win_prob * 100)}%`
                    : `${game.away_team} ${Math.round(prediction.away_win_prob * 100)}%`}
                  {prediction.over_prob != null && ` · over ${Math.round(prediction.over_prob * 100)}%`}
                  {game.market_home_win_probability == null && " · no market price"}
                </small>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
