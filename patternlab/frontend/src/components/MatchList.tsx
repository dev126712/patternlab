import type { Match } from "../api";

interface Props {
  matches: Match[] | null;
  selectedId?: number | null;
  onSelect?: (m: Match) => void;
}

export default function MatchList({ matches, selectedId = null, onSelect }: Props) {
  if (matches === null) {
    return (
      <div className="panel">
        <h2>Historical matches</h2>
        <p className="muted">Persistence is off (no DATABASE_URL) — no history to match against.</p>
      </div>
    );
  }
  return (
    <div className="panel">
      <h2>Historical matches</h2>
      {matches.length === 0 ? (
        <p className="muted">No stored sessions yet. Save sessions to build the match library.</p>
      ) : (
        <>
        <p className="muted match-hint">Click a match to compare it with today.</p>
        <table className="matches">
          <thead>
            <tr>
              <th>Date</th>
              <th>Instrument</th>
              <th>Score</th>
              <th>Fingerprint</th>
              <th>DTW</th>
            </tr>
          </thead>
          <tbody>
            {matches.map((m) => (
              <tr
                key={m.session_id}
                className={m.session_id === selectedId ? "selected" : undefined}
                onClick={() => onSelect?.(m)}
              >
                <td>{m.session_date}</td>
                <td>{m.instrument}</td>
                <td>
                  <div className="score-cell">
                    <div className="score-bar" style={{ width: `${m.similarity_score * 100}%` }} />
                    <span>{m.similarity_score.toFixed(3)}</span>
                  </div>
                </td>
                <td>{m.fingerprint_cosine.toFixed(3)}</td>
                <td>{m.dtw_similarity.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        </>
      )}
    </div>
  );
}
