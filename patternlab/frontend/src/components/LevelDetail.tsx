import type { Analysis, Level } from "../api";

interface Props {
  level: Level;
  analysis: Analysis;
  firstBarTime: number; // unix seconds of the first charted bar
  onClose: () => void;
}

const fmt = (t: number) =>
  new Date(t * 1000).toLocaleString("en-GB", {
    timeZone: "UTC",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }) + " UTC";

export default function LevelDetail({ level, analysis, firstBarTime, onClose }: Props) {
  const events = level.events ?? [];
  const asResistance = events.filter((e) => e.kind === "H").length;
  const asSupport = events.length - asResistance;
  const isKey = analysis.key_level !== null && level.center === analysis.key_level.center;
  const first = events[0];
  const last = events[events.length - 1];
  // Role flip: resistance earlier, support later (or vice versa) = the classic S/R polarity change.
  const flipped =
    events.length >= 2 &&
    new Set(events.map((e) => e.kind)).size === 2 &&
    events[0].kind !== events[events.length - 1].kind;

  return (
    <div className="level-detail">
      <div className="level-detail-head">
        <span className="level-detail-title">
          Level {level.center.toFixed(2)}
          {isKey && <span className="key-badge">KEY</span>}
        </span>
        <span className="level-detail-sub">
          zone {level.lo.toFixed(2)}–{level.hi.toFixed(2)} · weight {level.weight.toFixed(1)}
        </span>
        <button className="level-detail-close" onClick={onClose} title="Close">
          ×
        </button>
      </div>

      <p className="level-detail-why">
        {isKey
          ? "This is the KEY level: the nearest detected level at or above the current price — the resistance the market is pressing into right now. "
          : "Detected S/R level: swing pivots clustered within a fraction of ATR of each other. "}
        Price reversed here <b>{events.length}</b> times
        {events.length > 0 && (
          <>
            {" "}
            — <b>{asResistance}</b> rejected as resistance, <b>{asSupport}</b> held as support
          </>
        )}
        {first && last && (
          <>
            , first on <b>{fmt(first.time)}</b>, most recently on <b>{fmt(last.time)}</b>
          </>
        )}
        .
        {flipped &&
          (events[0].kind === "H"
            ? " The role flipped: it started as resistance and later held as support — broken resistance acting as support."
            : " The role flipped: it started as support and later capped price as resistance — broken support acting as resistance.")}
      </p>

      {events.length > 0 ? (
        <table className="level-detail-table">
          <thead>
            <tr>
              <th>When</th>
              <th>Price</th>
              <th>What happened</th>
            </tr>
          </thead>
          <tbody>
            {[...events].reverse().map((e, i) => (
              <tr key={i}>
                <td>
                  {fmt(e.time)}
                  {e.time < firstBarTime && <span className="muted"> (before chart)</span>}
                </td>
                <td>{e.price.toFixed(2)}</td>
                <td>
                  {e.kind === "H"
                    ? "swing high — sellers rejected price here"
                    : "swing low — buyers defended price here"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="muted">
          No per-touch history available for this level (older saved session).
        </p>
      )}
    </div>
  );
}
