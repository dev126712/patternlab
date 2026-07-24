import type { Analysis, Regime } from "../api";

const REGIME_COLORS: Record<Regime["kind"], string> = {
  consolidation: "#3d4a63",
  impulse_up: "#26a69a",
  impulse_down: "#ef5350",
  mixed: "#5c6678",
};

const CONTEXT_LABELS: Record<string, string> = {
  fresh_base: "Fresh-base breakout",
  recovery_leg: "Recovery leg (trapped supply overhead)",
  mixed: "Mixed context",
};

function RegimeStrip({ regimes, nBars }: { regimes: Regime[]; nBars: number }) {
  return (
    <div className="regime-strip" title="Session regimes (time-ordered)">
      {regimes.map((r, i) => (
        <div
          key={i}
          className="regime-seg"
          style={{
            width: `${(r.bars / nBars) * 100}%`,
            background: REGIME_COLORS[r.kind] ?? REGIME_COLORS.mixed,
          }}
          title={`${r.kind} · ${r.bars} bars · ${r.net_move_atr} ATR net · ER ${r.er}`}
        />
      ))}
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`stat${accent ? " stat-accent" : ""}`}>
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
    </div>
  );
}

export default function AnalysisPanel({ analysis }: { analysis: Analysis }) {
  const { approach, context, stairstep, time_at_key_level: tal, key_level } = analysis;
  return (
    <div className="panel">
      <h2>Structural read</h2>
      <RegimeStrip regimes={analysis.regimes} nBars={analysis.n_bars} />
      <div className="stat-grid">
        <Stat
          label="Context"
          value={CONTEXT_LABELS[context.kind] ?? context.kind}
          accent={context.trapped_supply_overhead}
        />
        <Stat
          label="Approach"
          value={`${approach.classification} (${approach.slope_atr_per_bar} ATR/bar ${approach.direction})`}
          accent={approach.classification === "steep"}
        />
        <Stat
          label="Key level"
          value={key_level ? `${key_level.center.toFixed(2)} (×${key_level.touches})` : "—"}
        />
        <Stat
          label="Time at key level"
          value={
            tal.first_tap
              ? "first tap"
              : `${tal.bars_at_level} bars · ${tal.rejection_events} rejections`
          }
          accent={tal.first_tap}
        />
        <Stat
          label="Stair-stepping"
          value={
            stairstep.is_stairstep
              ? `yes — ${stairstep.steps_held_as_support}/${stairstep.steps_broken} held as support`
              : "no"
          }
        />
        <Stat
          label="Momentum candles (recent)"
          value={`${(analysis.momentum_candle_frac_recent * 100).toFixed(0)}%`}
        />
        <Stat label="Close" value={analysis.close.toFixed(2)} />
        <Stat label="ATR (median)" value={analysis.atr.toFixed(2)} />
      </div>
    </div>
  );
}
