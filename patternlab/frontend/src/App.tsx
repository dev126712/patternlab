import { useCallback, useEffect, useRef, useState } from "react";
import {
  FetchParams,
  health,
  liveAnalyze,
  LiveAnalyzeResponse,
  liveCompare,
  liveNarrate,
  Match,
  saveLiveSession,
  sessionBars,
  SessionBars,
} from "./api";
import AnalysisPanel from "./components/AnalysisPanel";
import LevelDetail from "./components/LevelDetail";
import MatchList from "./components/MatchList";
import PriceChart from "./components/PriceChart";

const INTERVALS = ["1", "3", "5", "15", "30", "45", "1H", "2H", "4H", "1D"];
const REFRESH_OPTIONS = [
  { value: 0, label: "Off" },
  { value: 5, label: "5s" },
  { value: 15, label: "15s" },
  { value: 30, label: "30s" },
  { value: 60, label: "60s" },
];

export default function App() {
  const [form, setForm] = useState<FetchParams>({
    symbol: "XAUUSD",
    exchange: "OANDA",
    interval: "1",
    n_bars: 1500,
    lookback_bars: 0,
  });
  const [data, setData] = useState<LiveAnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [persistence, setPersistence] = useState(false);
  const [refreshSecs, setRefreshSecs] = useState(15);
  const [narrative, setNarrative] = useState<string | null>(null);
  const [narrativeAt, setNarrativeAt] = useState<Date | null>(null);
  const [narrating, setNarrating] = useState(false);
  const [selectedCenter, setSelectedCenter] = useState<number | null>(null);
  const [compareMatch, setCompareMatch] = useState<Match | null>(null);
  const [compareData, setCompareData] = useState<SessionBars | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareNarrative, setCompareNarrative] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  // Poll with the last params that actually fetched, not the form mid-edit.
  const [activeParams, setActiveParams] = useState<FetchParams | null>(null);
  const inFlight = useRef(false);

  useEffect(() => {
    health().then((h) => setPersistence(h.persistence)).catch(() => {});
  }, []);

  const fetchData = useCallback(async (p: FetchParams, silent: boolean) => {
    if (inFlight.current) return;
    inFlight.current = true;
    if (!silent) {
      setLoading(true);
      setNotice(null);
    }
    try {
      setData(await liveAnalyze(p));
      setActiveParams(p);
      setLastUpdated(new Date());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      inFlight.current = false;
      if (!silent) setLoading(false);
    }
  }, []);

  const analyze = useCallback(() => fetchData(form, false), [fetchData, form]);

  useEffect(() => {
    if (!activeParams || refreshSecs <= 0) return;
    const id = setInterval(() => {
      if (document.hidden) return; // don't poll a background tab
      fetchData(activeParams, true);
    }, refreshSecs * 1000);
    return () => clearInterval(id);
  }, [activeParams, refreshSecs, fetchData]);

  // On-demand only — never wired to the auto-refresh loop (slow, costs LLM tokens).
  const narrate = useCallback(async () => {
    setNarrating(true);
    setError(null);
    try {
      const r = await liveNarrate(activeParams ?? form);
      setNarrative(r.narrative);
      setNarrativeAt(new Date());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setNarrating(false);
    }
  }, [activeParams, form]);

  const selectMatch = useCallback(async (m: Match) => {
    setCompareMatch(m);
    setCompareData(null);
    setCompareNarrative(null);
    setCompareLoading(true);
    setError(null);
    try {
      setCompareData(await sessionBars(m.session_id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCompareLoading(false);
    }
  }, []);

  const closeCompare = useCallback(() => {
    setCompareMatch(null);
    setCompareData(null);
    setCompareNarrative(null);
  }, []);

  // On-demand LLM comparison of today vs the selected match — never auto-refreshed.
  const narrateCompare = useCallback(async () => {
    if (!compareMatch) return;
    setComparing(true);
    setError(null);
    try {
      const r = await liveCompare(compareMatch.session_id, activeParams ?? form);
      setCompareNarrative(r.narrative ?? "");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setComparing(false);
    }
  }, [compareMatch, activeParams, form]);

  const save = useCallback(async () => {
    setSaving(true);
    setError(null);
    try {
      const r = await saveLiveSession(form);
      setNotice(`Saved session #${r.session_id} (${r.instrument} ${r.session_date}, ${r.n_bars} bars)`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }, [form]);

  const set = (k: keyof FetchParams) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setForm((f) => ({
      ...f,
      [k]: k === "n_bars" || k === "lookback_bars" ? Number(e.target.value) : e.target.value.toUpperCase(),
    }));

  return (
    <div className="app">
      <header>
        <h1>
          Pattern<span className="accent">Lab</span>
        </h1>
        <span className="tagline">structural chart analysis · fractal rhymes</span>
        {lastUpdated && (
          <span className="live-status">
            {refreshSecs > 0 && <span className="live-dot" />}
            updated {lastUpdated.toLocaleTimeString()}
            {data && ` · ${data.source}`}
          </span>
        )}
      </header>

      <div className="controls">
        <label>
          Exchange
          <input value={form.exchange} onChange={set("exchange")} placeholder="OANDA" />
        </label>
        <label>
          Symbol
          <input value={form.symbol} onChange={set("symbol")} placeholder="XAUUSD" />
        </label>
        <label>
          Interval
          <select value={form.interval} onChange={set("interval")}>
            {INTERVALS.map((i) => (
              <option key={i} value={i}>
                {i}
              </option>
            ))}
          </select>
        </label>
        <label>
          Bars
          <input type="number" min={100} max={5000} step={100} value={form.n_bars} onChange={set("n_bars")} />
        </label>
        <label title="Detect S/R levels over this many bars while charting only 'Bars'. 0 = same window.">
          Lookback
          <input type="number" min={0} max={5000} step={100} value={form.lookback_bars}
                 onChange={set("lookback_bars")} placeholder="0" />
        </label>
        <label>
          Refresh
          <select value={refreshSecs} onChange={(e) => setRefreshSecs(Number(e.target.value))}>
            {REFRESH_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <button onClick={analyze} disabled={loading}>
          {loading ? "Fetching…" : "Analyze"}
        </button>
        <button onClick={save} disabled={saving || !persistence} className="secondary"
                title={persistence ? "Persist this session for future matching" : "Requires DATABASE_URL"}>
          {saving ? "Saving…" : "Save session"}
        </button>
        <button onClick={narrate} disabled={narrating || !data} className="secondary"
                title="LLM read of the current session (on demand — not auto-refreshed)">
          {narrating ? "Narrating…" : "Narrate"}
        </button>
      </div>

      {error && <div className="banner error">{error}</div>}
      {notice && <div className="banner notice">{notice}</div>}

      {data ? (
        <>
          <PriceChart
            key={`${data.exchange}:${data.symbol}:${data.interval}`}
            bars={data.bars}
            analysis={data.analysis}
            selectedCenter={selectedCenter}
            onSelectLevel={(lvl) => setSelectedCenter(lvl ? lvl.center : null)}
          />
          {selectedCenter !== null &&
            (() => {
              // Re-resolve on every refresh: centers drift slightly as new bars arrive.
              const lvl = data.analysis.levels.find(
                (l) => Math.abs(l.center - selectedCenter) <= data.analysis.atr * 0.5,
              );
              return lvl ? (
                <LevelDetail
                  level={lvl}
                  analysis={data.analysis}
                  firstBarTime={data.bars.length ? data.bars[0].time : 0}
                  onClose={() => setSelectedCenter(null)}
                />
              ) : null;
            })()}
          {narrative && (
            <div className="narrative">
              <div className="narrative-head">
                <span>LLM read</span>
                {narrativeAt && <span className="narrative-time">as of {narrativeAt.toLocaleTimeString()}</span>}
              </div>
              {narrative}
            </div>
          )}
          <div className="panels">
            <AnalysisPanel analysis={data.analysis} />
            <MatchList
              matches={data.matches}
              selectedId={compareMatch?.session_id ?? null}
              onSelect={selectMatch}
            />
          </div>
          {compareMatch && (
            <div className="compare">
              <div className="compare-head">
                <span className="compare-title">
                  Today vs {compareMatch.session_date}
                </span>
                <span className="compare-sub">
                  score {compareMatch.similarity_score.toFixed(3)} · fingerprint{" "}
                  {compareMatch.fingerprint_cosine.toFixed(3)} · shape {compareMatch.dtw_similarity.toFixed(3)}
                </span>
                <button className="secondary" onClick={narrateCompare} disabled={comparing}
                        title="LLM narrative comparing today's session with this historical day">
                  {comparing ? "Narrating…" : "Narrate comparison"}
                </button>
                <button className="level-detail-close" onClick={closeCompare} title="Close">
                  ×
                </button>
              </div>
              {compareLoading && <p className="muted">Loading {compareMatch.session_date} bars from OANDA…</p>}
              {compareData && (
                <PriceChart
                  key={`cmp:${compareData.session_id}`}
                  bars={compareData.bars}
                  analysis={compareData.analysis}
                />
              )}
              {compareNarrative && (
                <div className="narrative">
                  <div className="narrative-head">
                    <span>LLM comparison</span>
                    <span className="narrative-time">today vs {compareMatch.session_date}</span>
                  </div>
                  {compareNarrative}
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        !loading && (
          <div className="empty">
            Pick an instrument and hit <b>Analyze</b> — data is fetched live from TradingView.
          </div>
        )
      )}
    </div>
  );
}
