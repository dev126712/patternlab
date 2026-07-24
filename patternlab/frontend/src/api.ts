// Typed client for the PatternLab API (paths proxied by vite dev server / nginx).

export interface Bar {
  time: number; // unix seconds UTC
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface LevelEvent {
  time: number; // unix seconds UTC
  price: number;
  kind: "H" | "L"; // swing high (rejected as resistance) | swing low (held as support)
}

export interface Level {
  lo: number;
  hi: number;
  center: number;
  touches: number;
  weight: number;
  events?: LevelEvent[]; // absent on sessions stored before this field existed
}

export interface Regime {
  start: number;
  end: number;
  bars: number;
  kind: "consolidation" | "impulse_up" | "impulse_down" | "mixed";
  net_move_atr: number;
  er: number;
  momentum_candle_frac: number;
}

export interface Analysis {
  n_bars: number;
  close: number;
  atr: number;
  levels: Level[];
  key_level: Level | null;
  regimes: Regime[];
  context: {
    kind: string;
    rally_origin_price: number;
    rally_origin_idx: number;
    lookback_high: number;
    lookback_low: number;
    origin_drawdown_frac: number;
    overhead_to_lookback_high_atr: number;
    trapped_supply_overhead: boolean;
  };
  stairstep: { steps_broken: number; steps_held_as_support: number; is_stairstep: boolean };
  approach: {
    slope_atr_per_bar: number;
    direction: string;
    classification: string;
    r2: number;
    lookback_bars: number;
  };
  time_at_key_level: { bars_at_level: number; rejection_events: number; first_tap: boolean };
  momentum_candle_frac_recent: number;
}

export interface Match {
  session_id: number;
  instrument: string;
  session_date: string;
  similarity_score: number;
  fingerprint_cosine: number;
  dtw_similarity: number;
  analysis: Analysis;
}

export interface LiveAnalyzeResponse {
  symbol: string;
  exchange: string;
  interval: string;
  source: "tradingview" | "oanda";
  bars: Bar[];
  analysis: Analysis;
  matches: Match[] | null;
}

export interface FetchParams {
  symbol: string;
  exchange: string;
  interval: string;
  n_bars: number;
  lookback_bars: number; // 0 = same window as n_bars
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not json */
    }
    throw new Error(detail);
  }
  return res.json();
}

const qs = (p: FetchParams) =>
  new URLSearchParams({
    symbol: p.symbol,
    exchange: p.exchange,
    interval: p.interval,
    n_bars: String(p.n_bars),
    ...(p.lookback_bars > p.n_bars ? { lookback_bars: String(p.lookback_bars) } : {}),
  }).toString();

export const liveAnalyze = (p: FetchParams) =>
  request<LiveAnalyzeResponse>(`/live/analyze?${qs(p)}`);

export const saveLiveSession = (p: FetchParams) =>
  request<{ session_id: number; instrument: string; session_date: string; n_bars: number }>(
    `/live/sessions?${qs(p)}`,
    { method: "POST" },
  );

export interface SessionBars {
  session_id: number;
  instrument: string;
  session_date: string;
  bars: Bar[];
  analysis: Analysis;
}

export const sessionBars = (id: number) => request<SessionBars>(`/sessions/${id}/bars`);

export interface CompareResult {
  session_id: number;
  session_date: string;
  instrument: string;
  similarity_score: number;
  fingerprint_cosine: number;
  dtw_similarity: number;
  narrative: string | null;
}

export const liveCompare = (id: number, p: FetchParams) =>
  request<CompareResult>(`/live/compare?session_id=${id}&${qs(p)}&narrate=true`);

export const liveNarrate = (p: FetchParams) =>
  request<{ narrative: string; llm_provider: string; as_of: number | null }>(
    `/live/narrate?${qs(p)}`,
  );

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export const liveChat = (messages: ChatMessage[], p: FetchParams) =>
  request<{ reply: string; llm_provider: string; as_of: number | null }>("/live/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      symbol: p.symbol,
      exchange: p.exchange,
      interval: p.interval,
      n_bars: p.n_bars,
      ...(p.lookback_bars > p.n_bars ? { lookback_bars: p.lookback_bars } : {}),
    }),
  });

export const health = () =>
  request<{ status: string; llm_provider: string; persistence: boolean }>("/health");
