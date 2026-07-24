import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  IChartApi,
  IPriceLine,
  ISeriesApi,
  ISeriesMarkersPluginApi,
  LineStyle,
  SeriesMarker,
  Time,
  UTCTimestamp,
} from "lightweight-charts";
import type { Analysis, Bar, Level } from "../api";

interface Props {
  bars: Bar[];
  analysis: Analysis;
  selectedCenter?: number | null;
  onSelectLevel?: (level: Level | null) => void;
}

// Top-weighted S/R zones drawn as price lines; the key level gets the accent color.
const MAX_LEVEL_LINES = 8;

const drawnLevels = (analysis: Analysis): Level[] =>
  [...analysis.levels].sort((a, b) => b.weight - a.weight).slice(0, MAX_LEVEL_LINES);

export default function PriceChart({ bars, analysis, selectedCenter = null, onSelectLevel }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const priceLinesRef = useRef<IPriceLine[]>([]);
  const fittedRef = useRef(false);
  // Latest props for the click handler (subscribed once at creation).
  const analysisRef = useRef(analysis);
  const onSelectRef = useRef<(level: Level | null) => void>(() => {});
  analysisRef.current = analysis;
  onSelectRef.current = onSelectLevel ?? (() => {});

  // Create the chart once; refreshes update it in place so zoom/scroll survive.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#12161f" },
        textColor: "#9aa4b5",
      },
      grid: {
        vertLines: { color: "#1c2230" },
        horzLines: { color: "#1c2230" },
      },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: "#2a3244" },
      rightPriceScale: { borderColor: "#2a3244" },
      crosshair: { horzLine: { labelBackgroundColor: "#3d4a63" }, vertLine: { labelBackgroundColor: "#3d4a63" } },
    });
    chartRef.current = chart;
    const series = chart.addSeries(CandlestickSeries, {
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderVisible: false,
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    });
    seriesRef.current = series;
    markersRef.current = createSeriesMarkers(series, []);

    // Click on/near a drawn level line selects it; clicking empty space deselects.
    chart.subscribeClick((param) => {
      if (!param.point) return;
      const price = series.coordinateToPrice(param.point.y);
      if (price === null) {
        onSelectRef.current(null);
        return;
      }
      const a = analysisRef.current;
      const tol = a.atr * 0.5;
      let best: Level | null = null;
      let bestDist = Infinity;
      for (const lvl of drawnLevels(a)) {
        const dist = Math.abs(price - lvl.center);
        if (price >= lvl.lo - tol && price <= lvl.hi + tol && dist < bestDist) {
          best = lvl;
          bestDist = dist;
        }
      }
      onSelectRef.current(best);
    });

    return () => {
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      markersRef.current = null;
      priceLinesRef.current = [];
      fittedRef.current = false;
    };
  }, []);

  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

    series.setData(
      bars.map((b) => ({
        time: b.time as UTCTimestamp,
        open: b.open,
        high: b.high,
        low: b.low,
        close: b.close,
      })),
    );

    for (const line of priceLinesRef.current) series.removePriceLine(line);
    const key = analysis.key_level;
    priceLinesRef.current = drawnLevels(analysis).map((lvl) => {
      const isKey = key !== null && lvl.center === key.center;
      const isSelected = selectedCenter !== null && lvl.center === selectedCenter;
      return series.createPriceLine({
        price: lvl.center,
        color: isKey ? "#f5a623" : isSelected ? "#9aa4b5" : "#3d4a63",
        lineWidth: isKey || isSelected ? 2 : 1,
        lineStyle: isKey || isSelected ? LineStyle.Solid : LineStyle.Dashed,
        axisLabelVisible: isKey || isSelected,
        title: isKey ? `KEY ×${lvl.touches}` : `×${lvl.touches}`,
      });
    });

    // Touch markers for the selected level: when price rejected (H) / held (L) there.
    const selected =
      selectedCenter === null
        ? null
        : drawnLevels(analysis).find((l) => Math.abs(l.center - selectedCenter) <= analysis.atr * 0.5) ?? null;
    const firstBar = bars.length ? bars[0].time : 0;
    const markers: SeriesMarker<Time>[] = (selected?.events ?? [])
      .filter((e) => e.time >= firstBar) // lookback touches before the chart window can't be drawn
      .map((e) => ({
        time: e.time as UTCTimestamp,
        position: e.kind === "H" ? ("aboveBar" as const) : ("belowBar" as const),
        color: "#f5a623",
        shape: e.kind === "H" ? ("arrowDown" as const) : ("arrowUp" as const),
        text: "",
      }));
    markersRef.current?.setMarkers(markers);

    if (!fittedRef.current) {
      chartRef.current?.timeScale().fitContent();
      fittedRef.current = true;
    }
  }, [bars, analysis, selectedCenter]);

  return <div ref={containerRef} className="chart" />;
}
