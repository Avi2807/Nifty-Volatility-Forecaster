import {
  ColorType,
  createChart,
  createSeriesMarkers,
  CrosshairMode,
  HistogramSeries,
  LineSeries,
  LineStyle,
  TickMarkType,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { useTokenReader } from "../../hooks/useTokens";
import { formatCompactDate, formatDate } from "../../lib/format";
import styles from "./TimeSeriesChart.module.css";

export interface ChartPoint {
  time: string;
  value: number;
  colorToken?: string;
}

export type ChartLineStyle = "solid" | "dashed" | "long-dash" | "dotted";

/** The CSS border style that stands for a line style in a legend key. CSS has no long dash. */
export const KEY_BORDER_STYLE: Record<ChartLineStyle, "solid" | "dashed" | "dotted"> = { solid: "solid", dashed: "dashed", "long-dash": "dashed", dotted: "dotted" };

export interface ChartSeriesSpec {
  id: string;
  label: string;
  kind: "line" | "histogram";
  colorToken: string;
  points: ChartPoint[];
  lineWidth?: 1 | 2 | 3;
  /** Reference lines share greys, so the dash tells them apart. */
  lineStyle?: ChartLineStyle;
}

interface TimeSeriesChartProps {
  series: ChartSeriesSpec[];
  label: string;
  formatValue: (value: number) => string;
  /** Times here are the `time` of each point. Clicking the chart reports one. */
  onSelectTime?: (time: string) => void;
  selectedTime?: string;
  /** The view refits when this changes, for example when a range preset is chosen. */
  fitKey?: string;
  /** Times marked with a short tick under the axis, like a printed annotation. */
  tickTimes?: string[];
}

interface TooltipState {
  x: number;
  y: number;
  time: string;
  rows: { id: string; label: string; colorToken: string; lineStyle: ChartLineStyle; value: string; raw: number }[];
}

interface MountedSeries {
  spec: ChartSeriesSpec;
  api: ISeriesApi<"Line"> | ISeriesApi<"Histogram">;
  markers: ISeriesMarkersPluginApi<Time>;
}

const TOOLTIP_WIDTH = 220;
const LINE_STYLES: Record<ChartLineStyle, LineStyle> = { solid: LineStyle.Solid, dashed: LineStyle.Dashed, "long-dash": LineStyle.LargeDashed, dotted: LineStyle.SparseDotted };

function asIsoDate(time: Time | undefined): string | null {
  if (typeof time === "string") return time;
  if (time && typeof time === "object") {
    const pad = (value: number) => String(value).padStart(2, "0");
    return `${time.year}-${pad(time.month)}-${pad(time.day)}`;
  }
  return null;
}

/** A thin React wrapper around lightweight-charts. Colours are token names, resolved at draw time. */
export function TimeSeriesChart({ series, label, formatValue, onSelectTime, selectedTime, fitKey, tickTimes }: TimeSeriesChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mountedRef = useRef(new Map<string, MountedSeries>());
  const readToken = useTokenReader();
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const [ticks, setTicks] = useState<{ time: string; x: number }[]>([]);

  // Handlers read the latest props through refs, so the chart isn't rebuilt when they change.
  const latest = useRef({ formatValue, onSelectTime });
  latest.current = { formatValue, onSelectTime };

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const mounted = mountedRef.current;
    const chart = createChart(container, {
      autoSize: true,
      crosshair: { mode: CrosshairMode.Magnet },
      handleScroll: { vertTouchDrag: false }, // a vertical swipe scrolls the page, not the chart
      localization: { priceFormatter: (value: number) => latest.current.formatValue(value) },
      timeScale: {
        rightOffset: 6,
        // The library labels some ticks with a bare day number, which reads as a stray "7" among months.
        tickMarkFormatter: (time: Time, type: TickMarkType) => {
          const date = asIsoDate(time);
          return date && type === TickMarkType.DayOfMonth ? formatCompactDate(date) : null;
        },
      },
    });
    chartRef.current = chart;

    const onMove = (param: MouseEventParams<Time>) => {
      const time = asIsoDate(param.time);
      if (!time || !param.point) {
        setTooltip(null);
        return;
      }
      const rows: TooltipState["rows"] = [];
      for (const { api, spec } of mounted.values()) {
        const datum = param.seriesData.get(api);
        if (datum && "value" in datum) {
          rows.push({
            id: spec.id,
            label: spec.label,
            colorToken: spec.colorToken,
            lineStyle: spec.lineStyle ?? "solid",
            value: latest.current.formatValue(datum.value),
            raw: datum.value,
          });
        }
      }
      rows.sort((a, b) => b.raw - a.raw); // top line first, matching what's on the chart
      const fitsRight = param.point.x + TOOLTIP_WIDTH + 24 < container.clientWidth;
      const x = fitsRight ? param.point.x + 16 : Math.max(0, param.point.x - TOOLTIP_WIDTH - 16);
      setTooltip({ x, y: 8, time, rows });
    };
    const onClick = (param: MouseEventParams<Time>) => {
      const time = asIsoDate(param.time);
      if (time) latest.current.onSelectTime?.(time);
    };
    chart.subscribeCrosshairMove(onMove);
    chart.subscribeClick(onClick);

    return () => {
      chart.unsubscribeCrosshairMove(onMove);
      chart.unsubscribeClick(onClick);
      chart.remove();
      chartRef.current = null;
      mounted.clear();
    };
  }, []);

  useEffect(() => {
    const crosshairLine = { color: readToken("--color-axis"), labelBackgroundColor: readToken("--color-text") };
    chartRef.current?.applyOptions({
      layout: {
        background: { type: ColorType.Solid, color: "transparent" }, // the paper and the column rules show through
        textColor: readToken("--color-text-secondary"),
        fontFamily: readToken("--font-mono"),
        fontSize: 11,
        attributionLogo: true, // the chart library's licence asks for this link
      },
      crosshair: { vertLine: crosshairLine, horzLine: crosshairLine },
      grid: { vertLines: { visible: false }, horzLines: { color: readToken("--color-grid") } },
      rightPriceScale: { borderColor: readToken("--color-rule"), scaleMargins: { top: 0.06, bottom: 0.04 }, entireTextOnly: true },
      timeScale: { borderColor: readToken("--color-rule") },
    });
  }, [readToken]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const mounted = mountedRef.current;
    const wanted = new Set(series.map((spec) => spec.id));
    for (const [id, entry] of mounted) {
      if (!wanted.has(id)) {
        chart.removeSeries(entry.api);
        mounted.delete(id);
      }
    }

    series.forEach((spec, order) => {
      const color = readToken(spec.colorToken);
      let entry = mounted.get(spec.id);
      if (!entry) {
        const shared = { priceLineVisible: false, lastValueVisible: false };
        const api =
          spec.kind === "line"
            ? chart.addSeries(LineSeries, { ...shared, crosshairMarkerRadius: 4 })
            : chart.addSeries(HistogramSeries, shared);
        entry = { spec, api, markers: createSeriesMarkers(api as ISeriesApi<"Line">, []) };
        mounted.set(spec.id, entry);
      }
      entry.spec = spec;
      entry.api.setSeriesOrder(order); // a series switched back on keeps its place in the stack
      if (spec.kind === "line") {
        (entry.api as ISeriesApi<"Line">).applyOptions({ color, lineWidth: spec.lineWidth ?? 2, lineStyle: LINE_STYLES[spec.lineStyle ?? "solid"] });
        entry.api.setData(spec.points.map(({ time, value }) => ({ time, value })));
      } else {
        (entry.api as ISeriesApi<"Histogram">).applyOptions({ color });
        (entry.api as ISeriesApi<"Histogram">).setData(
          spec.points.map(({ time, value, colorToken }) => ({
            time,
            value,
            color: colorToken ? readToken(colorToken) : color,
          })),
        );
      }

      const times = new Set(spec.points.map((point) => point.time));
      const markers: SeriesMarker<Time>[] = [];
      if (selectedTime && times.has(selectedTime) && spec.id === series[0]?.id) {
        markers.push({
          time: selectedTime,
          position: "aboveBar",
          shape: "arrowDown",
          color: readToken("--chart-selected"),
          text: "Selected",
        });
      }
      entry.markers.setMarkers(markers);
    });
  }, [series, selectedTime, readToken]);

  useEffect(() => {
    chartRef.current?.timeScale().fitContent();
  }, [fitKey]);

  // The ticks live outside the canvas, so they have to follow the chart when it pans, zooms or resizes.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !tickTimes) return;
    const scale = chart.timeScale();
    const place = () => {
      const width = scale.width();
      setTicks(
        tickTimes.flatMap((time) => {
          const x = scale.timeToCoordinate(time);
          return x !== null && x >= 0 && x <= width ? [{ time, x }] : [];
        }),
      );
    };
    place();
    scale.subscribeVisibleLogicalRangeChange(place);
    scale.subscribeSizeChange(place);
    return () => {
      scale.unsubscribeVisibleLogicalRangeChange(place);
      scale.unsubscribeSizeChange(place);
    };
  }, [tickTimes, series]);

  return (
    <div className={styles.frame}>
      <div
        ref={containerRef}
        className={styles.canvas}
        role="group" // not "img": the chart library's attribution link inside must stay focusable
        aria-label={`${label}. The same values are in the data table below the chart.`}
        onMouseLeave={() => setTooltip(null)}
      />
      {tickTimes && (
        <div className={styles.ticks} aria-hidden="true">
          {ticks.map((tick) => (
            <span key={tick.time} className={styles.tick} style={{ left: tick.x }} />
          ))}
        </div>
      )}
      {tooltip && (
        <div className={styles.tooltip} style={{ left: tooltip.x, top: tooltip.y, width: TOOLTIP_WIDTH }} aria-hidden="true">
          <p className={styles.tooltipDate}>{formatDate(tooltip.time)}</p>
          <ul>
            {tooltip.rows.map((row) => (
              <li key={row.id} className={styles.tooltipRow}>
                <span className={styles.tooltipKey} style={{ borderTopColor: `var(${row.colorToken})`, borderTopStyle: KEY_BORDER_STYLE[row.lineStyle] }} />
                <strong>{row.value}</strong>
                <span className={styles.tooltipLabel}>{row.label}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
