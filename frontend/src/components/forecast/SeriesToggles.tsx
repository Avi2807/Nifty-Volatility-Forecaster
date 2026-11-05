import { SERIES_TOKENS, MODEL_LABELS } from "../../lib/models";
import { KEY_BORDER_STYLE, type ChartLineStyle } from "../charts/TimeSeriesChart";
import styles from "./ForecastVsActual.module.css";

export type ChartSeriesId = "actual" | "transformer" | "transformer_nosent" | "har" | "naive" | "vix_implied" | "garch" | "mlp" | "lstm";

export const CHART_SERIES: { id: ChartSeriesId; label: string; colorToken: string; lineWidth: 1 | 2 | 3; lineStyle: ChartLineStyle }[] = [
  { id: "actual", label: "Actual", colorToken: SERIES_TOKENS.actual, lineWidth: 1, lineStyle: "solid" },
  { id: "transformer", label: MODEL_LABELS.transformer, colorToken: SERIES_TOKENS.transformer, lineWidth: 2, lineStyle: "solid" },
  { id: "transformer_nosent", label: MODEL_LABELS.transformer_nosent, colorToken: SERIES_TOKENS.transformer_nosent, lineWidth: 2, lineStyle: "dashed" },
  { id: "har", label: MODEL_LABELS.har, colorToken: SERIES_TOKENS.har, lineWidth: 1, lineStyle: "dashed" },
  { id: "naive", label: MODEL_LABELS.naive, colorToken: SERIES_TOKENS.naive, lineWidth: 1, lineStyle: "dotted" },
  { id: "vix_implied", label: MODEL_LABELS.vix_implied, colorToken: SERIES_TOKENS.vix_implied, lineWidth: 2, lineStyle: "dotted" },
  { id: "garch", label: MODEL_LABELS.garch, colorToken: SERIES_TOKENS.garch, lineWidth: 2, lineStyle: "long-dash" },
  { id: "mlp", label: MODEL_LABELS.mlp, colorToken: SERIES_TOKENS.mlp, lineWidth: 1, lineStyle: "long-dash" },
  { id: "lstm", label: MODEL_LABELS.lstm, colorToken: SERIES_TOKENS.lstm, lineWidth: 3, lineStyle: "dotted" },
];

interface SeriesTogglesProps {
  hidden: ReadonlySet<ChartSeriesId>;
  /** Series with no forecasts in the chosen range, which can't be switched on. */
  unavailable: ReadonlySet<ChartSeriesId>;
  onToggle: (id: ChartSeriesId) => void;
}

/** These double as the chart's legend, so each shows its line colour next to the name. */
export function SeriesToggles({ hidden, unavailable, onToggle }: SeriesTogglesProps) {
  return (
    <div className={styles.group} role="group" aria-label="Series shown on the chart">
      {CHART_SERIES.map((series) => (
        <button
          key={series.id}
          type="button"
          className={styles.chip}
          aria-pressed={!hidden.has(series.id) && !unavailable.has(series.id)}
          disabled={unavailable.has(series.id)}
          onClick={() => onToggle(series.id)}
        >
          <span className={styles.swatch} style={{ borderTopColor: `var(${series.colorToken})`, borderTopStyle: KEY_BORDER_STYLE[series.lineStyle] }} aria-hidden="true" />
          {series.label}
        </button>
      ))}
    </div>
  );
}
