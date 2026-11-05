import type { ReplayRow } from "../../lib/replay";
import styles from "./ForecastVsActual.module.css";

export const RANGE_PRESETS = [
  { id: "full", label: "Full test", period: "test", start: "", end: "9999-12-31" },
  { id: "2021", label: "2021", period: "test", start: "2021-01-01", end: "2021-12-31" },
  { id: "2022h1", label: "2022 H1", period: "test", start: "2022-01-01", end: "2022-06-30" },
  { id: "jun2022", label: "Jun 2022", period: "test", start: "2022-06-01", end: "2022-06-30" },
  { id: "later", label: "Later check", period: "later", start: "", end: "9999-12-31" },
] as const;

export type PresetId = (typeof RANGE_PRESETS)[number]["id"];

/** Presets filter on the session a forecast was for, which is the date the chart draws. No preset mixes the two periods. */
export function rowsInPreset(rows: ReplayRow[], id: PresetId): ReplayRow[] {
  const preset = RANGE_PRESETS.find((item) => item.id === id);
  return preset ? rows.filter((row) => row.period === preset.period && row.target_date >= preset.start && row.target_date <= preset.end) : rows;
}

export function RangePresets({ value, hasLater, onChange }: { value: PresetId; hasLater: boolean; onChange: (id: PresetId) => void }) {
  return (
    <div className={styles.group} role="group" aria-label="Date range">
      {RANGE_PRESETS.filter((preset) => hasLater || preset.period === "test").map((preset) => (
        <button key={preset.id} type="button" className={styles.chip} aria-pressed={preset.id === value} onClick={() => onChange(preset.id)}>
          {preset.label}
        </button>
      ))}
    </div>
  );
}
