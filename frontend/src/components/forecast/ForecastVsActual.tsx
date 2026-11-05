import { useEffect, useMemo, useState } from "react";
import { formatDate, formatPct } from "../../lib/format";
import type { ReplayRow } from "../../lib/replay";
import { logRvToAnnualVolPct } from "../../lib/volatility";
import { TimeSeriesChart, type ChartSeriesSpec } from "../charts/TimeSeriesChart";
import { DataTableDisclosure, type TableColumn } from "../charts/DataTableDisclosure";
import { EmptyState } from "../feedback/EmptyState";
import { Section } from "../layout/Section";
import { RANGE_PRESETS, RangePresets, rowsInPreset, type PresetId } from "./RangePresets";
import { CHART_SERIES, SeriesToggles, type ChartSeriesId } from "./SeriesToggles";
import styles from "./ForecastVsActual.module.css";

/** Later series draw on top, so the two lines being compared stay visible. */
const DRAW_ORDER: ChartSeriesId[] = ["naive", "vix_implied", "garch", "mlp", "lstm", "har", "actual", "transformer_nosent", "transformer"];

const valueOf = (row: ReplayRow, id: ChartSeriesId) => (id === "actual" ? row.actual_log_rv : row[id]);

const COLUMNS: TableColumn<ReplayRow>[] = [
  { header: "Session", value: (row) => formatDate(row.target_date) },
  ...CHART_SERIES.map(
    (series): TableColumn<ReplayRow> => ({
      header: series.label,
      value: (row) => {
        const value = valueOf(row, series.id);
        return value === null ? "n/a" : formatPct(logRvToAnnualVolPct(value), 1);
      },
    }),
  ),
];

interface ForecastVsActualProps {
  rows: ReplayRow[];
  selectedDate: string;
  onSelect: (date: string) => void;
}

export function ForecastVsActual({ rows, selectedDate, onSelect }: ForecastVsActualProps) {
  const [preset, setPreset] = useState<PresetId>("full");
  // Naive is Actual a day late, and hides it. The no-news line sits almost on the Transformer's in the test period.
  // GARCH, the MLP and the LSTM start off too, because nine lines at once can't be read.
  const [hidden, setHidden] = useState<ReadonlySet<ChartSeriesId>>(new Set(["naive", "transformer_nosent", "garch", "mlp", "lstm"]));

  // The chart follows the selected day across the two periods, and shows that period's own Transformer.
  const selectedPeriod = rows.find((row) => row.date === selectedDate)?.period;
  useEffect(() => {
    if (!selectedPeriod) return;
    setPreset((current) => (RANGE_PRESETS.find((item) => item.id === current)?.period === selectedPeriod ? current : selectedPeriod === "later" ? "later" : "full"));
  }, [selectedPeriod]);
  const presetPeriod = RANGE_PRESETS.find((item) => item.id === preset)?.period;
  useEffect(() => {
    setHidden((current) => {
      const next = new Set(current);
      if (presetPeriod === "later") next.delete("transformer_nosent");
      else next.add("transformer_nosent");
      return next;
    });
  }, [presetPeriod]);

  const visibleRows = useMemo(() => rowsInPreset(rows, preset), [rows, preset]);
  const unavailable = useMemo(
    () => new Set(CHART_SERIES.filter((item) => visibleRows.every((row) => valueOf(row, item.id) === null)).map((item) => item.id)),
    [visibleRows],
  );
  const series = useMemo<ChartSeriesSpec[]>(
    () =>
      DRAW_ORDER.filter((id) => !hidden.has(id) && !unavailable.has(id)).flatMap((id) => {
        const spec = CHART_SERIES.find((item) => item.id === id);
        if (!spec) return [];
        const points = visibleRows.flatMap((row) => {
          const value = valueOf(row, id);
          return value === null ? [] : [{ time: row.target_date, value: logRvToAnnualVolPct(value) }];
        });
        return [{ id, label: spec.label, colorToken: spec.colorToken, lineWidth: spec.lineWidth, lineStyle: spec.lineStyle, kind: "line" as const, points }];
      }),
    [visibleRows, hidden, unavailable],
  );
  const selectedTarget = rows.find((row) => row.date === selectedDate)?.target_date;

  const toggle = (id: ChartSeriesId) =>
    setHidden((current) => {
      const next = new Set(current);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  const selectByTarget = (time: string) => {
    const match = rows.find((row) => row.target_date === time);
    if (match) onSelect(match.date);
  };

  return (
    <Section
      id="forecast-vs-actual"
      kicker="Forecast vs actual"
      title="What was called, and what came"
      standfirst="Annualised volatility for each session, as forecast the evening before and as measured afterwards."
    >
      <div className={styles.controls}>
        <RangePresets value={preset} hasLater={rows.some((row) => row.period === "later")} onChange={setPreset} />
        <SeriesToggles hidden={hidden} unavailable={unavailable} onToggle={toggle} />
      </div>
      {series.length === 0 ? (
        <EmptyState>All series are switched off. Turn one on above to see the chart.</EmptyState>
      ) : (
        <TimeSeriesChart
          series={series}
          label={`Line chart of annualised volatility for ${series.map((item) => item.label).join(", ")}, ${RANGE_PRESETS.find((item) => item.id === preset)?.label}`}
          formatValue={(value) => formatPct(value, 1)}
          onSelectTime={selectByTarget}
          selectedTime={selectedTarget}
          fitKey={`${preset}-${series.length > 0}`}
        />
      )}
      <p className={styles.note}>
        {presetPeriod === "later"
          ? "Click the chart to replay that day. This is a separate check, not part of the test results. The with-news Transformer isn't here because the news archive had ended, and GARCH, the MLP and the LSTM were only scored on the test period."
          : "Click the chart to replay that day. The 2020 COVID crash isn't here because it was the validation period, so the model had already seen it."}
      </p>
      <DataTableDisclosure summary="Show these values as a table" caption="Annualised volatility by session" columns={COLUMNS} rows={visibleRows} />
    </Section>
  );
}
