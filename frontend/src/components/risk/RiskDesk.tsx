import { useMemo, useState } from "react";
import { useValueAtRisk } from "../../api/queries";
import type { ModelName, VarPoint } from "../../api/types";
import { formatDate, formatPct, formatSignedPct } from "../../lib/format";
import { ALL_MODELS, MODEL_LABELS } from "../../lib/models";
import type { ReplayRow } from "../../lib/replay";
import { DataTableDisclosure, type TableColumn } from "../charts/DataTableDisclosure";
import { TimeSeriesChart, type ChartSeriesSpec } from "../charts/TimeSeriesChart";
import { EmptyState } from "../feedback/EmptyState";
import { ErrorState } from "../feedback/ErrorState";
import { Skeleton } from "../feedback/Skeleton";
import { Section } from "../layout/Section";
import { KupiecPanel } from "./KupiecPanel";
import styles from "./RiskDesk.module.css";

const COLUMNS: TableColumn<VarPoint>[] = [
  { header: "Session", value: (point) => formatDate(point.target_date) },
  { header: "Open-to-close return", value: (point) => formatSignedPct(point.return_oc * 100) },
  { header: "99% VaR (loss)", value: (point) => formatPct(point.var99 * 100) },
  { header: "Breach", value: (point) => (point.breach ? "Yes" : "No") },
];

interface RiskDeskProps {
  rows: ReplayRow[];
  selectedDate: string;
  onSelect: (date: string) => void;
}

export function RiskDesk({ rows, selectedDate, onSelect }: RiskDeskProps) {
  const [model, setModel] = useState<ModelName>("transformer");
  const { data, error, isPending, refetch } = useValueAtRisk(model);

  const series = useMemo<ChartSeriesSpec[]>(() => {
    if (!data) return [];
    return [
      {
        id: "returns",
        label: "Open-to-close return",
        kind: "histogram",
        colorToken: "--chart-bar",
        points: data.series.map((point) => ({
          time: point.target_date,
          value: point.return_oc * 100,
          colorToken: point.breach ? "--chart-breach" : undefined,
        })),
      },
      {
        id: "var",
        label: "99% VaR limit",
        kind: "line",
        colorToken: "--chart-var-line",
        lineWidth: 2,
        points: data.series.map((point) => ({ time: point.target_date, value: -point.var99 * 100 })),
      },
    ];
  }, [data]);

  const breachTimes = useMemo(() => data?.series.filter((point) => point.breach).map((point) => point.target_date) ?? [], [data]);

  const selectedRow = rows.find((row) => row.date === selectedDate);
  const selectedTarget = selectedRow?.target_date;
  const selectByTarget = (time: string) => {
    const match = rows.find((row) => row.target_date === time);
    if (match) onSelect(match.date);
  };

  return (
    <Section
      id="risk-desk"
      kicker="Risk desk"
      title="How often the 99% line broke"
      standfirst="Each bar is one session's open-to-close return, and a loss past the line set the evening before is a breach, marked with a tick under the axis."
    >
      {selectedRow?.period === "later" && (
        <p className={styles.periodNote}>This desk covers the test period, so {formatDate(selectedRow.target_date)} isn't on it. The later check's breach counts are on the Scoreboard.</p>
      )}
      <label className={styles.modelPicker}>
        <span>Model</span>
        <select className={styles.select} value={model} onChange={(event) => setModel(event.target.value as ModelName)}>
          {ALL_MODELS.map((name) => (
            <option key={name} value={name}>
              {MODEL_LABELS[name]}
            </option>
          ))}
        </select>
      </label>
      {error && !data && <ErrorState error={error} onRetry={() => void refetch()} />}
      {isPending && <Skeleton size="chart" label="Loading the VaR series" />}
      {data && data.series.length === 0 && <EmptyState>The backend returned no VaR series for this model.</EmptyState>}
      {data && data.series.length > 0 && (
        <div className={styles.layout}>
          <div className={styles.chart}>
            <TimeSeriesChart
              series={series}
              label={`Bar chart of daily open-to-close returns with the 99% VaR line for ${MODEL_LABELS[data.model]}`}
              formatValue={(value) => formatSignedPct(value)}
              onSelectTime={selectByTarget}
              selectedTime={selectedTarget}
              fitKey={data.model}
              tickTimes={breachTimes}
            />
            <ul className={styles.legend} aria-label="Chart legend">
              <li className={styles.legendItem}>
                <span className={styles.barKey} aria-hidden="true" />
                Return
              </li>
              <li className={styles.legendItem}>
                <span className={styles.breachKey} aria-hidden="true" />
                Breach
              </li>
              <li className={styles.legendItem}>
                <span className={styles.lineKey} aria-hidden="true" />
                99% VaR limit
              </li>
            </ul>
          </div>
          <KupiecPanel response={data} />
        </div>
      )}
      {data && (
        <DataTableDisclosure summary="Show these values as a table" caption={`VaR series for ${MODEL_LABELS[data.model]}`} columns={COLUMNS} rows={data.series} />
      )}
    </Section>
  );
}
