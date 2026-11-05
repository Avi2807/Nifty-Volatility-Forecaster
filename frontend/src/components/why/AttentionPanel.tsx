import { useState } from "react";
import type { DayDetails } from "../../api/types";
import { formatCompactDate, formatDate } from "../../lib/format";
import { DataTableDisclosure } from "../charts/DataTableDisclosure";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { AttentionStrip } from "./AttentionStrip";
import styles from "./WhyThisForecast.module.css";

type StripView = "last_day" | "average";

const VIEWS: { id: StripView; label: string }[] = [
  { id: "last_day", label: "Selected day's attention" },
  { id: "average", label: "Average of all 22 days" },
];

export function AttentionPanel({ day }: { day: DayDetails }) {
  const [view, setView] = useState<StripView>("last_day");
  const [expanded, setExpanded] = useState(false);
  const weights = view === "last_day" ? day.attention.last_day_row : day.attention.mean_over_input_days;

  const rows = day.input_dates.map((date, index) => ({ date, weight: weights[index] ?? 0 }));
  const top = [...rows].sort((a, b) => b.weight - a.weight).slice(0, 3);
  const uniform = 1 / rows.length;

  return (
    <div className={styles.block}>
      <h3 className={styles.heading}>Which of the past 22 days the model weighed</h3>
      <div className={styles.views} role="group" aria-label="Attention view">
        {VIEWS.map((item) => (
          <button key={item.id} type="button" className={styles.viewButton} aria-pressed={item.id === view} onClick={() => setView(item.id)}>
            {item.label}
          </button>
        ))}
      </div>
      <AttentionStrip dates={day.input_dates} weights={weights} />
      <p className={styles.readout}>
        Highest weights: {top.map((item) => `${formatCompactDate(item.date)} (${item.weight.toFixed(3)})`).join(", ")}. Equal weights would each be{" "}
        {uniform.toFixed(3)}.
      </p>
      <p className={styles.caution}>
        {view === "last_day"
          ? "This row feeds the forecast directly, because the model reads its output from the last day. "
          : "This averages what each of the 22 days paid attention to, so most of it reaches the forecast only indirectly. "}
        Attention describes how the model mixes days. It doesn't prove which day mattered most.
      </p>
      <DataTableDisclosure
        summary="Show these weights as a table"
        caption="Attention weight for each input day"
        columns={[
          { header: "Input day", value: (row) => formatDate(row.date) },
          { header: "Weight", value: (row) => row.weight.toFixed(4) },
        ]}
        rows={rows}
      />
      <button type="button" className={styles.expand} aria-expanded={expanded} aria-controls="attention-heatmap" onClick={() => setExpanded((current) => !current)}>
        {expanded ? "Hide" : "Show"} the full 22 × 22 heatmap
      </button>
      <div id="attention-heatmap" hidden={!expanded}>
        {expanded && <AttentionHeatmap dates={day.input_dates} matrix={day.attention.matrix} />}
      </div>
    </div>
  );
}
