import { scaleBand } from "d3";
import { useState } from "react";
import { useTokenReader } from "../../hooks/useTokens";
import { formatCompactDate } from "../../lib/format";
import { DataTableDisclosure } from "../charts/DataTableDisclosure";
import { attentionColorScale } from "./attentionScale";
import styles from "./WhyThisForecast.module.css";

const SIZE = 440;
const MARGIN = { top: 8, right: 8, bottom: 56, left: 64 };
const AXIS_EVERY = 7;

interface HoveredCell {
  from: string;
  to: string;
  weight: number;
}

export function AttentionHeatmap({ dates, matrix }: { dates: string[]; matrix: number[][] }) {
  const readToken = useTokenReader();
  const [hovered, setHovered] = useState<HoveredCell | null>(null);

  const values = matrix.flat();
  const low = readToken("--heat-low");
  const mid = readToken("--heat-mid");
  const high = readToken("--heat-high");
  const color = attentionColorScale(values, [low, mid, high]);
  const band = scaleBand<string>().domain(dates).range([0, SIZE - MARGIN.left - MARGIN.right]).paddingInner(0.04);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const ticks = dates.filter((_, index) => (dates.length - 1 - index) % AXIS_EVERY === 0);
  const half = band.bandwidth() / 2;

  return (
    <div className={styles.heatmap}>
      <svg
        className={styles.heatmapSvg}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        role="img"
        aria-label={`Attention between the 22 input days. Weights run from ${min.toFixed(3)} to ${max.toFixed(3)}. The same values are in the table below.`}
        onPointerLeave={() => setHovered(null)}
      >
        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {matrix.map((row, rowIndex) =>
            row.map((weight, columnIndex) => {
              const from = dates[rowIndex] ?? "";
              const to = dates[columnIndex] ?? "";
              return (
                <rect
                  key={`${rowIndex}-${columnIndex}`}
                  x={band(to)}
                  y={band(from)}
                  width={band.bandwidth()}
                  height={band.bandwidth()}
                  fill={color(weight)}
                  onPointerEnter={() => setHovered({ from, to, weight })}
                />
              );
            }),
          )}
          {ticks.map((date) => (
            <g key={date}>
              <text className={styles.axisLabel} x={-6} y={(band(date) ?? 0) + half} textAnchor="end" dominantBaseline="middle">
                {formatCompactDate(date)}
              </text>
              <text
                className={styles.axisLabel}
                textAnchor="end"
                transform={`translate(${(band(date) ?? 0) + half},${SIZE - MARGIN.top - MARGIN.bottom + 12}) rotate(-35)`}
              >
                {formatCompactDate(date)}
              </text>
            </g>
          ))}
        </g>
      </svg>
      <p className={styles.heatmapReadout} aria-hidden="true">
        {hovered
          ? `${formatCompactDate(hovered.from)} paying attention to ${formatCompactDate(hovered.to)}: ${hovered.weight.toFixed(3)}`
          : "Rows are the day paying attention, columns the day attended to. Hover a cell for its value."}
      </p>
      <div className={styles.scale} aria-hidden="true">
        <span>{min.toFixed(3)}</span>
        <span className={styles.scaleBar} style={{ background: `linear-gradient(90deg, ${low}, ${mid}, ${high})` }} />
        <span>{max.toFixed(3)}</span>
      </div>
      <DataTableDisclosure
        summary="Show the matrix as a table"
        caption="Attention matrix. Rows pay attention to columns."
        columns={[
          { header: "Day", value: (row: { date: string; weights: number[] }) => formatCompactDate(row.date) },
          ...dates.map((date, index) => ({
            header: formatCompactDate(date),
            value: (row: { date: string; weights: number[] }) => (row.weights[index] ?? 0).toFixed(3),
          })),
        ]}
        rows={dates.map((date, index) => ({ date, weights: matrix[index] ?? [] }))}
      />
    </div>
  );
}
