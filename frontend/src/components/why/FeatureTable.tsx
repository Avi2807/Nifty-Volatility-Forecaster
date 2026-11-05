import type { ModelCard } from "../../api/types";
import { MARKET_DISPLAY, NEWS_DISPLAY, standardScore, type FeatureDisplay } from "../../lib/features";
import { formatLongDate } from "../../lib/format";
import styles from "./WhyThisForecast.module.css";

interface FeatureTableProps {
  date: string;
  features: Record<string, number>;
  stats: ModelCard["feature_train_stats"];
}

function scoreText(score: number | null): string {
  if (score === null) return "—";
  return `${score > 0 ? "+" : score < 0 ? "−" : ""}${Math.abs(score).toFixed(1)} SD`;
}

function FeatureRows({ title, displays, features, stats }: { title: string; displays: FeatureDisplay[]; features: Record<string, number>; stats: ModelCard["feature_train_stats"] }) {
  return (
    <>
      <tr>
        <th colSpan={4} scope="colgroup" className={styles.groupRow}>
          {title}
        </th>
      </tr>
      {displays.map(({ name, label, format }) => {
        const value = features[name];
        const stat = stats[name];
        if (value === undefined || !stat) return null;
        return (
          <tr key={name}>
            <th scope="row" className={styles.rowLabel}>
              {label}
            </th>
            <td>{format(value)}</td>
            <td>{format(stat.mean)}</td>
            <td>{scoreText(standardScore(value, stat.mean, stat.std))}</td>
          </tr>
        );
      })}
    </>
  );
}

export function FeatureTable({ date, features, stats }: FeatureTableProps) {
  return (
    <div className={styles.block}>
      <h3 className={styles.heading}>Inputs that day vs their train-period average</h3>
      <p className={styles.caution}>The train period is 2015 to 2019. SD is the number of train-period standard deviations from the average. The weekday inputs aren't listed: the last day in the window is {formatLongDate(date)}.</p>
      <div className={styles.tableScroller} role="region" aria-label="Feature values" tabIndex={0}>
        <table className={styles.featureTable}>
          <thead>
            <tr>
              <th scope="col">Input</th>
              <th scope="col">That day</th>
              <th scope="col">Train average</th>
              <th scope="col">Difference</th>
            </tr>
          </thead>
          <tbody>
            <FeatureRows title="Market" displays={MARKET_DISPLAY} features={features} stats={stats} />
            <FeatureRows title="News" displays={NEWS_DISPLAY} features={features} stats={stats} />
          </tbody>
        </table>
      </div>
    </div>
  );
}
