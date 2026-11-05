import type { ReplayRow } from "../../lib/replay";
import { ReplayNotice } from "../layout/ReplayNotice";
import { ActualCard, ForecastCard, VixCard } from "./HeroCards";
import { DateScrubber } from "./DateScrubber";
import styles from "./Hero.module.css";

interface HeroProps {
  rows: ReplayRow[];
  index: number;
  adjustedFrom: string | null;
  onSelect: (date: string) => void;
}

export function Hero({ rows, index, adjustedFrom, onSelect }: HeroProps) {
  const row = rows[index];
  const first = rows[0];
  const last = rows[rows.length - 1];
  const testEndIndex = rows.filter((item) => item.period === "test").length - 1;
  const testEnd = rows[testEndIndex];
  if (!row || !first || !last || !testEnd) return null;
  return (
    <section id="replay" className={styles.hero} aria-labelledby="hero-title">
      <div className={styles.front}>
        <ForecastCard row={row} />
        <div className={styles.corrections}>
          <ActualCard row={row} />
          <VixCard row={row} />
        </div>
      </div>
      <div className={styles.controls}>
        <ReplayNotice days={rows.length} first={first.date} last={last.date} testEnd={testEnd.date} />
        <div>
          <DateScrubber rows={rows} index={index} testEndIndex={testEndIndex} onSelect={onSelect} />
          {adjustedFrom && (
            <p className={styles.adjusted} role="status">
              “{adjustedFrom}” isn't a day in this replay, so this shows the nearest one.
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
