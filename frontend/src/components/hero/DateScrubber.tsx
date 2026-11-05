import { useEffect, type CSSProperties } from "react";
import { formatDate, formatLongDate } from "../../lib/format";
import styles from "./DateScrubber.module.css";

interface DateScrubberProps {
  rows: { date: string }[];
  index: number;
  /** The last day of the test period. Days after it are the later check. */
  testEndIndex: number;
  onSelect: (date: string) => void;
}

/** Arrow keys that are safe to use for stepping: not while typing, choosing or using a browser shortcut. */
export function isStepKey(event: KeyboardEvent): boolean {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return false;
  if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return false;
  const target = event.target as HTMLElement | null;
  return !target?.closest("input, select, textarea, [contenteditable], [role='region']");
}

export function DateScrubber({ rows, index, testEndIndex, onSelect }: DateScrubberProps) {
  const lastIndex = rows.length - 1;
  const step = (delta: number) => {
    const next = rows[Math.min(lastIndex, Math.max(0, index + delta))];
    if (next) onSelect(next.date);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!isStepKey(event)) return;
      event.preventDefault();
      const next = rows[Math.min(lastIndex, Math.max(0, index + (event.key === "ArrowRight" ? 1 : -1)))];
      if (next) onSelect(next.date);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [rows, index, lastIndex, onSelect]);

  const current = rows[index];
  const first = rows[0];
  const last = rows[lastIndex];
  const testEnd = rows[testEndIndex];
  if (!current || !first || !last || !testEnd) return null;

  const pickDate = (value: string) => {
    const match = rows.find((row) => row.date >= value) ?? last;
    if (value) onSelect(match.date);
  };

  return (
    <div className={styles.scrubber}>
      <div className={styles.controls}>
        <button type="button" className={styles.step} onClick={() => step(-1)} disabled={index === 0} aria-label="Previous day">
          ←
        </button>
        <div className={styles.rangeWrap}>
          {testEndIndex < lastIndex && <span className={styles.boundary} style={{ "--boundary": testEndIndex / lastIndex } as CSSProperties} />}
          <input
            type="range"
            className={styles.range}
            min={0}
            max={lastIndex}
            value={index}
            onChange={(event) => {
              const next = rows[Number(event.target.value)];
              if (next) onSelect(next.date);
            }}
            aria-label="Forecast date"
            aria-valuetext={formatLongDate(current.date)}
          />
        </div>
        <button type="button" className={styles.step} onClick={() => step(1)} disabled={index === lastIndex} aria-label="Next day">
          →
        </button>
      </div>
      <div className={styles.footer}>
        <span>{formatDate(first.date)}</span>
        <label className={styles.jump}>
          <span>Go to date</span>
          <input type="date" className={styles.dateInput} min={first.date} max={last.date} value={current.date} onChange={(event) => pickDate(event.target.value)} />
        </label>
        <span>{formatDate(last.date)}</span>
      </div>
      {testEndIndex < lastIndex && <p className={styles.boundaryNote}>The mark on the slider is the end of the test period, {formatDate(testEnd.date)}.</p>}
      <p className={styles.hint}>Use the left and right arrow keys to step one trading day.</p>
    </div>
  );
}
