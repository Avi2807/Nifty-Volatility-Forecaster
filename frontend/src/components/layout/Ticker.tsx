import { useState } from "react";
import { formatCompactDate, formatIndexLevel, formatPct, formatSignedPct } from "../../lib/format";
import type { ReplayRow } from "../../lib/replay";
import { logRvToAnnualVolPct, logRvToDailyMovePct } from "../../lib/volatility";
import styles from "./Ticker.module.css";

function tickerItems(row: ReplayRow): { label: string; value: string }[] {
  return [
    { label: "Forecast made", value: formatCompactDate(row.date) },
    ...(row.nifty_close === null ? [] : [{ label: "Nifty close", value: formatIndexLevel(row.nifty_close) }]),
    { label: "India VIX", value: logRvToAnnualVolPct(row.vix_implied).toFixed(2) },
    { label: "Forecast move", value: `±${formatPct(logRvToDailyMovePct(row.forecast))}` },
    { label: "Actual move", value: formatSignedPct(row.return_oc * 100) },
    { label: "Realized", value: `±${formatPct(logRvToDailyMovePct(row.actual_log_rv))}` },
    { label: "99% VaR", value: formatPct(row.var99 * 100) },
  ];
}

function TickerGroup({ row, hidden }: { row: ReplayRow; hidden?: boolean }) {
  return (
    <ul className={styles.group} aria-hidden={hidden}>
      {tickerItems(row).map((item) => (
        <li key={item.label} className={styles.item}>
          <span className={styles.label}>{item.label}</span> <span className={styles.value}>{item.value}</span>
        </li>
      ))}
    </ul>
  );
}

/** The selected day's numbers. A second, hidden copy makes the loop seamless. */
export function Ticker({ row }: { row?: ReplayRow }) {
  const [paused, setPaused] = useState(false);
  return (
    <section className={styles.ticker} data-paused={paused} aria-label="The selected day's numbers">
      {row && (
        <>
          <div className={styles.track}>
            <TickerGroup row={row} />
            <TickerGroup row={row} hidden />
          </div>
          <button type="button" className={styles.pause} aria-pressed={paused} onClick={() => setPaused((current) => !current)}>
            {paused ? "Play" : "Pause"}
            <span className="visually-hidden"> the ticker</span>
          </button>
        </>
      )}
    </section>
  );
}
