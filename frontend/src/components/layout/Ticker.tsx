import { useState } from "react";
import type { LiveForecast } from "../../api/types";
import { formatCompactDate, formatIndexLevel, formatPct, formatSignedPct } from "../../lib/format";
import type { ReplayRow } from "../../lib/replay";
import { logRvToAnnualVolPct, logRvToDailyMovePct } from "../../lib/volatility";
import styles from "./Ticker.module.css";

interface TickerItem {
  label: string;
  value: string;
}

function liveItems(forecast: LiveForecast): TickerItem[] {
  return [
    { label: "Forecast for", value: formatCompactDate(forecast.target_date) },
    { label: "Nifty close", value: formatIndexLevel(forecast.nifty_close) },
    { label: "India VIX", value: forecast.india_vix.toFixed(2) },
    { label: "Forecast move", value: `±${formatPct(logRvToDailyMovePct(forecast.forecast))}` },
    { label: "Annualised", value: formatPct(logRvToAnnualVolPct(forecast.forecast), 1) },
    { label: "99% VaR", value: formatPct(forecast.var99 * 100) },
  ];
}

function replayItems(row: ReplayRow): TickerItem[] {
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

function TickerGroup({ items, hidden }: { items: TickerItem[]; hidden?: boolean }) {
  return (
    <ul className={styles.group} aria-hidden={hidden}>
      {items.map((item) => (
        <li key={item.label} className={styles.item}>
          <span className={styles.label}>{item.label}</span> <span className={styles.value}>{item.value}</span>
        </li>
      ))}
    </ul>
  );
}

/** The live forecast's numbers, or the selected replay day's if there's no live forecast. A second, hidden copy makes the loop seamless. */
export function Ticker({ row, live }: { row?: ReplayRow; live?: LiveForecast | null }) {
  const [paused, setPaused] = useState(false);
  const items = live ? liveItems(live) : row ? replayItems(row) : null;
  return (
    <section className={styles.ticker} data-paused={paused} aria-label={live ? "The live forecast's numbers" : "The selected day's numbers"}>
      {items && (
        <>
          <div className={styles.track}>
            <TickerGroup items={items} />
            <TickerGroup items={items} hidden />
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
