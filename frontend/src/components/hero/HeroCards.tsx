import { formatLongDate, formatPct, formatSignedPct } from "../../lib/format";
import type { ReplayRow } from "../../lib/replay";
import { logRvToAnnualVolPct, logRvToDailyMovePct } from "../../lib/volatility";
import styles from "./Hero.module.css";

/** The lead: one enormous number, with its label above and its deck below. */
export function ForecastCard({ row }: { row: ReplayRow }) {
  const isLater = row.period === "later";
  return (
    <div className={styles.lead}>
      <p className="kicker">From the replay{isLater && ", later check"}</p>
      <h2 id="hero-title" className={styles.label}>
        What the {isLater ? "no-news model" : "news model"} forecast for {formatLongDate(row.target_date)}
      </h2>
      <p className={styles.numberReplay}>
        ±{formatPct(logRvToDailyMovePct(row.forecast))}
        <span className="visually-hidden"> expected daily move</span>
      </p>
      <p className={styles.deck} aria-live="polite">
        One standard deviation, forecast after the close on {formatLongDate(row.date)}.
      </p>
      <p className={styles.leadData}>
        <span>{formatPct(logRvToAnnualVolPct(row.forecast), 1)} annualised volatility</span>
        <span>99% VaR: a loss of {formatPct(row.var99 * 100)}</span>
      </p>
    </div>
  );
}

/** Set beside the lead like a corrections column: what the forecast has to answer to. */
export function ActualCard({ row }: { row: ReplayRow }) {
  return (
    <article className={styles.correction}>
      <h2 className="kicker">What actually happened</h2>
      <p className={styles.value}>
        {formatSignedPct(row.return_oc * 100)}
        <span className="visually-hidden"> open-to-close return</span>
      </p>
      <p className={styles.note}>Open to close, against a forecast of ±{formatPct(logRvToDailyMovePct(row.forecast))}.</p>
      <p className={styles.data}>
        Realized move ±{formatPct(logRvToDailyMovePct(row.actual_log_rv))}, or {formatPct(logRvToAnnualVolPct(row.actual_log_rv), 1)} annualised
      </p>
      {row.breached ? (
        <p className={styles.breached}>
          <span aria-hidden="true">✕ </span>Loss went past the 99% VaR
        </p>
      ) : (
        <p className={styles.within}>
          <span aria-hidden="true">✓ </span>Stayed within the 99% VaR
        </p>
      )}
    </article>
  );
}

export function VixCard({ row }: { row: ReplayRow }) {
  return (
    <article className={styles.correction}>
      <h2 className="kicker">India VIX's implied view</h2>
      <p className={styles.valueSmall}>
        ±{formatPct(logRvToDailyMovePct(row.vix_implied))}
        <span className="visually-hidden"> implied daily move</span>
      </p>
      <p className={styles.data}>{formatPct(logRvToAnnualVolPct(row.vix_implied), 1)} annualised, which is the VIX level that day</p>
      <p className={styles.note}>VIX also covers overnight moves and carries a risk premium, so it usually sits above session-only volatility.</p>
    </article>
  );
}
