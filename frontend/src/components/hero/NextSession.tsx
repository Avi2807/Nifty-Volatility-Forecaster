import { useLiveForecast } from "../../api/queries";
import type { LiveForecast } from "../../api/types";
import { formatDate, formatIndexLevel, formatLongDate, formatPct } from "../../lib/format";
import { sessionHasClosed } from "../../lib/live";
import { logRvToAnnualVolPct, logRvToDailyMovePct } from "../../lib/volatility";
import { EmptyState } from "../feedback/EmptyState";
import { ApiError } from "../../api/client";
import { ErrorState } from "../feedback/ErrorState";
import { Skeleton } from "../feedback/Skeleton";
import heroStyles from "./Hero.module.css";
import styles from "./NextSession.module.css";

const REFRESH_COMMAND = "python -m src.live";

const move = (logRv: number) => formatPct(logRvToDailyMovePct(logRv));

/** The live forecast, as it stands at `now`. It has no outcome yet, so nothing here is a score. */
export function NextSessionForecast({ forecast, now }: { forecast: LiveForecast; now: Date }) {
  const closed = sessionHasClosed(forecast.target_date, now);
  const record = forecast.track_record;
  const percent = logRvToDailyMovePct(forecast.forecast);
  const points = Math.round((forecast.nifty_close * percent) / 100).toLocaleString("en-IN");
  const turnoverGap = (Math.exp(forecast.features.log_turnover_ratio) - 1) * 100;
  return (
    <>
      {closed && (
        <p className={styles.stale} role="status">
          <strong>This forecast is out of date.</strong> The {formatLongDate(forecast.target_date)} session has closed. To make the next one, run <code>{REFRESH_COMMAND}</code> once NSE has published that day's closing file, usually in the evening.
        </p>
      )}
      <div className={heroStyles.front}>
        <div className={heroStyles.lead}>
          <p className="kicker">{closed ? "Last live forecast" : "Next session, live forecast"}</p>
          <h1 id="next-session-title" className={heroStyles.label}>
            What the no-news model {closed ? "forecast" : "forecasts"} for {formatLongDate(forecast.target_date)}
          </h1>
          <p className={heroStyles.number}>
            ±{formatPct(percent)}
            <span className="visually-hidden"> expected move during the session</span>
          </p>
          <p className={heroStyles.deck}>
            One standard deviation, forecast after the close on {formatLongDate(forecast.date)}. That's about {points} points from the {formatIndexLevel(forecast.nifty_close)} close.
          </p>
          <p className={heroStyles.leadData}>
            <span>{formatPct(logRvToAnnualVolPct(forecast.forecast), 1)} annualised volatility</span>
            <span>99% VaR: a loss of {formatPct(forecast.var99 * 100)}</span>
          </p>
        </div>
        <div className={heroStyles.corrections}>
          <article className={heroStyles.correction}>
            <h2 className="kicker">How far off it could be</h2>
            <p className={heroStyles.valueSmall}>
              ±{move(forecast.forecast + record.error_band_low)} to ±{move(forecast.forecast + record.error_band_high)}
            </p>
            <p className={heroStyles.note}>
              Where the session's actual volatility should land about {Math.round(record.error_band_share * 100)}% of the time, going by this model's errors on the {record.n_days} later-check days.
            </p>
          </article>
          <article className={heroStyles.correction}>
            <h2 className="kicker">Other views of the same session</h2>
            <dl className={styles.figures}>
              <div>
                <dt>HAR benchmark</dt>
                <dd>±{move(forecast.references.har)}</dd>
              </div>
              <div>
                <dt>India VIX, which includes overnight moves</dt>
                <dd>±{move(forecast.references.vix_implied)}</dd>
              </div>
              <div>
                <dt>{formatDate(forecast.date)}, what actually happened</dt>
                <dd>±{move(forecast.references.naive)}</dd>
              </div>
            </dl>
          </article>
        </div>
      </div>
      <div className={heroStyles.controls}>
        <div>
          <h2 className="kicker">What went in</h2>
          <dl className={styles.figures}>
            <div>
              <dt>Sessions of 5-minute bars</dt>
              <dd>{forecast.sessions_used}</dd>
            </div>
            <div>
              <dt>India VIX close</dt>
              <dd>{forecast.india_vix.toFixed(2)}</dd>
            </div>
            <div>
              <dt>Turnover, against its 20-day average</dt>
              <dd>
                {turnoverGap >= 0 ? "+" : "−"}
                {Math.abs(turnoverGap).toFixed(0)}%
              </dd>
            </div>
            <div>
              <dt>Weekly expiry that session</dt>
              <dd>{forecast.features.expiry_next === 1 ? "Yes" : "No"}</dd>
            </div>
          </dl>
        </div>
        <div className={styles.caveats}>
          <h2 className="kicker">Read this before relying on it</h2>
          <p>
            <strong>The 99% VaR has run too low.</strong> On the later check, losses passed it on {record.var99_breaches} of {record.n_days} days, against about {Math.round(record.var99_expected_breaches)} expected, so don't treat it as a limit.
          </p>
          <p>It covers the session only, so a gap at the open comes on top.</p>
          <p>
            The prices come from Yahoo's unofficial 5-minute feed, not the feed the model was trained on. VIX, turnover and holidays come from NSE. This forecast is in none of the scores on this page, and it isn't trading advice.
          </p>
          <p className={styles.made}>
            Made at {forecast.made_at.slice(11, 16)} IST on {formatDate(forecast.made_at.slice(0, 10))}, by the same frozen model as the later check.
          </p>
        </div>
      </div>
    </>
  );
}

/** Loads on its own, so the forecast still shows if the replay can't load, and vice versa. */
export function NextSession() {
  const { data, error, isPending, refetch } = useLiveForecast();
  // With no backend at all, the page's own error already says so and its retry covers this too.
  if (error instanceof ApiError && error.status === undefined && !data) return null;
  return (
    <section id="next-session" className={styles.section} aria-labelledby="next-session-title">
      {data && <NextSessionForecast forecast={data} now={new Date()} />}
      {!data && (
        <>
          <p className="kicker">Next session, live forecast</p>
          <h1 id="next-session-title" className={heroStyles.label}>
            The next session
          </h1>
          {isPending && <Skeleton size="stat" label="Loading the live forecast" />}
          {error && <ErrorState title="The live forecast can't load" error={error} onRetry={() => void refetch()} />}
          {data === null && (
            <EmptyState>
              No live forecast has been made yet. To make one, run <code>{REFRESH_COMMAND}</code> after the market closes.
            </EmptyState>
          )}
        </>
      )}
    </section>
  );
}
