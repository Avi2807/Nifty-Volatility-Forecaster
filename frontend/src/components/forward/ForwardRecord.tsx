import { useForwardRecord } from "../../api/queries";
import type { ChallengerName, DieboldMarianoTest, ForwardRecord, ForwardRow, Metrics, ModelScore, RecordModel, VarLine } from "../../api/types";
import { formatDate, formatPct, formatPValue, formatSignedPct } from "../../lib/format";
import { MODEL_LABELS } from "../../lib/models";
import { logRvToDailyMovePct } from "../../lib/volatility";
import { ErrorState } from "../feedback/ErrorState";
import { Skeleton } from "../feedback/Skeleton";
import { Section } from "../layout/Section";
import { ScoreTableFrame } from "../scoreboard/ScoreTableFrame";
import { WinnableCell } from "../scoreboard/WinnableCell";
import styles from "../scoreboard/Scoreboard.module.css";

const RECORD_MODELS: RecordModel[] = ["transformer_nosent", "har", "naive", "vix_implied", "garch", "mlp_nosent", "lstm_nosent"];
const CHALLENGERS: ChallengerName[] = ["garch", "mlp_nosent", "lstm_nosent"];
const LOSS_LABELS = { squared_error_log_rv: "MSE", qlike: "QLIKE" } as const;

const recordLabel = (name: RecordModel): string => (name === "mlp_nosent" ? "MLP, no news" : name === "lstm_nosent" ? "LSTM, no news" : MODEL_LABELS[name]);
const sentenceLabel = (name: ChallengerName): string => (name === "mlp_nosent" ? "no-news MLP" : name === "lstm_nosent" ? "no-news LSTM" : MODEL_LABELS[name]);

const move = (logRv: number | null) => (logRv === null ? "n/a" : `±${formatPct(logRvToDailyMovePct(logRv))}`);
const score = (value: number | null) => (value === null ? "n/a" : value.toFixed(4));
const days = (n: number) => `${n} ${n === 1 ? "day" : "days"}`;

function RunningScores({ record }: { record: ForwardRecord }) {
  const { live, backfilled } = record.scores;
  return (
    <ScoreTableFrame
      title={
        <>
          Running scores, lower is better
          <span className={styles.captionNote}>Each model counts only the days it has a forecast for. A model added after a forecast was saved stays empty on that row.</span>
        </>
      }
    >
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col">Live days</th>
          <th scope="col">Live MSE</th>
          <th scope="col">Live QLIKE</th>
          <th scope="col">Backfilled days</th>
          <th scope="col">Backfilled MSE</th>
          <th scope="col">Backfilled QLIKE</th>
        </tr>
      </thead>
      <tbody>
        {RECORD_MODELS.map((name) => (
          <tr key={name}>
            <th scope="row">{recordLabel(name)}</th>
            <td>{live.models[name].n_days}</td>
            <td>{score(live.models[name].mse_log_rv)}</td>
            <td>{score(live.models[name].qlike)}</td>
            <td>{backfilled.models[name].n_days}</td>
            <td>{score(backfilled.models[name].mse_log_rv)}</td>
            <td>{score(backfilled.models[name].qlike)}</td>
          </tr>
        ))}
      </tbody>
    </ScoreTableFrame>
  );
}

function VarLines({ record, latest }: { record: ForwardRecord; latest: ForwardRow }) {
  const { live, backfilled } = record.scores;
  const lines: { id: VarLine; label: string }[] = [
    { id: "rule", label: `The rule: ${record.rule_multiplier} × the no-news Transformer's forecast` },
    { id: "calibrated", label: `Calibrated: ${record.calibrated_multiplier.toFixed(2)} × that same forecast` },
    { id: "garch", label: `GARCH: ${record.rule_multiplier} × GARCH's forecast` },
  ];
  const breaches = (period: typeof live, id: VarLine) => {
    const { n_days: n, breaches: count, expected_breaches: expected } = period.var99[id];
    return n === 0 ? "No days yet" : `${count} of about ${expected.toFixed(2)}, over ${days(n)}`;
  };
  return (
    <>
      <ScoreTableFrame title={`99% VaR lines, for the session on ${formatDate(latest.target_date)} and so far`}>
        <thead>
          <tr>
            <th scope="col">Line</th>
            <th scope="col">Loss it allows</th>
            <th scope="col">Live breaches</th>
            <th scope="col">Backfilled breaches</th>
          </tr>
        </thead>
        <tbody>
          {lines.map(({ id, label }) => (
            <tr key={id}>
              <th scope="row">{label}</th>
              <td>{latest.var99[id] === null ? "n/a" : formatPct(latest.var99[id]! * 100)}</td>
              <td>{breaches(live, id)}</td>
              <td>{breaches(backfilled, id)}</td>
            </tr>
          ))}
        </tbody>
      </ScoreTableFrame>
      <div className={styles.notes}>
        <p>
          <strong>Neither candidate has been judged yet.</strong> A sound 99% line breaks about once in 100 sessions, so this takes hundreds of new days. The calibrated multiplier was set on the later check, after the test period, so only days after that check can judge it.
        </p>
      </div>
    </>
  );
}

function RecordRows({ rows }: { rows: ForwardRow[] }) {
  return (
    <ScoreTableFrame title="Each forecast against what happened, newest first">
      <thead>
        <tr>
          <th scope="col">Session</th>
          <th scope="col">Kind</th>
          <th scope="col">Forecast</th>
          <th scope="col">HAR-RV</th>
          <th scope="col">Realized</th>
          <th scope="col">Open to close</th>
        </tr>
      </thead>
      <tbody>
        {[...rows].reverse().map((row) => (
          <tr key={row.date}>
            <th scope="row">{formatDate(row.target_date)}</th>
            <td>{row.kind === "live" ? "Live" : "Backfilled"}</td>
            <td>{move(row.forecasts.transformer_nosent)}</td>
            <td>{move(row.forecasts.har)}</td>
            <td>{row.actual_log_rv === null ? "Pending" : move(row.actual_log_rv)}</td>
            <td>{row.return_oc === null ? "Pending" : formatSignedPct(row.return_oc * 100)}</td>
          </tr>
        ))}
      </tbody>
    </ScoreTableFrame>
  );
}

/** A simpler model's significant wins over the no-news Transformer, in words, because they shouldn't rest on a reader spotting a p-value. */
export function challengerFindings(tests: DieboldMarianoTest[], limit: number): string[] {
  return tests
    .filter((test) => test.p_value < limit && test.mean_loss_difference > 0)
    .map((test) => `The ${sentenceLabel(test.model_b as ChallengerName)} beat the no-news Transformer on ${LOSS_LABELS[test.loss]} (p ${formatPValue(test.p_value)}, below the limit of ${limit.toFixed(4)} for these ${tests.length} tests).`);
}

function ChallengerTable({ record, later }: { record: ForwardRecord; later: NonNullable<Metrics["later_check"]> }) {
  const challengers = record.challengers;
  if (!challengers) return null;
  const { models, diebold_mariano: tests, bonferroni_limit: limit, n_days: nDays, target_dates: dates } = challengers.later_check;
  const baseline: { id: string; label: string; value: ModelScore | undefined }[] = [
    { id: "transformer_nosent", label: MODEL_LABELS.transformer_nosent, value: later.models.transformer_nosent },
    { id: "har", label: MODEL_LABELS.har, value: later.models.har },
  ];
  const rows = [...baseline, ...CHALLENGERS.map((name) => ({ id: name, label: recordLabel(name), value: models[name] }))].filter(
    (row): row is { id: string; label: string; value: ModelScore } => row.value !== undefined,
  );
  const best = (key: "mse_log_rv" | "qlike") => rows.reduce((a, b) => (b.value[key] < a.value[key] ? b : a)).id;
  const pValue = (id: string, loss: DieboldMarianoTest["loss"]) => {
    if (id === "transformer_nosent") return "reference";
    const test = tests.find((item) => item.model_b === id && item.loss === loss);
    return test ? formatPValue(test.p_value) : "not tested";
  };
  const findings = challengerFindings(tests, limit);
  return (
    <>
      <ScoreTableFrame
        title={
          <>
            The simpler models on the later check
            <span className={styles.captionNote}>
              {nDays} sessions from {formatDate(dates[0])} to {formatDate(dates[1])}. The MLP and LSTM were retrained without news, with no tuning, and neither saw this period.
            </span>
          </>
        }
      >
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">MSE (log RV)</th>
            <th scope="col">QLIKE</th>
            <th scope="col">VaR breaches</th>
            <th scope="col">p vs no-news Transformer, MSE</th>
            <th scope="col">p vs no-news Transformer, QLIKE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ id, label, value }) => (
            <tr key={id}>
              <th scope="row">{label}</th>
              <WinnableCell value={value.mse_log_rv.toFixed(4)} isWinner={id === best("mse_log_rv")} />
              <WinnableCell value={value.qlike.toFixed(4)} isWinner={id === best("qlike")} />
              <td>
                {value.var99.breaches} of about {value.var99.expected_breaches.toFixed(1)}
              </td>
              <td>{pValue(id, "squared_error_log_rv")}</td>
              <td>{pValue(id, "qlike")}</td>
            </tr>
          ))}
        </tbody>
      </ScoreTableFrame>
      <div className={styles.notes}>
        {findings.map((finding) => (
          <p key={finding}>
            <strong>{finding}</strong>
          </p>
        ))}
        <p>The live forecast stays with the frozen no-news Transformer. Changing that because of this table would be tuning on it, so the forward record has to make the case on new days.</p>
      </div>
    </>
  );
}

export function ForwardRecordContent({ record, later }: { record: ForwardRecord; later?: Metrics["later_check"] }) {
  const latest = record.rows[record.rows.length - 1];
  const { live, backfilled } = record.scores;
  return (
    <>
      <div className={styles.notes}>
        <p>
          So far {days(live.n_days)} of live forecasts and {days(backfilled.n_days)} of backfilled ones have an outcome. That's far too few to rank the models, so the running scores don't mark a best.
        </p>
        <p>No forecasts exist from mid-May until October 2026: the minute file stops in May, and Yahoo's bars only reach back about 60 days.</p>
      </div>
      <RunningScores record={record} />
      {latest && <VarLines record={record} latest={latest} />}
      <RecordRows rows={record.rows} />
      {later && <ChallengerTable record={record} later={later} />}
    </>
  );
}

/** Loads on its own, and steps aside if the live code hasn't made a record yet. */
export function ForwardRecordSection({ later }: { later?: Metrics["later_check"] }) {
  const { data, error, isPending, refetch } = useForwardRecord();
  if (data === null) return null;
  return (
    <Section
      id="forward-record"
      kicker="Forward record"
      title="How the forecasts have done since the study"
      standfirst="Kept out of every study score. A live forecast was saved before its session opened; a backfilled one was worked out later from the data up to the evening before, so it isn't live evidence."
    >
      {isPending && <Skeleton size="chart" label="Loading the forward record" />}
      {error && !data && <ErrorState title="The forward record can't load" error={error} onRetry={() => void refetch()} />}
      {data && <ForwardRecordContent record={data} later={later} />}
    </Section>
  );
}
