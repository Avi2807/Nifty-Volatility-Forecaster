import type { ModelComparison, ModelName, PeriodScores, SeedScores } from "../../api/types";
import { MODEL_LABELS } from "../../lib/models";
import { DataTableDisclosure, type TableColumn } from "../charts/DataTableDisclosure";
import { WinnerMark } from "./WinnableCell";
import styles from "./Scoreboard.module.css";

interface Rung {
  name: ModelName;
  qlike: number;
  mse: number;
  nParameters: number;
  seconds: number | null;
  seeds: SeedScores | null;
}

const seedCell = (mean: number | undefined, spread: number | undefined) => (mean === undefined || spread === undefined ? "n/a" : `${mean.toFixed(4)} ± ${spread.toFixed(4)}`);

function trainingTime(seconds: number | null): string {
  if (seconds === null) return "Nothing to fit";
  if (seconds < 0.001) return "Under 0.001s";
  return seconds < 1 ? `${seconds.toFixed(3)}s` : `${seconds.toFixed(1)}s`;
}

const COLUMNS: TableColumn<Rung>[] = [
  { header: "Model", value: (rung) => MODEL_LABELS[rung.name] },
  { header: "QLIKE", value: (rung) => rung.qlike.toFixed(4) },
  { header: "QLIKE, seed mean ± spread", value: (rung) => seedCell(rung.seeds?.qlike_mean, rung.seeds?.qlike_spread) },
  { header: "MSE (log RV)", value: (rung) => rung.mse.toFixed(4) },
  { header: "MSE, seed mean ± spread", value: (rung) => seedCell(rung.seeds?.mse_mean, rung.seeds?.mse_spread) },
  { header: "Parameters", value: (rung) => rung.nParameters.toLocaleString("en-GB") },
  { header: "Training time", value: (rung) => trainingTime(rung.seconds) },
];

/** Test QLIKE from the simplest model to the most complex. The order is the point, so it never re-sorts by score. */
export function ModelLadder({ comparison, period }: { comparison: ModelComparison; period: PeriodScores }) {
  const rungs = comparison.ladder.flatMap((name): Rung[] => {
    const score = period.models[name];
    const cost = comparison.models[name];
    return score ? [{ name, qlike: score.qlike, mse: score.mse_log_rv, nParameters: cost.n_parameters, seconds: cost.training_seconds, seeds: cost.seed_scores }] : [];
  });
  const largest = Math.max(...rungs.map((rung) => rung.qlike));
  const smallest = Math.min(...rungs.map((rung) => rung.qlike));
  return (
    <figure className={styles.frame}>
      <figcaption className={styles.caption}>
        The model ladder
        <span className={styles.captionNote}>Test QLIKE, from the simplest model at the top to the most complex at the bottom. Shorter is better.</span>
      </figcaption>
      <ol className={styles.ladder}>
        {rungs.map((rung) => (
          <li key={rung.name} className={styles.rung} title={`${MODEL_LABELS[rung.name]}: QLIKE ${rung.qlike.toFixed(4)}, MSE ${rung.mse.toFixed(4)}, ${rung.nParameters.toLocaleString("en-GB")} parameters`}>
            <span>{MODEL_LABELS[rung.name]}</span>
            <span className={styles.rungTrack} aria-hidden="true">
              <span className={rung.name.startsWith("transformer") ? styles.rungBarAccent : styles.rungBar} style={{ width: `${(rung.qlike / largest) * 100}%` }} />
            </span>
            <span className={rung.qlike === smallest ? `${styles.rungValue} ${styles.winner}` : styles.rungValue}>
              {rung.qlike === smallest && <WinnerMark />}
              {rung.qlike.toFixed(4)}
            </span>
          </li>
        ))}
      </ol>
      <DataTableDisclosure summary="Show costs and seed spread as a table" caption="Scores, seed spread, parameters and training time, by model" columns={COLUMNS} rows={rungs} />
    </figure>
  );
}
