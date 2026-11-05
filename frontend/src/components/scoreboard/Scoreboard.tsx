import type { Metrics } from "../../api/types";
import { formatDate } from "../../lib/format";
import { Section } from "../layout/Section";
import { AblationTable } from "./AblationTable";
import { ComparisonTests } from "./ComparisonTests";
import { LaterCheckTable } from "./LaterCheckTable";
import { ModelLadder } from "./ModelLadder";
import { ModelScoreTable } from "./ModelScoreTable";
import { findTest } from "./scoring";
import styles from "./Scoreboard.module.css";

/** How many single-seed models beat HAR on MSE. */
function seedsBeatingHar(metrics: Metrics): { beating: number; total: number } | null {
  const har = metrics.test.models.har?.mse_log_rv;
  if (har === undefined) return null;
  const seeds = Object.values(metrics.test.seed_mse_log_rv).flat();
  return { beating: seeds.filter((value) => value < har).length, total: seeds.length };
}

export function Scoreboard({ metrics }: { metrics: Metrics }) {
  const { test } = metrics;
  const seeds = seedsBeatingHar(metrics);
  const naive = findTest(test.diebold_mariano, "transformer", "naive", "squared_error_log_rv");
  return (
    <Section
      id="scoreboard"
      kicker="Scoreboard"
      title="The test-period table"
      standfirst={`Scores for ${test.n_days} forecasts, for sessions from ${formatDate(test.target_dates[0])} to ${formatDate(test.target_dates[1])}, where lower is better and the best in each column is marked.`}
    >
      <ModelScoreTable period={test} metrics={metrics} />
      <div className={styles.notes}>
        <p>
          Diebold-Mariano tests use {metrics.dm_lags} Newey-West lags and aren't adjusted for running {test.diebold_mariano.length} of them. With a Bonferroni adjustment, only
          p-values below {(0.05 / test.diebold_mariano.length).toFixed(4)} hold up.{metrics.comparison && " The comparison further down runs more tests, so its limit is stricter."}
        </p>
        {seeds && (
          <p>
            Each Transformer forecast averages the forecasts of several seeds. Taken one at a time, {seeds.beating} of the {seeds.total} single-seed models beat HAR on MSE.
          </p>
        )}
        {naive && <p>Transformer vs naive on MSE: p {naive.p_value < 0.0001 ? "< 0.0001" : `= ${naive.p_value.toFixed(4)}`}.</p>}
      </div>
      {metrics.comparison && (
        <>
          <ModelLadder comparison={metrics.comparison} period={test} />
          <ComparisonTests comparison={metrics.comparison} dmLags={metrics.dm_lags} originalTestCount={test.diebold_mariano.length} />
        </>
      )}
      <AblationTable period={test} />
      {metrics.later_check && <LaterCheckTable later={metrics.later_check} />}
    </Section>
  );
}
