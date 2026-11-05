import type { ComparisonTest, ModelComparison, ModelName } from "../../api/types";
import { formatPValue, formatStatistic } from "../../lib/format";
import { MODEL_LABELS } from "../../lib/models";
import { bestSingleSeed, groupLeads, leadReading, testsAgainst } from "./scoring";
import { ScoreTableFrame } from "./ScoreTableFrame";
import styles from "./Scoreboard.module.css";

const ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"];

function joined(names: ModelName[]): string {
  const labels = names.map((name) => (name === "transformer_nosent" ? "the no-news Transformer" : MODEL_LABELS[name]));
  return labels.length < 2 ? (labels[0] ?? "") : `${labels.slice(0, -1).join(", ")} and ${labels[labels.length - 1]}`;
}

function TestCells({ test }: { test: ComparisonTest | undefined }) {
  if (!test) return <td colSpan={3}>not tested</td>;
  return (
    <>
      <td>{formatStatistic(test.statistic)}</td>
      <td>{formatPValue(test.p_value)}</td>
      <td>{leadReading(test)}</td>
    </>
  );
}

/** Whether each of the with-news Transformer's leads is real, in numbers and then in words. */
export function ComparisonTests({ comparison, dmLags, originalTestCount }: { comparison: ModelComparison; dmLags: number; originalTestCount: number }) {
  const others = comparison.ladder.filter((name) => name !== "transformer");
  const groups = groupLeads(comparison);
  const family = comparison.diebold_mariano.filter((test) => test.model_a === "transformer");
  const limit = family[0]?.bonferroni_limit;
  const rankMse = ORDINALS[comparison.ranking_mse.indexOf("transformer")];
  const rankQlike = ORDINALS[comparison.ranking_qlike.indexOf("transformer")];
  const seedMse = bestSingleSeed(comparison, "mse_mean");
  const seedQlike = bestSingleSeed(comparison, "qlike_mean");
  return (
    <>
      <ScoreTableFrame
        title={
          <>
            The Transformer against every other model
            <span className={styles.captionNote}>
              Diebold-Mariano tests with {dmLags} Newey-West lags. A negative statistic means the with-news Transformer had the lower loss.
              {limit !== undefined && ` "Holds up" means p is below ${limit.toFixed(4)}, the Bonferroni limit for these ${family.length} tests, which is stricter than the limit for the original ${originalTestCount}.`}
            </span>
          </>
        }
      >
        <thead>
          <tr>
            <th scope="col">Against</th>
            <th scope="col">MSE statistic</th>
            <th scope="col">MSE p</th>
            <th scope="col">On MSE</th>
            <th scope="col">QLIKE statistic</th>
            <th scope="col">QLIKE p</th>
            <th scope="col">On QLIKE</th>
          </tr>
        </thead>
        <tbody>
          {others.map((name) => {
            const { mse, qlike } = testsAgainst(comparison, name);
            return (
              <tr key={name}>
                <th scope="row">{MODEL_LABELS[name]}</th>
                <TestCells test={mse} />
                <TestCells test={qlike} />
              </tr>
            );
          })}
        </tbody>
      </ScoreTableFrame>
      <div className={styles.notes}>
        <p>
          Of the {comparison.ladder.length} models, the with-news Transformer ranks {rankMse} on MSE and {rankQlike} on QLIKE.
          {groups.clear.length > 0 && ` Its lead over ${joined(groups.clear)} holds up on both scores.`}
          {groups.mixed.length > 0 && ` The evidence against ${joined(groups.mixed)} is weaker, because it doesn't hold up on both.`}
          {groups.inseparable.length > 0 && ` It can't be told apart from ${joined(groups.inseparable)} on either score at the 5% level.`}
        </p>
        {seedMse && seedQlike && (
          <p>
            Those scores use forecasts averaged over 3 seeds. One seed at a time, {joined([seedMse])} has the lowest mean MSE and {joined([seedQlike])} the lowest mean QLIKE, so averaging helps some models more than others.
          </p>
        )}
        <p>
          The MLP and LSTM use the brief's training settings with no tuning, while the Transformer's were chosen on validation data. GARCH forecasts the variance of the open-to-close return rather than RV, so it's scored here on a target it doesn't model, and that also explains its higher VaR line on the Risk desk.
        </p>
      </div>
    </>
  );
}
