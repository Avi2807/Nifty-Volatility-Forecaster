import type { DieboldMarianoTest, Metrics, ModelName, PeriodScores } from "../../api/types";
import { formatPValue } from "../../lib/format";
import { MODEL_LABELS } from "../../lib/models";
import { findTest, scoredModels, winner } from "./scoring";
import { WinnableCell } from "./WinnableCell";
import { ScoreTableFrame } from "./ScoreTableFrame";

function harPValue(name: ModelName, loss: "squared_error_log_rv" | "qlike", tests: DieboldMarianoTest[]): string {
  if (name === "har") return "reference";
  const test = findTest(tests, name, "har", loss);
  return test ? formatPValue(test.p_value) : "not tested";
}

export function ModelScoreTable({ period, metrics }: { period: PeriodScores; metrics: Metrics }) {
  const bestMse = winner(period, "mse_log_rv");
  const bestQlike = winner(period, "qlike");
  const tests = [...period.diebold_mariano, ...(metrics.comparison?.diebold_mariano ?? [])];
  return (
    <ScoreTableFrame title="Test period, every model">
      <thead>
        <tr>
          <th scope="col">Model</th>
          <th scope="col">MSE (log RV)</th>
          <th scope="col">QLIKE</th>
          <th scope="col">p vs HAR, MSE</th>
          <th scope="col">p vs HAR, QLIKE</th>
          <th scope="col">VaR breaches</th>
          <th scope="col">Kupiec (limit {metrics.kupiec_critical_value})</th>
        </tr>
      </thead>
      <tbody>
        {scoredModels(period).map(({ name, score }) => (
          <tr key={name}>
            <th scope="row">{MODEL_LABELS[name]}</th>
            <WinnableCell value={score.mse_log_rv.toFixed(4)} isWinner={name === bestMse} />
            <WinnableCell value={score.qlike.toFixed(4)} isWinner={name === bestQlike} />
            <td>{harPValue(name, "squared_error_log_rv", tests)}</td>
            <td>{harPValue(name, "qlike", tests)}</td>
            <td>
              {score.var99.breaches} of about {score.var99.expected_breaches.toFixed(1)}
            </td>
            <td>
              {score.var99.kupiec_lr.toFixed(2)}, {score.var99.kupiec_rejected ? "fails" : "passes"}
            </td>
          </tr>
        ))}
      </tbody>
    </ScoreTableFrame>
  );
}
