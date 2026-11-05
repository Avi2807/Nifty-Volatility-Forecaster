import type { PeriodScores } from "../../api/types";
import { formatPValue } from "../../lib/format";
import { findTest } from "./scoring";
import { WinnableCell } from "./WinnableCell";
import { ScoreTableFrame } from "./ScoreTableFrame";

const LOSSES = [
  { key: "mse_log_rv", test: "squared_error_log_rv", label: "MSE (log RV)" },
  { key: "qlike", test: "qlike", label: "QLIKE" },
] as const;

/** Does news help? A marked winner with a large p-value still means the gap could be chance, so it says so. */
export function AblationTable({ period }: { period: PeriodScores }) {
  const withNews = period.models.transformer;
  const withoutNews = period.models.transformer_nosent;
  if (!withNews || !withoutNews) return null;
  return (
    <ScoreTableFrame title="Does news help? With and without the sentiment inputs">
      <thead>
        <tr>
          <th scope="col">Score</th>
          <th scope="col">With news</th>
          <th scope="col">No news</th>
          <th scope="col">Diebold-Mariano p</th>
          <th scope="col">Reading</th>
        </tr>
      </thead>
      <tbody>
        {LOSSES.map(({ key, test: lossName, label }) => {
          const test = findTest(period.diebold_mariano, "transformer", "transformer_nosent", lossName);
          return (
            <tr key={key}>
              <th scope="row">{label}</th>
              <WinnableCell value={withNews[key].toFixed(4)} isWinner={withNews[key] < withoutNews[key]} />
              <WinnableCell value={withoutNews[key].toFixed(4)} isWinner={withoutNews[key] < withNews[key]} />
              <td>{test ? formatPValue(test.p_value) : "not tested"}</td>
              <td>{test ? (test.p_value < 0.05 ? "Significant at 5%" : "Not significant at 5%") : "—"}</td>
            </tr>
          );
        })}
      </tbody>
    </ScoreTableFrame>
  );
}
