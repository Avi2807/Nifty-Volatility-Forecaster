import type { Metrics } from "../../api/types";
import { formatDate } from "../../lib/format";
import { MODEL_LABELS } from "../../lib/models";
import { scoredModels, winner } from "./scoring";
import { WinnableCell } from "./WinnableCell";
import { ScoreTableFrame } from "./ScoreTableFrame";
import styles from "./Scoreboard.module.css";

export function LaterCheckTable({ later }: { later: NonNullable<Metrics["later_check"]> }) {
  const bestMse = winner(later, "mse_log_rv");
  const bestQlike = winner(later, "qlike");
  return (
    <ScoreTableFrame
      title={
        <>
          Later check, kept separate
          <span className={styles.captionNote}>
            {later.n_days} sessions from {formatDate(later.target_dates[0])} to {formatDate(later.target_dates[1])}. Left out: {later.excluded}
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
        </tr>
      </thead>
      <tbody>
        {scoredModels(later).map(({ name, score }) => (
          <tr key={name}>
            <th scope="row">{MODEL_LABELS[name]}</th>
            <WinnableCell value={score.mse_log_rv.toFixed(4)} isWinner={name === bestMse} />
            <WinnableCell value={score.qlike.toFixed(4)} isWinner={name === bestQlike} />
            <td>
              {score.var99.breaches} of about {score.var99.expected_breaches.toFixed(1)}
            </td>
          </tr>
        ))}
      </tbody>
    </ScoreTableFrame>
  );
}
