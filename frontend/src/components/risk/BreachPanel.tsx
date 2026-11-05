import type { VarResponse } from "../../api/types";
import styles from "./RiskDesk.module.css";

/** The breach count beside the chart, against what a sound line would give. */
export function BreachPanel({ response }: { response: VarResponse }) {
  const { kupiec, diagnostics } = response;
  return (
    <aside className={styles.panel} aria-label="Breach count">
      <p className="kicker">Breaches</p>
      <p className={styles.bigNumber}>{kupiec.breaches}</p>
      <p className={styles.panelNote}>against about {kupiec.expected_breaches.toFixed(1)} expected, because a sound 99% VaR breaks about 1 session in 100</p>

      <p className={styles.why}>
        For context: the squared open-to-close return averages {diagnostics.mean_squared_return_over_mean_rv.toFixed(2)} times RV in this period. With next-day RV known
        exactly, this rule would breach {diagnostics.breaches_if_rv_were_known} times.
      </p>
    </aside>
  );
}
