import type { VarResponse } from "../../api/types";
import styles from "./RiskDesk.module.css";

/** The result beside the chart. A failing VaR is stated as a failure, with what the failure means. */
export function KupiecPanel({ response }: { response: VarResponse }) {
  const { kupiec, diagnostics } = response;
  const tooMany = kupiec.breaches > kupiec.expected_breaches;
  return (
    <aside className={styles.panel} aria-label="Kupiec test result">
      <p className="kicker">Breaches</p>
      <p className={styles.bigNumber}>{kupiec.breaches}</p>
      <p className={styles.panelNote}>against about {kupiec.expected_breaches.toFixed(1)} expected, because a sound 99% VaR breaks about 1 session in 100</p>

      <p className="kicker">Kupiec test</p>
      {kupiec.rejected ? (
        <p className={styles.fail}>
          <span aria-hidden="true">✕ </span>Fails, with {tooMany ? "too many" : "too few"} breaches
        </p>
      ) : (
        <p className={styles.pass}>
          <span aria-hidden="true">✓ </span>Passes
        </p>
      )}
      <p className={styles.panelNote}>
        <span className={styles.lr}>LR {kupiec.likelihood_ratio.toFixed(2)}</span>, and it fails above <span className={styles.lr}>{kupiec.critical_value.toFixed(2)}</span>
      </p>

      <p className={styles.why}>
        For context: the squared open-to-close return averages {diagnostics.mean_squared_return_over_mean_rv.toFixed(2)} times RV in this period. With next-day RV known
        exactly, this rule would breach {diagnostics.breaches_if_rv_were_known} times.
      </p>
    </aside>
  );
}
