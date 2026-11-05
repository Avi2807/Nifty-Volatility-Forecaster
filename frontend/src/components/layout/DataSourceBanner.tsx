import type { DataSource } from "../../api/types";
import styles from "./Notices.module.css";

const MESSAGES: Partial<Record<DataSource, string>> = {
  mock: "Mock mode: every number on this page comes from made-up fixtures, not from the model.",
  synthetic: "Synthetic run: every number on this page comes from simulated data and must not be reported.",
};

/** Real results show nothing. Anything else is labelled on every view, so it can't pass for results. */
export function DataSourceBanner({ source }: { source?: DataSource }) {
  const message = source && MESSAGES[source];
  if (!message) return null;
  return (
    <p className={styles.banner} role="status">
      <span aria-hidden="true">⚠ </span>
      {message}
    </p>
  );
}
