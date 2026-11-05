import { useId, type ReactNode } from "react";
import styles from "./Scoreboard.module.css";

/** The title sits outside the scrolling area, so a wide table on a phone can't clip it. */
export function ScoreTableFrame({ title, children }: { title: ReactNode; children: ReactNode }) {
  const titleId = useId();
  return (
    <figure className={styles.frame}>
      <figcaption id={titleId} className={styles.caption}>
        {title}
      </figcaption>
      <div className={styles.scroller} role="region" aria-labelledby={titleId} tabIndex={0}>
        <table className={styles.table} aria-labelledby={titleId}>
          {children}
        </table>
      </div>
    </figure>
  );
}
