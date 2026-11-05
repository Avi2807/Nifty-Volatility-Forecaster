import styles from "./Scoreboard.module.css";

/** A small marker for the best value, plus hidden text for screen readers. */
export function WinnerMark() {
  return (
    <>
      <span className={styles.marker} aria-hidden="true">◆ </span>
      <span className="visually-hidden">Best: </span>
    </>
  );
}

/** A score cell. The winner is bold with the marker. */
export function WinnableCell({ value, isWinner }: { value: string; isWinner: boolean }) {
  return (
    <td className={isWinner ? styles.winner : undefined}>
      {isWinner && <WinnerMark />}
      {value}
    </td>
  );
}
