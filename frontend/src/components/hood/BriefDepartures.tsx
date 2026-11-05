import styles from "./UnderTheHood.module.css";

export function BriefDepartures({ departures }: { departures: string[] }) {
  if (departures.length === 0) return null;
  return (
    <div className={styles.departures}>
      <h3 className={styles.departuresTitle}>Where this differs from the original brief</h3>
      <ul className={styles.departuresList}>
        {departures.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </div>
  );
}
