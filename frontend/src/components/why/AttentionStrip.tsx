import { useTokenReader } from "../../hooks/useTokens";
import { formatCompactDate, formatDate } from "../../lib/format";
import { attentionColorScale } from "./attentionScale";
import styles from "./WhyThisForecast.module.css";

/** 22 small bars. Height is true to zero, so near-equal weights look near-equal. The shading stretches to show the order. */
export function AttentionStrip({ dates, weights }: { dates: string[]; weights: number[] }) {
  const readToken = useTokenReader();
  const color = attentionColorScale(weights, [readToken("--heat-low"), readToken("--heat-mid"), readToken("--heat-high")]);
  const highest = Math.max(...weights);

  return (
    <figure className={styles.stripFigure}>
      <ol className={styles.strip} aria-label="Attention weight for each of the 22 input days, oldest first">
        {dates.map((date, index) => {
          const weight = weights[index] ?? 0;
          const description = `${formatDate(date)}: ${weight.toFixed(3)}`;
          return (
            <li key={date} className={styles.stripCell} title={description}>
              <span className={styles.stripTrack}>
                <span className={styles.stripBar} style={{ height: `${highest > 0 ? (weight / highest) * 100 : 0}%`, background: color(weight) }} />
              </span>
              <span className={styles.stripDate} aria-hidden="true">
                {formatCompactDate(date)}
              </span>
              <span className="visually-hidden">{description}</span>
            </li>
          );
        })}
      </ol>
      <figcaption className={styles.stripCaption}>
        Taller and heavier means more weight, from {Math.min(...weights).toFixed(3)} to {highest.toFixed(3)}.
      </figcaption>
    </figure>
  );
}
