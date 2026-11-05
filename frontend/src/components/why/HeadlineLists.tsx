import type { DayDetails, ScoredHeadline } from "../../api/types";
import { EmptyState } from "../feedback/EmptyState";
import styles from "./WhyThisForecast.module.css";

function scoreText(score: number): string {
  const rounded = Number(score.toFixed(2));
  return `${rounded > 0 ? "+" : rounded < 0 ? "\u2212" : ""}${Math.abs(rounded).toFixed(2)}`;
}

function HeadlineList({ title, items }: { title: string; items: ScoredHeadline[] }) {
  return (
    <div>
      <h4 className={`kicker ${styles.subheading}`}>{title}</h4>
      <ol className={styles.headlines}>
        {items.map((item) => (
          <li key={item.headline} className={styles.headline}>
            <span>{item.headline}</span>
            <span className={styles.score}>
              <span className="visually-hidden">Score </span>
              {scoreText(item.score)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function HeadlineLists({ headlines }: { headlines: DayDetails["headlines"] }) {
  const hasNews = headlines.negative.length > 0 || headlines.positive.length > 0;
  return (
    <div className={styles.block}>
      <h3 className={styles.heading}>The day's headlines, by sentiment score</h3>
      {hasNews ? (
        <>
          <div className={styles.headlineColumns}>
            <HeadlineList title="Lowest scores (most negative)" items={headlines.negative} />
            <HeadlineList title="Highest scores (most positive)" items={headlines.positive} />
          </div>
          <p className={styles.caution}>
            Scores run from −1 to +1 and come from a classifier trained on Financial PhraseBank. Its accuracy on Indian news wasn't measured, and it does misread some
            headlines.
          </p>
        </>
      ) : (
        <EmptyState>The news archive has no business headlines for this day, so the news inputs are zero.</EmptyState>
      )}
    </div>
  );
}
