import { formatDate } from "../../lib/format";
import styles from "./Notices.module.css";

interface ReplayNoticeProps {
  days: number;
  first: string;
  last: string;
  testEnd: string;
}

export function ReplayNotice({ days, first, last, testEnd }: ReplayNoticeProps) {
  return (
    <p className={styles.replay}>
      <strong>This is a replay, not a live forecast.</strong> Pick any of the {days} days from {formatDate(first)} to{" "}
      {formatDate(last)} to see what a frozen model forecast for the next session and what actually happened.{" "}
      {last === testEnd
        ? "The replay stops there because the news archive does."
        : `Up to ${formatDate(testEnd)} that's the test period, with the news model. The news archive ends there, so later days are a separate check using the no-news model, up to where the price data ends.`}
    </p>
  );
}
