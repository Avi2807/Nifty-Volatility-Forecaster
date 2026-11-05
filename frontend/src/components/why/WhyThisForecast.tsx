import { useDayDetails } from "../../api/queries";
import type { ModelCard } from "../../api/types";
import { formatLongDate } from "../../lib/format";
import type { ReplayPeriod } from "../../lib/replay";
import { EmptyState } from "../feedback/EmptyState";
import { ErrorState } from "../feedback/ErrorState";
import { Skeleton } from "../feedback/Skeleton";
import { Section } from "../layout/Section";
import { AttentionPanel } from "./AttentionPanel";
import { FeatureTable } from "./FeatureTable";
import { HeadlineLists } from "./HeadlineLists";
import styles from "./WhyThisForecast.module.css";

interface WhyThisForecastProps {
  selectedDate: string;
  period: ReplayPeriod;
  modelCard: ModelCard;
}

export function WhyThisForecast({ selectedDate, period, modelCard }: WhyThisForecastProps) {
  const isLater = period === "later";
  const { data, error, isPending, isPlaceholderData, refetch } = useDayDetails(isLater ? undefined : selectedDate);
  const updating = isPlaceholderData;

  return (
    <Section
      id="why"
      kicker="Why this forecast"
      title="What the model had in front of it"
      standfirst={
        isLater ? (
          <>
            This forecast came from the no-news model, after the close on <strong>{formatLongDate(selectedDate)}</strong>.
          </>
        ) : (
          <>
            The past 22 sessions, the day's headlines and the day's inputs, as they stood after the close on <strong>{formatLongDate(selectedDate)}</strong>.
          </>
        )
      }
    >
      {isLater && (
        <EmptyState>Attention weights, headlines and inputs were saved for test-period days only, so there are none to show for {formatLongDate(selectedDate)}. The news archive had also ended by then.</EmptyState>
      )}
      {!isLater && error && !updating && <ErrorState error={error} onRetry={() => void refetch()} />}
      {!isLater && isPending && !error && (
        <div className={styles.loading}>
          <Skeleton size="stat" label="Loading this day's details" />
          <Skeleton size="stat" />
        </div>
      )}
      {!isLater && data && !error && (
        <div className={updating ? styles.updating : styles.content} aria-busy={updating}>
          <AttentionPanel day={data} />
          <HeadlineLists headlines={data.headlines} />
          <FeatureTable date={data.date} features={data.features} stats={modelCard.feature_train_stats} />
        </div>
      )}
    </Section>
  );
}
