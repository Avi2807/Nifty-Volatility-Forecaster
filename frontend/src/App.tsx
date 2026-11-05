import { useMemo } from "react";
import styles from "./App.module.css";
import { useLaterForecasts, useLiveForecast, useSummary, useTestForecasts } from "./api/queries";
import { EmptyState } from "./components/feedback/EmptyState";
import { ErrorState } from "./components/feedback/ErrorState";
import { Skeleton } from "./components/feedback/Skeleton";
import { ForecastVsActual } from "./components/forecast/ForecastVsActual";
import { Hero } from "./components/hero/Hero";
import { NextSession } from "./components/hero/NextSession";
import { UnderTheHood } from "./components/hood/UnderTheHood";
import { DataSourceBanner } from "./components/layout/DataSourceBanner";
import { TopBar } from "./components/layout/TopBar";
import { RiskDesk } from "./components/risk/RiskDesk";
import { Scoreboard } from "./components/scoreboard/Scoreboard";
import { WhyThisForecast } from "./components/why/WhyThisForecast";
import { useSelectedDate } from "./hooks/useSelectedDate";
import { buildReplay } from "./lib/replay";

const NO_ROWS: never[] = [];
const COLUMN_GAPS = Array.from({ length: 11 }, (_, gap) => gap);

export function App() {
  const summary = useSummary();
  const [firstDay, lastDay] = summary.data?.metrics.test.forecast_dates ?? [];
  const forecasts = useTestForecasts(firstDay, lastDay);
  const hasLaterCheck = Boolean(summary.data?.metrics.later_check);
  const later = useLaterForecasts(hasLaterCheck);
  const live = useLiveForecast();
  const rows = useMemo(() => buildReplay(forecasts.data ?? NO_ROWS, later.data ?? NO_ROWS), [forecasts.data, later.data]);
  const replayDates = useMemo(() => rows.map((row) => row.date), [rows]);
  const { date, index, select, adjustedFrom } = useSelectedDate(replayDates);
  const selected = rows[index];

  const error = summary.error ?? forecasts.error ?? later.error;
  const laterLoading = hasLaterCheck && later.isPending;
  const loading = !error && (summary.isPending || forecasts.isPending || laterLoading);
  const ready = summary.data && forecasts.data && !laterLoading && date && selected;

  return (
    <>
      <a className={styles.skipLink} href="#main">
        Skip to content
      </a>
      <DataSourceBanner source={summary.data?.data_source} />
      <TopBar selectedRow={selected} />
      <main id="main" className={styles.main}>
        <div className={styles.columnRules} aria-hidden="true">
          {COLUMN_GAPS.map((gap) => (
            <span key={gap} />
          ))}
        </div>
        <NextSession />
        {error && (
          <div className={styles.fullState}>
            <ErrorState
              title="The app can't load its data"
              error={error}
              onRetry={() => {
                void summary.refetch();
                void forecasts.refetch();
                if (hasLaterCheck) void later.refetch();
                void live.refetch();
              }}
            />
          </div>
        )}
        {loading && (
          <div className={styles.fullState}>
            <Skeleton label="Loading the replay" />
            <Skeleton size="stat" />
            <Skeleton size="chart" />
          </div>
        )}
        {!error && forecasts.data?.length === 0 && (
          <div className={styles.fullState}>
            <EmptyState>The backend returned no test-period forecasts. Run python run_all.py to rebuild them.</EmptyState>
          </div>
        )}
        {ready && (
          <>
            <Hero rows={rows} index={index} adjustedFrom={adjustedFrom} onSelect={select} />
            <ForecastVsActual rows={rows} selectedDate={date} onSelect={select} />
            <RiskDesk rows={rows} selectedDate={date} onSelect={select} />
            <WhyThisForecast selectedDate={date} period={selected.period} modelCard={summary.data.model_card} />
            <Scoreboard metrics={summary.data.metrics} />
            <UnderTheHood card={summary.data.model_card} />
          </>
        )}
      </main>
    </>
  );
}
