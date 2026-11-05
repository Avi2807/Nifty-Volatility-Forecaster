import { useCallback, useEffect, useMemo, useState } from "react";

const readParam = () => new URLSearchParams(window.location.search).get("date");

/** Resolve a requested date against the replay's dates: exact match, else the next one, else the last. */
export function resolveDate(requested: string | null, dates: string[]): string | undefined {
  if (dates.length === 0) return undefined;
  const last = dates[dates.length - 1];
  if (!requested || !/^\d{4}-\d{2}-\d{2}$/.test(requested)) return last;
  return dates.find((date) => date >= requested) ?? last;
}

/**
 * The selected forecast date, kept in ?date=YYYY-MM-DD so any view can be linked to.
 * With no date in the URL it opens on the latest forecast. A date outside the list moves to the nearest
 * valid one and reports that through `adjustedFrom`, so the page can say what happened.
 */
export function useSelectedDate(dates: string[]) {
  const [requested, setRequested] = useState<string | null>(readParam);

  useEffect(() => {
    const onPopState = () => setRequested(readParam());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const date = useMemo(() => resolveDate(requested, dates), [requested, dates]);
  const index = date ? dates.indexOf(date) : -1;

  const select = useCallback((next: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("date", next);
    window.history.replaceState(null, "", url); // stepping through days shouldn't fill the back button
    setRequested(next);
  }, []);

  const adjustedFrom = requested && date && requested !== date ? requested : null;
  return { date, index, select, adjustedFrom };
}
