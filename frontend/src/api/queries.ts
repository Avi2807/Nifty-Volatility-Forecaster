import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ApiError, getClient } from "./client";
import type { ModelName } from "./types";

export function useSummary() {
  return useQuery({ queryKey: ["summary"], queryFn: async () => (await getClient()).summary() });
}

/** Every test-period row. The test dates come from the backend, so nothing here hard-codes them. */
export function useTestForecasts(start?: string, end?: string) {
  return useQuery({
    queryKey: ["forecasts", start, end],
    enabled: Boolean(start && end),
    queryFn: async () => {
      const response = await (await getClient()).forecasts({ start, end });
      return response.rows.filter((row) => row.split === "test");
    },
  });
}

/** The later check's rows. A run with no later check has no such endpoint, so the caller says whether to ask. */
export function useLaterForecasts(enabled: boolean) {
  return useQuery({
    queryKey: ["later"],
    enabled,
    queryFn: async () => (await (await getClient()).later()).rows,
  });
}

/** The latest next-session forecast, or null if none has been made. It changes each evening, so coming back to the tab checks for a new one. */
export function useLiveForecast() {
  return useQuery({
    queryKey: ["live"],
    staleTime: 0,
    refetchOnWindowFocus: true,
    queryFn: async () => {
      try {
        return await (await getClient()).live();
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) return null;
        throw error;
      }
    },
  });
}

/** Keeps the previous day on screen while the next one loads, so stepping through days doesn't flash. */
export function useDayDetails(date: string | undefined) {
  return useQuery({
    queryKey: ["day", date],
    enabled: Boolean(date),
    placeholderData: keepPreviousData,
    queryFn: async () => (await getClient()).day(date as string),
  });
}

export function useValueAtRisk(model: ModelName) {
  return useQuery({
    queryKey: ["var", model],
    placeholderData: keepPreviousData,
    queryFn: async () => (await getClient()).valueAtRisk(model),
  });
}
