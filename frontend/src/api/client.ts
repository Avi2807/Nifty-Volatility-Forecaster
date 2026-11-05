import type {
  ApiClient,
  DateRange,
  DayDetails,
  ForecastsResponse,
  ForwardRecord,
  LaterResponse,
  LiveForecast,
  ModelName,
  SentimentResponse,
  SummaryResponse,
  VarResponse,
} from "./types";

export const API_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
export const USE_MOCKS: boolean = import.meta.env.VITE_USE_MOCKS === "true";

/** A failed request. `status` is undefined when the backend could not be reached at all. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function withQuery(path: string, query: Record<string, string | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value) params.set(key, value);
  }
  const text = params.toString();
  return text ? `${path}?${text}` : path;
}

async function get<T>(path: string, fetcher: typeof fetch): Promise<T> {
  let response: Response;
  try {
    response = await fetcher(`${API_URL}/api${path}`);
  } catch {
    throw new ApiError(`Can't reach the backend at ${API_URL}.`);
  }
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = typeof body?.detail === "string" ? body.detail : `The request failed with status ${response.status}.`;
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export function createHttpClient(fetcher: typeof fetch = (...args) => fetch(...args)): ApiClient {
  return {
    summary: () => get<SummaryResponse>("/summary", fetcher),
    forecasts: (range: DateRange = {}, models?: ModelName[]) =>
      get<ForecastsResponse>(withQuery("/forecasts", { ...range, models: models?.join(",") }), fetcher),
    later: () => get<LaterResponse>("/later", fetcher),
    live: () => get<LiveForecast>("/live", fetcher),
    forward: () => get<ForwardRecord>("/forward", fetcher),
    day: (date: string) => get<DayDetails>(`/day/${encodeURIComponent(date)}`, fetcher),
    valueAtRisk: (model: ModelName) => get<VarResponse>(withQuery("/var", { model }), fetcher),
    sentiment: (range: DateRange = {}) => get<SentimentResponse>(withQuery("/sentiment", { ...range }), fetcher),
  };
}

let clientPromise: Promise<ApiClient> | null = null;

/** The one client the app uses. Mock code loads only in mock mode, so it stays out of the real build. */
export function getClient(): Promise<ApiClient> {
  clientPromise ??= USE_MOCKS
    ? import("../mocks/mockClient").then((module) => module.mockClient)
    : Promise.resolve(createHttpClient());
  return clientPromise;
}
