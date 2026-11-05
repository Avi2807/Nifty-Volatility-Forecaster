import { describe, expect, it, vi } from "vitest";
import { API_URL, ApiError, createHttpClient } from "./client";

const respond = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

describe("createHttpClient", () => {
  it("builds each URL with only the parameters given", async () => {
    const fetcher = vi.fn<typeof fetch>(async () => respond({ rows: [] }));
    const client = createHttpClient(fetcher);
    await client.forecasts({ start: "2021-01-01" }, ["har", "naive"]);
    await client.forecasts();
    await client.valueAtRisk("transformer");
    await client.day("2022-06-13");
    expect(fetcher.mock.calls.map(([url]) => url)).toEqual([
      `${API_URL}/api/forecasts?start=2021-01-01&models=har%2Cnaive`,
      `${API_URL}/api/forecasts`,
      `${API_URL}/api/var?model=transformer`,
      `${API_URL}/api/day/2022-06-13`,
    ]);
  });

  it("reports an unreachable backend without a status", async () => {
    const client = createHttpClient(async () => {
      throw new TypeError("Failed to fetch");
    });
    const error = await client.summary().catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBeUndefined();
    expect((error as ApiError).message).toContain(API_URL);
  });

  it("passes on the backend's message and status for a refused request", async () => {
    const client = createHttpClient(async () => respond({ detail: "2020-03-23 is not a test date." }, 404));
    const error = (await client.day("2020-03-23").catch((caught: unknown) => caught)) as ApiError;
    expect(error.status).toBe(404);
    expect(error.message).toBe("2020-03-23 is not a test date.");
  });
});
