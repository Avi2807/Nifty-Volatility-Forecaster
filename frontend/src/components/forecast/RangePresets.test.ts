import { describe, expect, it } from "vitest";
import type { ReplayRow } from "../../lib/replay";
import { rowsInPreset } from "./RangePresets";

const row = (date: string, target_date: string, period: ReplayRow["period"] = "test") => ({ date, target_date, period }) as ReplayRow;
const TEST = [row("2021-01-01", "2021-01-04"), row("2021-12-31", "2022-01-03"), row("2022-05-31", "2022-06-01"), row("2022-06-30", "2022-07-01")];
const LATER = [row("2023-07-03", "2023-07-04", "later"), row("2026-05-13", "2026-05-14", "later")];
const ROWS = [...TEST, ...LATER];

describe("rowsInPreset", () => {
  const targets = (id: Parameters<typeof rowsInPreset>[1]) => rowsInPreset(ROWS, id).map((item) => item.target_date);

  it("keeps every test row for the full test period, and no later row", () => {
    expect(rowsInPreset(ROWS, "full")).toEqual(TEST);
  });

  it("keeps the later check apart", () => {
    expect(rowsInPreset(ROWS, "later")).toEqual(LATER);
  });

  it("filters on the session a forecast was for, not the day it was made", () => {
    expect(targets("2021")).toEqual(["2021-01-04"]);
    expect(targets("2022h1")).toEqual(["2022-01-03", "2022-06-01"]);
    expect(targets("jun2022")).toEqual(["2022-06-01"]);
  });
});
