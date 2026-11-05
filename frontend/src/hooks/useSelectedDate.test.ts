import { describe, expect, it } from "vitest";
import { resolveDate } from "./useSelectedDate";

const DATES = ["2021-01-04", "2021-01-05", "2021-01-07"];

describe("resolveDate", () => {
  it("keeps a date that is in the test set", () => {
    expect(resolveDate("2021-01-05", DATES)).toBe("2021-01-05");
  });

  it("moves a holiday to the next test day", () => {
    expect(resolveDate("2021-01-06", DATES)).toBe("2021-01-07");
  });

  it("moves a date before the test set to its first day", () => {
    expect(resolveDate("2020-03-23", DATES)).toBe("2021-01-04");
  });

  it("opens on the last day when the date is missing, malformed or too late", () => {
    expect(resolveDate(null, DATES)).toBe("2021-01-07");
    expect(resolveDate("yesterday", DATES)).toBe("2021-01-07");
    expect(resolveDate("2030-01-01", DATES)).toBe("2021-01-07");
  });

  it("has no date before the data arrives", () => {
    expect(resolveDate("2021-01-05", [])).toBeUndefined();
  });
});
