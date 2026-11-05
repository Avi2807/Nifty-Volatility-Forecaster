const LONG_DATE = new Intl.DateTimeFormat("en-GB", {
  weekday: "short",
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});
const SHORT_DATE = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const COMPACT_DATE = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });

// Dates are calendar days with no time zone, so they are always read and written as UTC.
const asDate = (iso: string) => new Date(`${iso}T00:00:00Z`);

export const formatLongDate = (iso: string) => LONG_DATE.format(asDate(iso));
export const formatDate = (iso: string) => SHORT_DATE.format(asDate(iso));
export const formatCompactDate = (iso: string) => COMPACT_DATE.format(asDate(iso));

const INDEX_LEVEL = new Intl.NumberFormat("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
export const formatIndexLevel = (value: number) => INDEX_LEVEL.format(value);

export const formatPct = (value: number, digits = 2) => `${value.toFixed(digits)}%`;
export const formatSignedPct = (value: number, digits = 2) =>
  `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(value).toFixed(digits)}%`;

export function formatPValue(p: number): string {
  return p < 0.0001 ? "<0.0001" : p.toFixed(4);
}

export function formatStatistic(value: number): string {
  return `${value < 0 ? "−" : ""}${Math.abs(value).toFixed(2)}`;
}

export function formatSeconds(seconds: number): string {
  return seconds < 90 ? `${Math.round(seconds)} seconds` : `${(seconds / 60).toFixed(1)} minutes`;
}
