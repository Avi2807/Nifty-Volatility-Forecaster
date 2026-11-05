import { describe, expect, it } from "vitest";
import tokens from "./tokens.css?raw";

function block(selector: string): Record<string, string> {
  const start = tokens.indexOf(`${selector} {`);
  const body = tokens.slice(start, tokens.indexOf("\n}", start));
  return Object.fromEntries([...body.matchAll(/(--[a-z0-9-]+):\s*([^;]+);/g)].map((match) => [match[1], match[2]]));
}

function luminance(hex: string): number {
  const channels = [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16) / 255);
  const [r, g, b] = channels.map((value) => (value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4)) as [number, number, number];
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a: string, b: string): number {
  const [light, dark] = [luminance(a), luminance(b)].sort((x, y) => y - x) as [number, number];
  return (light + 0.05) / (dark + 0.05);
}

const day = block(":root");
const night = { ...day, ...block(':root[data-theme="night"]') };

const TEXT = ["--color-text", "--color-text-secondary", "--color-text-muted", "--color-link", "--color-accent", "--status-good", "--status-critical-text"];
const MARKS = ["--series-actual", "--series-transformer", "--series-transformer-nosent", "--series-har", "--series-naive", "--series-vix", "--series-garch", "--series-mlp", "--series-lstm", "--chart-bar", "--chart-breach", "--chart-var-line", "--color-focus"];

describe.each([
  ["day", day],
  ["night", night],
])("%s theme", (_, theme) => {
  const page = theme["--color-page"] as string;

  it.each(TEXT)("%s is readable on the page at 4.5:1", (token) => {
    expect(contrast(theme[token] as string, page)).toBeGreaterThanOrEqual(4.5);
  });

  it.each(MARKS)("%s is visible on the page at 3:1", (token) => {
    expect(contrast(theme[token] as string, page)).toBeGreaterThanOrEqual(3);
  });

  it("the warning band is readable", () => {
    expect(contrast(theme["--status-warning-text"] as string, theme["--status-warning-surface"] as string)).toBeGreaterThanOrEqual(4.5);
  });
});

describe("the brief's colours", () => {
  it("uses the specified day values", () => {
    expect(day).toMatchObject({ "--color-page": "#f6e3d4", "--color-text": "#1a1612", "--color-text-secondary": "#5c5048", "--color-accent": "#b3261e", "--status-good": "#2f6b4f", "--highlight-surface": "#f2c14e" });
  });

  it("uses the specified night values", () => {
    expect(night).toMatchObject({ "--color-page": "#0e0f0c", "--color-text": "#e8e2d6", "--color-accent": "#e8a33d", "--status-critical": "#e5534b", "--status-good": "#5fa37e" });
  });

  it("highlighter text is readable on its mark", () => {
    expect(contrast(day["--highlight-text"] as string, day["--highlight-surface"] as string)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(night["--highlight-text"] as string, night["--color-page"] as string)).toBeGreaterThanOrEqual(4.5);
  });
});
