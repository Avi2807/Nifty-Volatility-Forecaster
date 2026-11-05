import { interpolateRgbBasis, scaleSequential } from "d3";

/** One hue, light to dark, spanning the values on screen. Weights sit close together, so the labels matter. */
export function attentionColorScale(values: number[], colors: [string, string, string]) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  return scaleSequential(interpolateRgbBasis(colors)).domain(max > min ? [min, max] : [min, min + 1e-9]);
}
