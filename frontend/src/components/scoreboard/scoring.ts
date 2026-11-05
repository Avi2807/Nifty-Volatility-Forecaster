import type { ComparisonTest, DieboldMarianoTest, ModelComparison, ModelName, ModelScore, PeriodScores } from "../../api/types";
import { ALL_MODELS } from "../../lib/models";

export type ScoreKey = "mse_log_rv" | "qlike";

export function scoredModels(period: PeriodScores): { name: ModelName; score: ModelScore }[] {
  return ALL_MODELS.flatMap((name) => {
    const score = period.models[name];
    return score ? [{ name, score }] : [];
  });
}

/** The model with the lowest loss. */
export function winner(period: PeriodScores, key: ScoreKey): ModelName | null {
  const models = scoredModels(period);
  const best = models.reduce<(typeof models)[number] | null>((lowest, item) => (!lowest || item.score[key] < lowest.score[key] ? item : lowest), null);
  return best?.name ?? null;
}

export function findTest(tests: DieboldMarianoTest[], a: string, b: string, loss: DieboldMarianoTest["loss"]): DieboldMarianoTest | undefined {
  return tests.find((test) => test.model_a === a && test.model_b === b && test.loss === loss);
}

/** The with-news Transformer's tests against one other model, on each loss. */
export function testsAgainst(comparison: ModelComparison, other: ModelName): { mse?: ComparisonTest; qlike?: ComparisonTest } {
  const tests = comparison.diebold_mariano.filter((test) => test.model_a === "transformer" && test.model_b === other);
  return { mse: tests.find((test) => test.loss === "squared_error_log_rv"), qlike: tests.find((test) => test.loss === "qlike") };
}

/** A lower loss alone isn't a win, so the reading always carries how far the evidence goes. */
export function leadReading(test: ComparisonTest): string {
  const side = test.lower_loss === "transformer" ? "Ahead" : "Behind";
  if (test.significant_bonferroni) return `${side}, holds up`;
  return test.significant_5pct ? `${side}, at 5% only` : `${side}, not significant`;
}

export interface LeadGroups {
  /** Ahead on both losses, after the adjustment for running many tests. */
  clear: ModelName[];
  /** Neither loss differs at 5%, in either direction. */
  inseparable: ModelName[];
  /** Everything between those two. */
  mixed: ModelName[];
}

export function groupLeads(comparison: ModelComparison): LeadGroups {
  const groups: LeadGroups = { clear: [], inseparable: [], mixed: [] };
  for (const name of comparison.ladder) {
    if (name === "transformer") continue;
    const { mse, qlike } = testsAgainst(comparison, name);
    if (!mse || !qlike) continue;
    const both = [mse, qlike];
    if (both.every((test) => test.significant_bonferroni && test.lower_loss === "transformer")) groups.clear.push(name);
    else if (both.every((test) => !test.significant_5pct)) groups.inseparable.push(name);
    else groups.mixed.push(name);
  }
  return groups;
}

/** The neural model with the lowest mean score, one seed at a time. */
export function bestSingleSeed(comparison: ModelComparison, key: "mse_mean" | "qlike_mean"): ModelName | null {
  let best: ModelName | null = null;
  for (const name of comparison.ladder) {
    const seeds = comparison.models[name].seed_scores;
    const current = best && comparison.models[best].seed_scores;
    if (seeds && (!current || seeds[key] < current[key])) best = name;
  }
  return best;
}
