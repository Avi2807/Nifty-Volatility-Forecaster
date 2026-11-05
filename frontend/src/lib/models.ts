import type { ModelName } from "../api/types";

export const MODEL_LABELS: Record<ModelName, string> = {
  transformer: "Transformer",
  transformer_nosent: "Transformer, no news",
  har: "HAR-RV",
  naive: "Naive",
  vix_implied: "VIX-implied",
  garch: "GARCH(1,1)",
  mlp: "MLP",
  lstm: "LSTM",
};

export const ALL_MODELS: ModelName[] = ["transformer", "transformer_nosent", "har", "naive", "vix_implied", "garch", "mlp", "lstm"];

/** Colour follows the model and never its position, so hiding a series can't repaint the others. */
export const SERIES_TOKENS: Record<ModelName | "actual", string> = {
  actual: "--series-actual",
  transformer: "--series-transformer",
  transformer_nosent: "--series-transformer-nosent",
  har: "--series-har",
  naive: "--series-naive",
  vix_implied: "--series-vix",
  garch: "--series-garch",
  mlp: "--series-mlp",
  lstm: "--series-lstm",
};
