import type { ModelCard } from "../../api/types";
import { formatSeconds } from "../../lib/format";
import styles from "./UnderTheHood.module.css";

export function ModelFacts({ card }: { card: ModelCard }) {
  const { training, gradcheck, sentiment_classifier: sentiment } = card;
  const modelCount = Object.keys(training.best_epochs).length * training.seeds.length;
  const facts = [
    {
      label: "Parameters",
      value: (card.parameter_count.transformer ?? 0).toLocaleString("en-GB"),
      note: `${(card.parameter_count.transformer_nosent ?? 0).toLocaleString("en-GB")} without news`,
    },
    {
      label: "Training time",
      value: formatSeconds(training.seconds_all_models),
      note: `All ${modelCount} models`,
    },
    {
      label: "Gradient checks",
      value: `${gradcheck.passed} of ${gradcheck.checks} pass`,
      note: `Worst relative error ${gradcheck.worst_relative_error.toExponential(1)}, limit ${gradcheck.tolerance.toExponential(0)}`,
    },
    {
      label: "Sentiment classifier",
      value: `${(sentiment.holdout_accuracy * 100).toFixed(1)}% accuracy`,
      note: `Macro F1 ${sentiment.holdout_macro_f1.toFixed(3)}. ${sentiment.note}`,
    },
  ];
  return (
    <dl className={styles.facts}>
      {facts.map((fact) => (
        <div key={fact.label} className={styles.fact}>
          <dt className="kicker">{fact.label}</dt>
          <dd className={styles.factValue}>{fact.value}</dd>
          <dd className={styles.factNote}>{fact.note}</dd>
        </div>
      ))}
    </dl>
  );
}
