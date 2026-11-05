import type { ModelCard } from "../../api/types";
import styles from "./UnderTheHood.module.css";

interface Stage {
  title: string;
  detail: string;
}

function stagesFrom(card: ModelCard): Stage[] {
  const { window_days: days, d_model: d, heads, dropout } = card.architecture;
  const inputs = card.architecture.features.transformer?.length ?? 0;
  return [
    { title: "Input", detail: `${days} days × ${inputs} features` },
    { title: "Projection", detail: `Dense ${inputs} → ${d}` },
    { title: "Positional embedding", detail: `Learned, ${days} × ${d}` },
    { title: "Encoder block", detail: `${heads} heads, FFN ${d * 2}, dropout ${dropout}` },
    { title: "Last-day pooling", detail: `Keep day t, ${d} values` },
    { title: "Output", detail: `Dense ${d} → 1: next-day log RV` },
  ];
}

const BOX = { width: 196, height: 84 };
const GAP = 34;

function StageBox({ stage, x, y }: { stage: Stage; x: number; y: number }) {
  return (
    <g transform={`translate(${x},${y})`}>
      <rect className={styles.box} width={BOX.width} height={BOX.height} />
      <text className={styles.boxTitle} x={BOX.width / 2} y={35} textAnchor="middle">
        {stage.title}
      </text>
      <text className={styles.boxDetail} x={BOX.width / 2} y={58} textAnchor="middle">
        {stage.detail}
      </text>
    </g>
  );
}

function Arrow({ x1, y1, x2, y2, marker }: { x1: number; y1: number; x2: number; y2: number; marker: string }) {
  return <line className={styles.arrow} x1={x1} y1={y1} x2={x2} y2={y2} markerEnd={`url(#${marker})`} />;
}

/** Each SVG needs its own marker, because CSS hides one of the two and a hidden marker doesn't draw. */
function ArrowDefs({ id }: { id: string }) {
  return (
    <defs>
      <marker id={id} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
        <path className={styles.arrowHead} d="M0 0 L10 5 L0 10 Z" />
      </marker>
    </defs>
  );
}

/** Two rows on wide screens and one column on narrow ones. CSS shows one, so text stays readable at 375px. */
export function ArchitectureDiagram({ card }: { card: ModelCard }) {
  const stages = stagesFrom(card);
  const summary = stages.map((stage) => `${stage.title}: ${stage.detail}`).join(". Then ");
  const perRow = 3;
  const wideWidth = perRow * BOX.width + (perRow - 1) * GAP;
  const wideHeight = 2 * BOX.height + GAP;
  const tallHeight = stages.length * BOX.height + (stages.length - 1) * (GAP - 8);

  return (
    <figure className={styles.diagram}>
      <svg className={styles.wide} viewBox={`0 0 ${wideWidth} ${wideHeight}`} role="img" aria-label={`Model architecture. ${summary}.`}>
        <ArrowDefs id="architecture-arrow-wide" />
        {stages.map((stage, index) => {
          const column = index % perRow;
          const row = Math.floor(index / perRow);
          const x = column * (BOX.width + GAP);
          const y = row * (BOX.height + GAP);
          const atRowEnd = column === perRow - 1;
          return (
            <g key={stage.title}>
              <StageBox stage={stage} x={x} y={y} />
              {index < stages.length - 1 && !atRowEnd && <Arrow x1={x + BOX.width + 2} y1={y + BOX.height / 2} x2={x + BOX.width + GAP - 3} y2={y + BOX.height / 2} marker="architecture-arrow-wide" />}
              {index < stages.length - 1 && atRowEnd && <path className={styles.wrap} d={`M${x + BOX.width / 2} ${y + BOX.height + 2} V${y + BOX.height + GAP / 2} H${BOX.width / 2} V${y + BOX.height + GAP - 3}`} markerEnd="url(#architecture-arrow-wide)" />}
            </g>
          );
        })}
      </svg>
      <svg className={styles.tall} viewBox={`0 0 ${BOX.width} ${tallHeight}`} role="img" aria-label={`Model architecture. ${summary}.`}>
        <ArrowDefs id="architecture-arrow-tall" />
        {stages.map((stage, index) => {
          const y = index * (BOX.height + GAP - 8);
          return (
            <g key={stage.title}>
              <StageBox stage={stage} x={0} y={y} />
              {index < stages.length - 1 && <Arrow x1={BOX.width / 2} y1={y + BOX.height + 2} x2={BOX.width / 2} y2={y + BOX.height + GAP - 11} marker="architecture-arrow-tall" />}
            </g>
          );
        })}
      </svg>
      <figcaption className={styles.figcaption}>The no-news variant is identical, but has {card.architecture.features.transformer_nosent?.length ?? 0} input features.</figcaption>
    </figure>
  );
}
