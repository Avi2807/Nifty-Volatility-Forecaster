import type { ModelCard } from "../../api/types";
import { Section } from "../layout/Section";
import { ArchitectureDiagram } from "./ArchitectureDiagram";
import { BriefDepartures } from "./BriefDepartures";
import { GradcheckTable } from "./GradcheckTable";
import { ModelFacts } from "./ModelFacts";

export function UnderTheHood({ card }: { card: ModelCard }) {
  return (
    <Section id="under-the-hood" kicker="Under the hood" title="Written by hand, in NumPy" standfirst="Every model, layer, gradient and optimiser here is implemented from scratch in NumPy, with no machine-learning libraries.">
      <ArchitectureDiagram card={card} />
      <ModelFacts card={card} />
      <GradcheckTable gradcheck={card.gradcheck} />
      <BriefDepartures departures={card.architecture.departures_from_brief} />
    </Section>
  );
}
