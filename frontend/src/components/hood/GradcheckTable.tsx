import type { GradcheckResult, ModelCard } from "../../api/types";
import { DataTableDisclosure, type TableColumn } from "../charts/DataTableDisclosure";

const COLUMNS: TableColumn<GradcheckResult>[] = [
  { header: "Component", value: (row) => row.case },
  { header: "Array", value: (row) => row.array },
  { header: "Shape", value: (row) => row.shape.join(" × ") },
  { header: "Relative error", value: (row) => row.relative_error.toExponential(2) },
  { header: "Result", value: (row) => (row.passed ? "Pass" : "Fail") },
];

export function GradcheckTable({ gradcheck }: { gradcheck: ModelCard["gradcheck"] }) {
  return <DataTableDisclosure summary={`Show all ${gradcheck.checks} gradient checks`} caption="Gradient check results by component" columns={COLUMNS} rows={gradcheck.results} />;
}
