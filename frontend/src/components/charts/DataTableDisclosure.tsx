import { useState } from "react";
import styles from "./DataTableDisclosure.module.css";

export interface TableColumn<Row> {
  header: string;
  value: (row: Row) => string;
}

interface DataTableDisclosureProps<Row> {
  summary: string;
  caption: string;
  columns: TableColumn<Row>[];
  rows: Row[];
}

/** The chart's values as a table, for keyboards, screen readers and anyone who wants the numbers. */
export function DataTableDisclosure<Row>({ summary, caption, columns, rows }: DataTableDisclosureProps<Row>) {
  const [open, setOpen] = useState(false);
  return (
    <details className={styles.details} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className={styles.summary}>{summary}</summary>
      {open && (
        <div className={styles.scroller} role="region" aria-label={caption} tabIndex={0}>
          <table className={styles.table}>
            <caption className="visually-hidden">{caption}</caption>
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column.header} scope="col">
                    {column.header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={index}>
                  {columns.map((column) => (
                    <td key={column.header}>{column.value(row)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </details>
  );
}
