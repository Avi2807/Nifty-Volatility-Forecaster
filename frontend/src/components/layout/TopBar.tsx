import { formatDate, formatLongDate } from "../../lib/format";
import type { ReplayRow } from "../../lib/replay";
import { ThemeToggle } from "./ThemeToggle";
import { Ticker } from "./Ticker";
import styles from "./TopBar.module.css";

const LINKS = [
  { href: "#next-session", label: "Next session" },
  { href: "#replay", label: "Replay" },
  { href: "#forecast-vs-actual", label: "Forecast vs actual" },
  { href: "#risk-desk", label: "Risk desk" },
  { href: "#why", label: "Why this forecast" },
  { href: "#scoreboard", label: "Scoreboard" },
  { href: "#under-the-hood", label: "Under the hood" },
];

/** The masthead and ticker scroll away with the page. The section links below them stay stuck to the top. */
export function TopBar({ selectedRow }: { selectedRow?: ReplayRow }) {
  return (
    <>
      <header className={styles.masthead}>
        <div className={styles.dateline}>
          <p className={styles.edition}>
            <span className="kicker">Replay edition</span>
            {selectedRow && <span className={styles.editionDate}>{formatLongDate(selectedRow.date)}</span>}
          </p>
          <ThemeToggle />
        </div>
        <a className={styles.brand} href="#next-session">
          Nifty Volatility Forecaster
        </a>
      </header>
      <Ticker row={selectedRow} />
      <nav aria-label="Sections" className={styles.bar}>
        <div className={styles.inner}>
          <ul className={styles.links}>
            {LINKS.map((link) => (
              <li key={link.href}>
                <a className={styles.link} href={link.href}>
                  {link.label}
                </a>
              </li>
            ))}
          </ul>
          {selectedRow && (
            <p className={styles.date}>
              <span className="visually-hidden">Selected date: </span>
              {formatDate(selectedRow.date)}
            </p>
          )}
        </div>
      </nav>
    </>
  );
}
