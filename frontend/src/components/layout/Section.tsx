import type { ReactNode } from "react";
import styles from "./Section.module.css";

interface SectionProps {
  id: string;
  kicker: string;
  title: string;
  /** One sentence. */
  standfirst?: ReactNode;
  children: ReactNode;
}

export function Section({ id, kicker, title, standfirst, children }: SectionProps) {
  return (
    <section id={id} className={styles.section} aria-labelledby={`${id}-title`}>
      <header className={styles.head}>
        <p className="kicker">{kicker}</p>
        <h2 id={`${id}-title`} className={styles.title}>
          {title}
        </h2>
        {standfirst && <p className={styles.standfirst}>{standfirst}</p>}
      </header>
      {children}
    </section>
  );
}
