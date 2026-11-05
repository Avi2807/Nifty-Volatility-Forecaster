import styles from "./Feedback.module.css";

type SkeletonSize = "line" | "stat" | "chart";

/** A placeholder shaped like what's loading, so the page doesn't jump when data arrives. */
export function Skeleton({ size = "line", label }: { size?: SkeletonSize; label?: string }) {
  return (
    <div className={`${styles.skeleton} ${styles[size]}`} role={label ? "status" : undefined} aria-hidden={label ? undefined : true}>
      {label && <span className="visually-hidden">{label}</span>}
    </div>
  );
}
