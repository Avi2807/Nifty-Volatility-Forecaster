import { useTheme, type ThemePreference } from "../../hooks/useTheme";
import styles from "./ThemeToggle.module.css";

const OPTIONS: { value: ThemePreference; label: string }[] = [
  { value: "system", label: "Auto" },
  { value: "day", label: "Day" },
  { value: "night", label: "Night" },
];

export function ThemeToggle() {
  const { preference, setPreference } = useTheme();
  return (
    <div className={styles.group} role="group" aria-label="Theme">
      {OPTIONS.map((option) => (
        <button key={option.value} type="button" className={styles.option} aria-pressed={preference === option.value} onClick={() => setPreference(option.value)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}
