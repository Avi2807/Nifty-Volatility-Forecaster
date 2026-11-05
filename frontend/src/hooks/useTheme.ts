import { useCallback, useEffect, useState } from "react";

export type ThemePreference = "system" | "day" | "night";
export type Theme = "day" | "night";

const STORAGE_KEY = "nv-theme"; // index.html reads this too, to set the theme before first paint

export function resolveTheme(preference: ThemePreference, systemIsDark: boolean): Theme {
  if (preference === "system") return systemIsDark ? "night" : "day";
  return preference;
}

function readPreference(): ThemePreference {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored === "day" || stored === "night" ? stored : "system";
  } catch {
    return "system"; // storage can be blocked, and the system setting is the default anyway
  }
}

/** Follows the system setting until the reader picks a theme, and remembers that pick on this browser. */
export function useTheme() {
  const [preference, setPreferenceState] = useState<ThemePreference>(readPreference);

  useEffect(() => {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      document.documentElement.dataset.theme = resolveTheme(preference, query.matches);
    };
    apply();
    query.addEventListener("change", apply);
    return () => query.removeEventListener("change", apply);
  }, [preference]);

  const setPreference = useCallback((next: ThemePreference) => {
    setPreferenceState(next);
    try {
      if (next === "system") localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // The theme still changes for this visit.
    }
  }, []);

  return { preference, setPreference };
}
