// Light / dark theme: a "dark" class on <html>, remembered in localStorage, light by default.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Theme = "dark" | "light";
// v2: the default changed from dark to light; a new key lets the new default reach existing browsers
const KEY = "sugar-twin-theme-v2";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage blocked: fall back to the default */
  }
  return "light";
}

const ThemeCtx = createContext<{ theme: Theme; toggle: () => void }>({ theme: "light", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(initialTheme);
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem(KEY, theme);
    } catch {
      /* ignore */
    }
  }, [theme]);
  return <ThemeCtx.Provider value={{ theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) }}>{children}</ThemeCtx.Provider>;
}

export const useTheme = () => useContext(ThemeCtx);

/** Colours for SVG charts, which can't use Tailwind classes. Status colours follow the roles in index.css
 *  (validated per mode): inRange = normal, high = caution, low = critical, forecast = informational. */
export function useChartTheme() {
  const dark = useTheme().theme === "dark";
  return dark
    ? {
        inRange: "#3a9a70", high: "#b5862a", low: "#d0607c", forecast: "#4a8cc6", brand: "#2fa3ad", activity: "#4a8cc6",
        grid: "#26262a", ink: "#ededef", now: "#f7f7f8", surface: "#131315", muted: "#8b8b94",
        tooltip: { background: "#1a1a1d", border: "1px solid #2a2a2e", color: "#ededef" },
        mealFill: "#23415a", mealStroke: "#6aa3d4", mealText: "#98c0e3", unloggedFill: "#202024", unloggedStroke: "#6b6b74",
        actual: "#8b8b94", base: "#8b8b94",
      }
    : {
        inRange: "#2e9a6a", high: "#b8860b", low: "#c23b5e", forecast: "#2d76ad", brand: "#1d7b84", activity: "#3b84bf",
        grid: "#e4e4e7", ink: "#27272a", now: "#18181b", surface: "#ffffff", muted: "#71717a",
        tooltip: { background: "#ffffff", border: "1px solid #e4e4e7", color: "#27272a" },
        mealFill: "#d6e6f3", mealStroke: "#2d76ad", mealText: "#255f8b", unloggedFill: "#f6f6f7", unloggedStroke: "#a1a1aa",
        actual: "#a1a1aa", base: "#a1a1aa",
      };
}

export type ChartTheme = ReturnType<typeof useChartTheme>;
