// Light / dark theme: a "dark" class on <html>, remembered in localStorage, dark by default.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type Theme = "dark" | "light";
const KEY = "sugar-twin-theme";

function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage blocked: fall back to the default */
  }
  return "dark";
}

const ThemeCtx = createContext<{ theme: Theme; toggle: () => void }>({ theme: "dark", toggle: () => {} });

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

/** Colours for SVG charts, which can't use Tailwind classes. */
export function useChartTheme() {
  const dark = useTheme().theme === "dark";
  return dark
    ? {
        grid: "#1e2a44", ink: "#e2e8f0", now: "#f8fafc", surface: "#111a2e", muted: "#8b9ab3",
        tooltip: { background: "#17223a", border: "1px solid #2b3a57", color: "#e2e8f0" },
        mealFill: "#78350f", mealStroke: "#f59e0b", mealText: "#fcd34d", unloggedFill: "#1e2a44", unloggedStroke: "#64748b",
        actual: "#94a3b8", base: "#8b9ab3",
      }
    : {
        grid: "#f1f5f9", ink: "#334155", now: "#0f172a", surface: "#ffffff", muted: "#64748b",
        tooltip: { background: "#ffffff", border: "1px solid #e2e8f0", color: "#334155" },
        mealFill: "#fef3c7", mealStroke: "#d97706", mealText: "#92400e", unloggedFill: "#f1f5f9", unloggedStroke: "#94a3b8",
        actual: "#94a3b8", base: "#94a3b8",
      };
}

export type ChartTheme = ReturnType<typeof useChartTheme>;
