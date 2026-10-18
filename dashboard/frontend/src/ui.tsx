// Shared formatting helpers and small UI building blocks.
import type { ReactNode } from "react";
import { ArrowDown, ArrowDownRight, ArrowRight, ArrowUp, ArrowUpRight, HelpCircle } from "lucide-react";
import type { Level, Trend } from "./api";

export const COLORS = {
  veryLow: "#991b1b",
  low: "#ef4444",
  inRange: "#22c55e",
  high: "#f59e0b",
  veryHigh: "#ea580c",
  brand: "#0d9488",
  forecast: "#7c3aed",
  ink: "#334155",
};

const fmtTime = new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true, timeZone: "UTC" });
const fmtDay = new Intl.DateTimeFormat("en-IN", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
export const hhmm = (ms: number) => fmtTime.format(new Date(ms));
export const dayLabel = (ms: number) => fmtDay.format(new Date(ms));
export const dateTime = (ms: number) => `${dayLabel(ms)}, ${hhmm(ms)}`;

export function glucoseColor(g: number | null | undefined) {
  if (g == null) return "text-slate-400";
  if (g < 54) return "text-red-800";
  if (g < 70) return "text-red-500";
  if (g <= 180) return "text-green-600";
  if (g <= 250) return "text-amber-500";
  return "text-orange-600";
}

export function glucoseWord(g: number | null | undefined) {
  if (g == null) return "No reading";
  if (g < 54) return "Very low";
  if (g < 70) return "Low";
  if (g <= 180) return "In range";
  if (g <= 250) return "High";
  return "Very high";
}

export function TrendIcon({ trend, className = "h-5 w-5" }: { trend: Trend; className?: string }) {
  const map = {
    rising_fast: <ArrowUp className={className} />,
    rising: <ArrowUpRight className={className} />,
    steady: <ArrowRight className={className} />,
    falling: <ArrowDownRight className={className} />,
    falling_fast: <ArrowDown className={className} />,
    unknown: <HelpCircle className={className} />,
  };
  return map[trend];
}

export const trendWord: Record<Trend, string> = {
  rising_fast: "rising fast",
  rising: "rising",
  steady: "steady",
  falling: "falling",
  falling_fast: "falling fast",
  unknown: "trend unknown",
};

export const levelStyle: Record<Level, string> = {
  high: "bg-red-50 text-red-700 ring-red-200",
  medium: "bg-amber-50 text-amber-700 ring-amber-200",
  low: "bg-emerald-50 text-emerald-700 ring-emerald-200",
  now: "bg-violet-50 text-violet-700 ring-violet-200",
};

export function Pill({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset ${className}`}>{children}</span>;
}

export function Card({ title, icon, hint, right, children, className = "" }: {
  title?: string; icon?: ReactNode; hint?: string; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`rounded-2xl bg-white p-5 shadow-sm ring-1 ring-slate-200/70 ${className}`}>
      {(title || right) && (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-slate-500">
              {icon}
              {title}
            </h2>
            {hint && <p className="mt-0.5 text-xs text-slate-400">{hint}</p>}
          </div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, sub, className = "" }: { label: string; value: ReactNode; sub?: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl bg-slate-50 px-3 py-2 ${className}`}>
      <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">{label}</div>
      <div className="text-lg font-bold text-slate-800">{value}</div>
      {sub && <div className="text-xs text-slate-500">{sub}</div>}
    </div>
  );
}

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "–" : `${v.toFixed(d)}%`);
export const mg = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v)}`);
