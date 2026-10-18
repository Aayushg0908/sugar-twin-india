// Shared formatting helpers and small UI building blocks.
import { useState, type ReactNode } from "react";
import { ArrowDown, ArrowDownRight, ArrowRight, ArrowUp, ArrowUpRight, ChevronDown, HelpCircle } from "lucide-react";
import type { Level, Trend } from "./api";

const fmtTime = new Intl.DateTimeFormat("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true, timeZone: "UTC" });
const fmtDay = new Intl.DateTimeFormat("en-IN", { weekday: "short", day: "numeric", month: "short", timeZone: "UTC" });
export const hhmm = (ms: number) => fmtTime.format(new Date(ms));
export const dayLabel = (ms: number) => fmtDay.format(new Date(ms));
export const dateTime = (ms: number) => `${dayLabel(ms)}, ${hhmm(ms)}`;

export function glucoseColor(g: number | null | undefined) {
  if (g == null) return "text-slate-400";
  if (g < 54) return "text-red-800";
  if (g < 70) return "text-red-600";
  if (g <= 180) return "text-green-600";
  if (g <= 250) return "text-amber-600";
  return "text-red-600";
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
  low: "bg-green-50 text-green-700 ring-green-200",
  now: "bg-blue-50 text-blue-700 ring-blue-200",
};

export function Pill({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ring-1 ring-inset ${className}`}>{children}</span>;
}

/** A panel. With `collapsible` it is a dropdown: only the header shows until it is clicked.
 *  Pass `open` + `onToggle` to control it from outside (e.g. open it from a button elsewhere). */
export function Card({ title, icon, hint, right, children, className = "", collapsible = false, open, onToggle, id }: {
  title?: string; icon?: ReactNode; hint?: string; right?: ReactNode; children: ReactNode; className?: string;
  collapsible?: boolean; open?: boolean; onToggle?: () => void; id?: string;
}) {
  const [ownOpen, setOwnOpen] = useState(false);
  const isOpen = !collapsible || (open ?? ownOpen);
  const toggle = onToggle ?? (() => setOwnOpen((o) => !o));
  // title with a small tinted icon badge, as in the reference dashboard
  const heading = (
    <h2 className="flex items-center gap-2.5 text-sm font-semibold text-slate-800">
      {icon && (
        <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-brand-50 text-brand-600 ring-1 ring-inset ring-brand-100 [&>svg]:h-3.5 [&>svg]:w-3.5">
          {icon}
        </span>
      )}
      {title}
    </h2>
  );
  const hintText = hint && <p className={`mt-1.5 text-xs text-slate-500 ${icon ? "pl-[38px]" : ""}`}>{hint}</p>;
  return (
    <section id={id} className={`rounded-xl border border-slate-200 bg-white px-5 py-4 shadow-[0_1px_2px_rgba(0,0,0,0.04)] ${className}`}>
      {collapsible ? (
        <button onClick={toggle} aria-expanded={isOpen}
          className={`group flex w-full items-center justify-between gap-3 text-left ${isOpen ? "mb-4" : ""}`}>
          <div>
            {heading}
            {isOpen && hintText}
          </div>
          <span className="flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs font-medium text-slate-600 transition group-hover:bg-slate-50">
            {isOpen ? "Hide" : "Show"}
            <ChevronDown className={`h-3.5 w-3.5 shrink-0 transition ${isOpen ? "rotate-180" : ""}`} />
          </span>
        </button>
      ) : (title || right) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>
            {heading}
            {hintText}
          </div>
          {right}
        </header>
      )}
      {isOpen && children}
    </section>
  );
}

const TONE = {
  red: { dot: "bg-red-500", text: "text-red-600" },
  amber: { dot: "bg-amber-500", text: "text-amber-600" },
  green: { dot: "bg-green-500", text: "text-green-600" },
};

/** A headline number tile: icon badge + label, big value, small status line. */
export function StatTile({ icon, label, value, note, tone = null, onClick, className = "" }: {
  icon: ReactNode; label: string; value: ReactNode; note?: ReactNode; tone?: keyof typeof TONE | null;
  onClick?: () => void; className?: string;
}) {
  const Tag = onClick ? "button" : "div";
  return (
    <Tag onClick={onClick} className={`rounded-xl border border-slate-200 bg-slate-50 px-3.5 py-3 text-left ${onClick ? "transition hover:border-slate-300" : ""} ${className}`}>
      <div className="flex items-center gap-2 text-xs font-medium text-slate-600">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-white text-slate-500 ring-1 ring-inset ring-slate-200 [&>svg]:h-3.5 [&>svg]:w-3.5">{icon}</span>
        {label}
      </div>
      <div className={`mt-2 text-2xl font-bold leading-none ${tone ? TONE[tone].text : "text-slate-900"}`}>{value}</div>
      {note && (
        <div className="mt-1.5 flex items-center gap-1.5 text-[11px] text-slate-500">
          {tone && <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${TONE[tone].dot}`} />}{note}
        </div>
      )}
    </Tag>
  );
}

export const pct = (v: number | null | undefined, d = 0) => (v == null ? "–" : `${v.toFixed(d)}%`);
export const mg = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v)}`);
