import { AlertOctagon, AlertTriangle, CheckCircle2, Info, Lightbulb } from "lucide-react";
import type { Finding, Severity } from "../api";

const STYLE: Record<Severity, { icon: React.ReactNode; row: string }> = {
  alert: { icon: <AlertOctagon className="h-4 w-4 text-red-600" />, row: "bg-red-50 text-red-800 ring-red-200" },
  warn: { icon: <AlertTriangle className="h-4 w-4 text-amber-600" />, row: "bg-amber-50 text-amber-800 ring-amber-200" },
  info: { icon: <Info className="h-4 w-4 text-slate-500" />, row: "bg-slate-50 text-slate-700 ring-slate-200" },
  ok: { icon: <CheckCircle2 className="h-4 w-4 text-emerald-600" />, row: "bg-emerald-50 text-emerald-700 ring-emerald-200" },
  tip: { icon: <Lightbulb className="h-4 w-4 text-brand-700" />, row: "bg-brand-50 text-brand-700 ring-brand-100" },
};

function List({ items }: { items: Finding[] }) {
  return (
    <ul className="space-y-1.5">
      {items.map((f) => (
        <li key={f.text} className={`flex items-start gap-2 rounded-xl px-3 py-2 text-sm ring-1 ring-inset ${STYLE[f.severity].row}`}>
          <span className="mt-0.5 shrink-0">{STYLE[f.severity].icon}</span>
          <span>{f.text}</span>
        </li>
      ))}
    </ul>
  );
}

/** The 30-second read: what is happening now, what is coming, and what the last 14 days show. */
export default function ClinicalSummary({ now, flags }: { now: Finding[]; flags: Finding[] }) {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Now and next 2 hours</h3>
        <List items={now} />
      </div>
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Last 14 days: findings to review</h3>
        <List items={flags} />
      </div>
    </div>
  );
}
