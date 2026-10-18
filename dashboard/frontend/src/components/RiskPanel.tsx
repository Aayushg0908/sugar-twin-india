import { AlertTriangle, ArrowDownCircle, Flame, Lightbulb } from "lucide-react";
import type { RiskKey, Snapshot, Suggestion } from "../api";
import { Pill, TrendIcon, glucoseColor, glucoseWord, levelStyle, trendWord } from "../ui";

const META: Record<RiskKey, { title: string; what: string; now: string; icon: React.ReactNode }> = {
  spike: { title: "Spike", what: "above 180 in the next 2 h", now: "Already above 180", icon: <AlertTriangle className="h-4 w-4" /> },
  severe: { title: "Severe high", what: "above 250 in the next 2 h", now: "Already above 250", icon: <Flame className="h-4 w-4" /> },
  hypo: { title: "Low sugar", what: "below 70 in the next 2 h", now: "Already below 70", icon: <ArrowDownCircle className="h-4 w-4" /> },
};
const LEVEL_WORD = { high: "High risk", medium: "Watch", low: "Low risk", now: "Happening now" };
const BAR = { high: "bg-red-500", medium: "bg-amber-400", low: "bg-emerald-500", now: "bg-violet-500" };

export function CurrentGlucose({ snap }: { snap: Snapshot }) {
  return (
    <div className="flex items-center gap-4 rounded-2xl bg-slate-50 p-4">
      <div className={glucoseColor(snap.glucose)}>
        <div className="flex items-end gap-1">
          <span className="text-5xl font-extrabold leading-none">{snap.glucose ?? "–"}</span>
          <span className="mb-1 text-sm font-semibold">mg/dL</span>
        </div>
      </div>
      <div className="space-y-1">
        <div className={`flex items-center gap-1 text-sm font-bold ${glucoseColor(snap.glucose)}`}>
          <TrendIcon trend={snap.trend} className="h-5 w-5" /> {glucoseWord(snap.glucose)}, {trendWord[snap.trend]}
        </div>
        <div className="text-xs text-slate-500">
          {snap.delta15 != null ? `${snap.delta15 > 0 ? "+" : ""}${Math.round(snap.delta15)} mg/dL in the last 15 min` : "No recent change data"}
        </div>
      </div>
    </div>
  );
}

export function RiskCards({ snap }: { snap: Snapshot }) {
  const order: RiskKey[] = ["spike", "hypo", "severe"];
  return (
    <div className="space-y-3">
      {order.map((k) => {
        const r = snap.risks[k];
        const m = META[k];
        const width = Math.min(100, (r.p / Math.max(r.threshold * 1.6, 0.01)) * 100);
        return (
          <div key={k} className={`rounded-2xl p-3.5 ring-1 ring-inset ${r.level === "high" ? "bg-red-50/60 ring-red-200" : "bg-white ring-slate-200"}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-sm font-bold text-slate-800">{m.icon}{m.title}</div>
              <Pill className={levelStyle[r.level]}>{LEVEL_WORD[r.level]}</Pill>
            </div>
            <div className="mt-1 text-xs text-slate-500">
              {r.level === "now" ? m.now : <>Chance of going <b>{m.what}</b>: <b className="text-slate-700">{Math.round(r.p * 100)}%</b></>}
            </div>
            {r.level !== "now" && (
              <div className="relative mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
                <div className={`h-full rounded-full ${BAR[r.level]}`} style={{ width: `${width}%` }} />
                <div className="absolute top-0 h-full w-0.5 bg-slate-500" style={{ left: `${(1 / 1.6) * 100}%` }} title="Alert threshold" />
              </div>
            )}
            {r.reasons.length > 0 && (r.level === "high" || r.level === "medium") && (
              <div className="mt-2.5">
                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Why</div>
                <ul className="mt-1 space-y-0.5">
                  {r.reasons.slice(0, 3).map((x) => (
                    <li key={x.text} className="flex gap-1.5 text-xs text-slate-600"><span className="text-slate-300">•</span>{x.text}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function Suggestions({ items, onTry }: { items: Suggestion[]; onTry: (s: Suggestion) => void }) {
  if (!items.length) {
    return <p className="text-sm text-slate-500">No action needed right now. Glucose is expected to stay in a safe range.</p>;
  }
  return (
    <ul className="space-y-2">
      {items.map((s) => (
        <li key={s.action} className="flex items-center gap-3 rounded-xl bg-slate-50 p-3">
          <Lightbulb className={`h-5 w-5 shrink-0 ${s.kind === "raise" ? "text-blue-500" : s.kind === "info" ? "text-slate-400" : "text-amber-500"}`} />
          <div className="flex-1 text-sm text-slate-700">{s.action}</div>
          {s.effect != null && (
            <span className={`whitespace-nowrap text-sm font-bold ${s.kind === "raise" ? "text-blue-600" : "text-emerald-600"}`}>
              {s.effect > 0 ? "+" : ""}{Math.round(s.effect)} mg/dL {s.kind === "lower" ? "peak" : ""}
            </span>
          )}
          {s.changes.length > 0 && (
            <button onClick={() => onTry(s)} className="rounded-lg bg-white px-2.5 py-1 text-xs font-semibold text-brand-700 ring-1 ring-brand-100 hover:bg-brand-50">
              Try it
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}
