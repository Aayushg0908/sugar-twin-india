import { AlertTriangle, ArrowDownCircle, Flame, Lightbulb } from "lucide-react";
import type { Risk, Snapshot, Suggestion } from "../api";
import { Pill, TrendIcon, glucoseColor, glucoseWord, levelStyle, trendWord } from "../ui";

const LEVEL_WORD = { high: "Likely", medium: "Possible", low: "Unlikely", now: "Happening now" };
const BAR = { high: "bg-red-500", medium: "bg-amber-400", low: "bg-green-500", now: "bg-blue-500" };

export function CurrentGlucose({ snap }: { snap: Snapshot }) {
  return (
    <div className="flex items-center gap-4 rounded-xl bg-slate-50 p-4">
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

function RiskCard({ icon, title, risk, what, nowText, extra }: {
  icon: React.ReactNode; title: string; risk: Risk; what: string; nowText: string; extra?: React.ReactNode;
}) {
  const width = Math.min(100, (risk.p / Math.max(risk.threshold * 1.6, 0.01)) * 100);
  const urgent = risk.level === "high" || risk.level === "now";
  return (
    <div className={`rounded-xl p-3.5 ring-1 ring-inset ${urgent ? "bg-red-50 ring-red-200" : "bg-white ring-slate-200"}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-bold text-slate-800">{icon}{title}</div>
        <Pill className={levelStyle[risk.level]}>{LEVEL_WORD[risk.level]}</Pill>
      </div>
      <div className="mt-1 text-xs text-slate-500">
        {risk.level === "now" ? nowText : <>{what} <span className="text-slate-400">({Math.round(risk.p * 100)}% chance)</span></>}
      </div>
      {risk.level !== "now" && (
        <div className="relative mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
          <div className={`h-full rounded-full ${BAR[risk.level]}`} style={{ width: `${width}%` }} />
          <div className="absolute top-0 h-full w-0.5 bg-slate-500" style={{ left: `${(1 / 1.6) * 100}%` }} title="Alert threshold" />
        </div>
      )}
      {extra}
      {risk.reasons.length > 0 && (risk.level === "high" || risk.level === "medium") && (
        <ul className="mt-2.5 space-y-0.5">
          {risk.reasons.slice(0, 3).map((x) => (
            <li key={x.text} className="flex gap-1.5 text-xs text-slate-600"><span className="text-slate-400">•</span>{x.text}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Low sugar first (more dangerous), then one combined high-sugar card (above 180 / above 250). */
export function RiskCards({ snap }: { snap: Snapshot }) {
  const { spike, severe, hypo } = snap.risks;
  const highMain = severe.level === "high" || severe.level === "now" ? severe : spike;
  const severeLine = highMain === spike && spike.level !== "now" && (
    <div className="mt-2 flex items-center justify-between rounded-lg bg-slate-50 px-2.5 py-1 text-xs">
      <span className="text-slate-500">Above 250 (severe)</span>
      <Pill className={levelStyle[severe.level]}>{LEVEL_WORD[severe.level]}</Pill>
    </div>
  );
  return (
    <div className="space-y-3">
      <RiskCard icon={<ArrowDownCircle className="h-4 w-4" />} title="Low sugar" risk={hypo}
        what="Going below 70 in the next 2 h" nowText="Below 70 right now: treat immediately" />
      <RiskCard icon={highMain === severe ? <Flame className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
        title={highMain === severe ? "High sugar (above 250)" : "High sugar"} risk={highMain}
        what={highMain === severe ? "Going above 250 in the next 2 h" : "Going above 180 in the next 2 h"}
        nowText={highMain === severe ? "Above 250 right now" : "Above 180 right now"} extra={severeLine} />
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
          <Lightbulb className={`h-5 w-5 shrink-0 ${s.kind === "raise" ? "text-blue-500" : s.kind === "info" ? "text-slate-400" : "text-brand-600"}`} />
          <div className="flex-1 text-sm text-slate-700">{s.action}</div>
          {s.effect != null && (
            <span className={`whitespace-nowrap text-sm font-bold ${s.kind === "raise" ? "text-blue-600" : "text-green-600"}`}>
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
