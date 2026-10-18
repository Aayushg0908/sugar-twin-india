import { useMemo, useState } from "react";
import { Search, Sparkles } from "lucide-react";
import type { PatientRow } from "../api";
import { Pill, TrendIcon, glucoseColor } from "../ui";

type Filter = "all" | "alerts" | "T2D" | "Prediabetes" | "Normoglycemic";
const FILTERS: { key: Filter; label: string }[] = [
  { key: "alerts", label: "Alerts" },
  { key: "all", label: "All" },
  { key: "T2D", label: "T2D" },
  { key: "Prediabetes", label: "Pre-DM" },
  { key: "Normoglycemic", label: "Normal" },
];

const hasAlert = (p: PatientRow) => p.levels.spike === "high" || p.levels.hypo === "high" || p.levels.severe === "high";

export default function PatientList({ patients, selected, onSelect }: {
  patients: PatientRow[]; selected: string | null; onSelect: (id: string) => void;
}) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("alerts");

  const rows = useMemo(() => patients.filter((p) => {
    if (q && !`${p.id} ${p.name ?? ""}`.toLowerCase().includes(q.toLowerCase())) return false;
    if (filter === "alerts") return hasAlert(p) || p.name;
    if (filter === "all") return true;
    return p.cohort === filter;
  }), [patients, q, filter]);

  const nAlerts = patients.filter(hasAlert).length;
  const nHypo = patients.filter((p) => p.levels.hypo === "high").length;

  return (
    <aside className="flex h-full w-[320px] shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="space-y-3 border-b border-slate-100 p-4">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Patients</h2>
          <span className="text-xs text-slate-400">{patients.length} monitored</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl bg-red-50 px-3 py-2">
            <div className="text-xl font-bold text-red-700">{nAlerts}</div>
            <div className="text-[11px] font-medium text-red-600">need attention now</div>
          </div>
          <div className="rounded-xl bg-blue-50 px-3 py-2">
            <div className="text-xl font-bold text-blue-700">{nHypo}</div>
            <div className="text-[11px] font-medium text-blue-600">low-sugar risk</div>
          </div>
        </div>
        <label className="flex items-center gap-2 rounded-xl bg-slate-100 px-3 py-2 text-sm">
          <Search className="h-4 w-4 text-slate-400" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search patient ID"
            className="w-full bg-transparent outline-none placeholder:text-slate-400" />
        </label>
        <div className="flex flex-wrap gap-1.5">
          {FILTERS.map((f) => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`rounded-full px-2.5 py-1 text-xs font-semibold transition ${filter === f.key ? "bg-brand-700 text-white" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
              {f.label}
            </button>
          ))}
        </div>
      </div>

      <ul className="scroll-thin flex-1 overflow-y-auto">
        {rows.map((p) => {
          const active = p.id === selected;
          return (
            <li key={p.id}>
              <button onClick={() => onSelect(p.id)}
                className={`flex w-full items-center gap-3 border-b border-slate-50 px-4 py-3 text-left transition ${active ? "bg-brand-50" : "hover:bg-slate-50"}`}>
                <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-xs font-bold ${p.name ? "bg-violet-100 text-violet-700" : "bg-slate-100 text-slate-600"}`}>
                  {p.name ? <Sparkles className="h-4 w-4" /> : `${p.sex}${p.age}`}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="truncate text-sm font-semibold text-slate-800">{p.name ?? p.id}</span>
                    {p.name && <span className="text-[10px] text-slate-400">{p.id}</span>}
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[11px] text-slate-500">
                    <span>{p.cohort === "Normoglycemic" ? "No diabetes" : p.cohort}</span>
                    <span>·</span>
                    <span>HbA1c {p.hba1c ?? "–"}</span>
                    <span>·</span>
                    <span>{Math.round(p.tir14)}% in range</span>
                  </div>
                  <div className="mt-1 flex gap-1">
                    {p.levels.spike === "high" && <Pill className="bg-amber-50 text-amber-700 ring-amber-200">Spike</Pill>}
                    {p.levels.severe === "high" && <Pill className="bg-orange-50 text-orange-700 ring-orange-200">&gt;250</Pill>}
                    {p.levels.hypo === "high" && <Pill className="bg-blue-50 text-blue-700 ring-blue-200">Low</Pill>}
                  </div>
                </div>
                <div className={`flex flex-col items-end ${glucoseColor(p.glucose)}`}>
                  <span className="text-lg font-bold leading-none">{p.glucose ?? "–"}</span>
                  <TrendIcon trend={p.trend} className="mt-0.5 h-4 w-4" />
                </div>
              </button>
            </li>
          );
        })}
        {rows.length === 0 && <li className="p-6 text-center text-sm text-slate-400">No patients match.</li>}
      </ul>
    </aside>
  );
}
