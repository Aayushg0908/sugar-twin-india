import { useMemo, useState } from "react";
import { Search, Sparkles } from "lucide-react";
import type { PatientRow } from "../api";
import { Pill, TrendIcon, glucoseColor } from "../ui";

type Filter = "all" | "urgent" | "watch" | "T2D" | "Prediabetes" | "Normoglycemic";
const FILTERS: { key: Filter; label: string }[] = [
  { key: "urgent", label: "Urgent" },
  { key: "watch", label: "Watch" },
  { key: "all", label: "All" },
  { key: "T2D", label: "T2D" },
  { key: "Prediabetes", label: "Pre-DM" },
  { key: "Normoglycemic", label: "Normal" },
];


export default function PatientList({ patients, selected, onSelect }: {
  patients: PatientRow[]; selected: string | null; onSelect: (id: string) => void;
}) {
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState<Filter>("urgent");

  const rows = useMemo(() => patients.filter((p) => {
    if (q && !`${p.id} ${p.name ?? ""}`.toLowerCase().includes(q.toLowerCase())) return false;
    if (filter === "urgent" || filter === "watch") return p.urgency === filter || p.name;
    if (filter === "all") return true;
    return p.cohort === filter;
  }), [patients, q, filter]);

  const nUrgent = patients.filter((p) => p.urgency === "urgent").length;
  const nWatch = patients.filter((p) => p.urgency === "watch").length;

  return (
    <aside className="flex h-full w-[320px] shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="space-y-3 border-b border-slate-100 p-4">
        <div className="flex items-baseline justify-between">
          <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Patients</h2>
          <span className="text-xs text-slate-400">{patients.length} monitored</span>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <button onClick={() => setFilter("urgent")} className="rounded-xl bg-red-50 px-3 py-2 text-left">
            <div className="text-xl font-bold text-red-700">{nUrgent}</div>
            <div className="text-[11px] font-medium text-red-600">urgent: low sugar or above 250 likely</div>
          </button>
          <button onClick={() => setFilter("watch")} className="rounded-xl bg-amber-50 px-3 py-2 text-left">
            <div className="text-xl font-bold text-amber-700">{nWatch}</div>
            <div className="text-[11px] font-medium text-amber-600">watch: spike above 180 likely</div>
          </button>
        </div>
        <label className="flex items-center gap-2 rounded-xl bg-slate-100 px-3 py-2 text-sm">
          <Search className="h-4 w-4 text-slate-400" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search patient ID"
            className="w-full bg-transparent outline-none placeholder:text-slate-400" />
        </label>
        <div className="flex flex-wrap gap-1.5">
          {FILTERS.map((f) => (
            <button key={f.key} onClick={() => setFilter(f.key)}
              className={`rounded-full px-2.5 py-1 text-xs font-semibold transition ${filter === f.key ? "bg-brand-600 text-[#fff]" : "bg-slate-100 text-slate-600 hover:bg-slate-200"}`}>
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
                  <div className="mt-0.5 flex items-center gap-1.5 whitespace-nowrap text-[11px] text-slate-500">
                    <span>{p.cohort === "Normoglycemic" ? "No diabetes" : p.cohort}</span>
                    {p.hba1c != null && <><span>·</span><span>HbA1c {p.hba1c}</span></>}
                    <span>·</span>
                    <span>{Math.round(p.tir14)}% in range</span>
                  </div>
                  <div className="mt-1 flex gap-1">
                    {p.levels.hypo === "high" && <Pill className="bg-red-50 text-red-700 ring-red-200">Low sugar likely</Pill>}
                    {p.levels.severe === "high" && <Pill className="bg-orange-50 text-orange-700 ring-orange-200">Above 250 likely</Pill>}
                    {p.levels.spike === "high" && p.levels.severe !== "high" && <Pill className="bg-amber-50 text-amber-700 ring-amber-200">Spike likely</Pill>}
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
