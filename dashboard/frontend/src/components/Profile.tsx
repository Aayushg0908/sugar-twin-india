import { Activity, Dna, HeartPulse, MapPin, Timer, Utensils } from "lucide-react";
import type { Summary } from "../api";
import { Pill } from "../ui";

export function ProfileCard({ s }: { s: Summary }) {
  const hba1c = s.labs.find((l) => l.test === "HbA1c")?.value;
  const facts = [
    { icon: <MapPin className="h-3.5 w-3.5" />, text: `${s.region} India · ${s.setting}` },
    { icon: <Utensils className="h-3.5 w-3.5" />, text: `${s.diet}, ${s.staple}-based meals` },
    { icon: <Activity className="h-3.5 w-3.5" />, text: `${s.activity} lifestyle` },
    { icon: <HeartPulse className="h-3.5 w-3.5" />, text: `BMI ${s.bmi ?? "–"}${s.smoking === "Current" ? " · smoker" : ""}` },
  ];
  return (
    <div className="flex flex-wrap items-center gap-5">
      <div className={`flex h-14 w-14 items-center justify-center rounded-2xl text-lg font-extrabold ${s.name ? "bg-violet-100 text-violet-700" : "bg-brand-100 text-brand-700"}`}>
        {s.name ? s.name[0] : s.sex}
      </div>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-extrabold text-slate-900">{s.name ?? `Patient ${s.id}`}</h1>
          {s.name && <span className="text-sm text-slate-400">{s.id}</span>}
          <Pill className={s.cohort === "T2D" ? "bg-rose-50 text-rose-700 ring-rose-200" : s.cohort === "Prediabetes" ? "bg-amber-50 text-amber-700 ring-amber-200" : "bg-emerald-50 text-emerald-700 ring-emerald-200"}>
            {s.cohort === "T2D" ? `Type 2 diabetes · ${Math.round(s.diabetes_years ?? 0)} yrs` : s.cohort === "Prediabetes" ? "Prediabetes" : "No diabetes"}
          </Pill>
        </div>
        <div className="mt-0.5 text-sm text-slate-500">{s.age} yrs · {s.sex === "F" ? "Female" : "Male"}{s.family_history ? " · family history of diabetes" : ""}</div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
          {facts.map((f) => <span key={f.text} className="flex items-center gap-1">{f.icon}{f.text}</span>)}
        </div>
      </div>
      <div className="ml-auto flex gap-2">
        <div className="rounded-xl bg-slate-50 px-4 py-2 text-center">
          <div className="text-[11px] font-medium uppercase text-slate-400">HbA1c</div>
          <div className={`text-xl font-extrabold ${hba1c == null ? "text-slate-400" : hba1c >= 8 ? "text-red-600" : hba1c >= 7 ? "text-amber-600" : "text-emerald-600"}`}>{hba1c ?? "–"}%</div>
        </div>
        <div className="rounded-xl bg-slate-50 px-4 py-2 text-center">
          <div className="text-[11px] font-medium uppercase text-slate-400">14-day in range</div>
          <div className="text-xl font-extrabold text-slate-800">{Math.round(s.cgm14.bands.in_range)}%</div>
        </div>
      </div>
    </div>
  );
}

function Meter({ label, value, sub, ratio, tone }: { label: string; value: string; sub: string; ratio: number; tone: string }) {
  return (
    <div className="rounded-xl bg-slate-50 p-3">
      <div className="text-xs font-medium text-slate-500">{label}</div>
      <div className="text-lg font-bold text-slate-800">{value}</div>
      <div className="relative mt-1.5 h-1.5 rounded-full bg-slate-200">
        <div className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-slate-400" style={{ left: "50%" }} title="Population average" />
        <div className={`absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-white ${tone}`}
          style={{ left: `${Math.min(95, Math.max(5, ratio * 50))}%` }} />
      </div>
      <div className="mt-1.5 text-[11px] text-slate-500">{sub}</div>
    </div>
  );
}

export function TwinCard({ s }: { s: Summary }) {
  const t = s.twin;
  const word = { high: "more sensitive than average", low: "less sensitive than average", average: "about average" };
  const rec = { slow: "slower than average", fast: "faster than average", average: "about average" };
  const walk = { strong: "helps more than average", weak: "helps less than average", average: "average benefit" };
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      <Meter label="Carb sensitivity" value={`+${t.p10} mg/dL`} sub={`per 10 g carbs · ${word[t.carb_label]}`} ratio={t.p10 / t.pop_p10}
        tone={t.carb_label === "high" ? "bg-red-500" : t.carb_label === "low" ? "bg-emerald-500" : "bg-slate-500"} />
      <Meter label="Recovery after meals" value={`${t.tau} min`} sub={`time constant · ${rec[t.recovery_label]}`} ratio={t.tau / t.pop_tau}
        tone={t.recovery_label === "slow" ? "bg-red-500" : t.recovery_label === "fast" ? "bg-emerald-500" : "bg-slate-500"} />
      <Meter label="Walking effect" value={`${t.w.toFixed(2)}×`} sub={walk[t.walk_label]} ratio={t.w}
        tone={t.walk_label === "strong" ? "bg-emerald-500" : t.walk_label === "weak" ? "bg-red-500" : "bg-slate-500"} />
    </div>
  );
}

export const TwinIcon = () => <Dna className="h-4 w-4" />;
export const ClockIcon = () => <Timer className="h-4 w-4" />;
