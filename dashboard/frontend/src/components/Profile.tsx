import { Activity, Dna, Droplet, HeartPulse, MapPin, Target, Timer, Utensils } from "lucide-react";
import type { Summary } from "../api";
import { Pill, StatTile } from "../ui";

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
      <div className={`flex h-14 w-14 items-center justify-center rounded-xl text-lg font-extrabold ${s.name ? "bg-blue-100 text-blue-700" : "bg-brand-100 text-brand-700"}`}>
        {s.name ? s.name[0] : s.sex}
      </div>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-extrabold text-slate-900">{s.name ?? `Patient ${s.id}`}</h1>
          {s.name && <span className="text-sm text-slate-400">{s.id}</span>}
          <Pill className="bg-slate-100 text-slate-700 ring-slate-200">
            {s.cohort === "T2D" ? `Type 2 diabetes · ${Math.round(s.diabetes_years ?? 0)} yrs` : s.cohort === "Prediabetes" ? "Prediabetes" : "No diabetes"}
          </Pill>
        </div>
        <div className="mt-0.5 text-sm text-slate-500">{s.age} yrs · {s.sex === "F" ? "Female" : "Male"}{s.family_history ? " · family history of diabetes" : ""}</div>
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
          {facts.map((f) => <span key={f.text} className="flex items-center gap-1">{f.icon}{f.text}</span>)}
        </div>
      </div>
      <div className="ml-auto flex gap-3">
        <StatTile className="min-w-[150px]" icon={<Droplet />} label="HbA1c" value={hba1c == null ? "–" : `${hba1c}%`}
          tone={hba1c == null ? null : hba1c >= 8 ? "red" : hba1c >= 7 ? "amber" : "green"}
          note={hba1c == null ? "no result" : hba1c >= 8 ? "over 8%" : hba1c >= 7 ? "above 7% target" : "on target"} />
        <StatTile className="min-w-[150px]" icon={<Target />} label="14-day in range" value={`${Math.round(s.cgm14.bands.in_range)}%`}
          tone={s.cgm14.bands.in_range > 70 ? "green" : "amber"} note="target over 70%" />
      </div>
    </div>
  );
}

function Meter({ label, value, sub, ratio, tone }: { label: string; value: string; sub: string; ratio: number; tone: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
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
  const rise = { high: "more than average", low: "less than average", average: "about average" }[t.carb_label];
  const rec = { slow: "slower than average", fast: "faster than average", average: "about average" }[t.recovery_label];
  const walk = { strong: "Helps a lot", weak: "Helps less", average: "Helps" }[t.walk_label];
  return (
    <div className="grid gap-2 sm:grid-cols-3">
      <Meter label="Sugar rise per roti (~15 g carbs)" value={`+${t.rise_per_roti} mg/dL`} sub={`${rise} (average +${t.pop_rise_per_roti})`}
        ratio={t.rise_per_roti / t.pop_rise_per_roti}
        tone={t.carb_label === "high" ? "bg-amber-500" : t.carb_label === "low" ? "bg-green-500" : "bg-slate-500"} />
      <Meter label="Back to normal after a meal" value={`~${t.recovery_hours} h`} sub={`${rec} (average ${t.pop_recovery_hours} h)`}
        ratio={t.recovery_hours / t.pop_recovery_hours}
        tone={t.recovery_label === "slow" ? "bg-amber-500" : t.recovery_label === "fast" ? "bg-green-500" : "bg-slate-500"} />
      <Meter label="Walking after meals" value={walk} sub={`${t.w.toFixed(2)}× the average effect`} ratio={t.w}
        tone={t.walk_label === "strong" ? "bg-green-500" : t.walk_label === "weak" ? "bg-amber-500" : "bg-slate-500"} />
    </div>
  );
}

export function twinSentence(s: Summary) {
  const t = s.twin;
  const first = s.name ? s.name.split(" ")[0] : "This patient";
  return `${first}'s sugar rises about ${t.rise_per_roti} mg/dL for every roti-sized portion of carbs and takes ~${t.recovery_hours} h `
    + `to settle after a meal. A 20-minute walk after eating ${t.walk_label === "weak" ? "helps a little" : "helps lower the peak"}.`;
}

export const TwinIcon = () => <Dna className="h-4 w-4" />;
export const ClockIcon = () => <Timer className="h-4 w-4" />;
