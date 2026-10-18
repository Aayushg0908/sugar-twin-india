import { Area, Bar, BarChart, CartesianGrid, ComposedChart, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, XCircle } from "lucide-react";
import type { Accuracy, Summary } from "../api";
import { COLORS, Stat, dayLabel } from "../ui";

const BANDS: { key: keyof Summary["cgm14"]["bands"]; label: string; color: string; target: string; ok: (v: number) => boolean }[] = [
  { key: "very_high", label: "Very high >250", color: COLORS.veryHigh, target: "< 5%", ok: (v) => v < 5 },
  { key: "high", label: "High 181–250", color: COLORS.high, target: "< 25% (with very high)", ok: () => true },
  { key: "in_range", label: "In range 70–180", color: COLORS.inRange, target: "> 70%", ok: (v) => v > 70 },
  { key: "low", label: "Low 54–69", color: COLORS.low, target: "< 4% (with very low)", ok: () => true },
  { key: "very_low", label: "Very low <54", color: COLORS.veryLow, target: "< 1%", ok: (v) => v < 1 },
];

export function TimeInRange({ s }: { s: Summary }) {
  const b = s.cgm14.bands;
  return (
    <div className="flex gap-4">
      <div className="flex h-48 w-9 shrink-0 flex-col overflow-hidden rounded-lg">
        {BANDS.map((x) => <div key={x.key} style={{ height: `${Math.max(b[x.key], 0.4)}%`, background: x.color }} title={`${x.label}: ${b[x.key]}%`} />)}
      </div>
      <ul className="flex h-48 flex-1 flex-col justify-between py-1">
        {BANDS.map((x) => (
          <li key={x.key} className="flex items-center justify-between gap-2 text-sm" title={`Target ${x.target}`}>
            <span className="flex items-center gap-2 whitespace-nowrap"><span className="h-2.5 w-2.5 rounded-full" style={{ background: x.color }} />{x.label}</span>
            <span className="flex items-center gap-1.5">
              <b className="text-slate-800">{b[x.key].toFixed(1)}%</b>
              {x.key === "in_range" || x.key === "very_high" || x.key === "very_low"
                ? x.ok(b[x.key]) ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> : <XCircle className="h-4 w-4 text-red-400" />
                : <span className="w-4" />}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function DailyPattern({ s }: { s: Summary }) {
  const data = s.agp.map((a) => ({ ...a, outer: [a.p5, a.p95], inner: [a.p25, a.p75] }));
  return (
    <div className="h-56">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -14 }}>
          <CartesianGrid stroke="#f1f5f9" vertical={false} />
          <ReferenceArea y1={70} y2={180} fill={COLORS.inRange} fillOpacity={0.07} />
          <ReferenceLine y={180} stroke={COLORS.high} strokeDasharray="4 4" />
          <ReferenceLine y={70} stroke={COLORS.low} strokeDasharray="4 4" />
          <XAxis dataKey="hour" ticks={[0, 4, 8, 12, 16, 20]} tickFormatter={(h) => `${h % 12 || 12}${h < 12 ? "am" : "pm"}`} />
          <YAxis domain={[40, 350]} ticks={[70, 180, 250]} width={46} />
          <Tooltip formatter={(v: number | number[], n: string) => [Array.isArray(v) ? `${Math.round(v[0])}–${Math.round(v[1])}` : Math.round(v),
            n === "outer" ? "5–95% of days" : n === "inner" ? "25–75% of days" : "Typical (median)"]}
            labelFormatter={(h) => `${h}:00 – ${h}:59`} contentStyle={{ borderRadius: 12, fontSize: 12 }} />
          <Area dataKey="outer" stroke="none" fill={COLORS.brand} fillOpacity={0.12} isAnimationActive={false} />
          <Area dataKey="inner" stroke="none" fill={COLORS.brand} fillOpacity={0.28} isAnimationActive={false} />
          <Line dataKey="p50" stroke={COLORS.brand} strokeWidth={2.5} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function DailyTir({ s }: { s: Summary }) {
  return (
    <div className="h-28">
      <ResponsiveContainer>
        <BarChart data={s.daily} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <XAxis dataKey="date" tickFormatter={(d) => dayLabel(d).split(",")[0].slice(0, 2)} interval={0} />
          <YAxis domain={[0, 100]} ticks={[70]} width={40} />
          <ReferenceLine y={70} stroke="#94a3b8" strokeDasharray="3 3" />
          <Tooltip formatter={(v: number) => [`${Math.round(v)}%`, "In range"]} labelFormatter={(d) => dayLabel(Number(d))}
            contentStyle={{ borderRadius: 12, fontSize: 12 }} />
          <Bar dataKey="tir" fill={COLORS.inRange} radius={[4, 4, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CgmStats({ s }: { s: Summary }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      <Stat label="Average" value={`${Math.round(s.cgm14.mean)}`} sub="mg/dL" />
      <Stat label="GMI (est. HbA1c)" value={`${s.cgm14.gmi.toFixed(1)}%`} sub="from CGM" />
      <Stat label="Variability (CV)" value={`${Math.round(s.cgm14.cv)}%`} sub={s.cgm14.cv <= 36 ? "stable" : "too variable"} />
      <Stat label="Sensor wear" value={`${Math.round(s.cgm14.coverage)}%`} sub="of 14 days" />
    </div>
  );
}

export function EhrPanel({ s }: { s: Summary }) {
  const order = ["HbA1c", "Fasting plasma glucose", "LDL cholesterol", "Triglycerides", "eGFR (CKD-EPI 2021)", "Systolic blood pressure", "Body weight"];
  const labs = order.map((t) => s.labs.find((l) => l.test === t)).filter(Boolean) as Summary["labs"];
  const short: Record<string, string> = { "Fasting plasma glucose": "Fasting glucose", "eGFR (CKD-EPI 2021)": "eGFR (kidney)", "Systolic blood pressure": "BP (systolic)" };
  return (
    <div className="grid gap-5 lg:grid-cols-3">
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Latest labs</h3>
        <ul className="divide-y divide-slate-100 text-sm">
          {labs.map((l) => (
            <li key={l.test} className="flex justify-between py-1.5">
              <span className="text-slate-600">{short[l.test] ?? l.test}</span>
              <span className="font-semibold text-slate-800">{l.value} <span className="text-xs font-normal text-slate-400">{l.unit}</span></span>
            </li>
          ))}
        </ul>
        {s.hba1c_history.length > 1 && (
          <div className="mt-3 h-20">
            <ResponsiveContainer>
              <LineChart data={s.hba1c_history} margin={{ top: 4, right: 4, bottom: 0, left: -28 }}>
                <YAxis domain={["dataMin - 0.3", "dataMax + 0.3"]} hide />
                <ReferenceLine y={7} stroke="#94a3b8" strokeDasharray="3 3" />
                <Tooltip formatter={(v: number) => [`${v}%`, "HbA1c"]} labelFormatter={() => ""} contentStyle={{ borderRadius: 12, fontSize: 12 }} />
                <Line dataKey="value" stroke="#e11d48" strokeWidth={2} dot={{ r: 2 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
            <div className="text-center text-[11px] text-slate-400">HbA1c over the last 3 years (dashed = 7% target)</div>
          </div>
        )}
      </div>
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Current medicines</h3>
        <ul className="space-y-1.5 text-sm">
          {s.medications.map((m) => (
            <li key={m.drug + m.dose} className="rounded-lg bg-slate-50 px-3 py-1.5">
              <div className="font-semibold text-slate-800">{m.drug} <span className="font-normal text-slate-500">{m.dose}</span></div>
              <div className="text-[11px] text-slate-400">{m.class} · {m.for}</div>
            </li>
          ))}
          {s.medications.length === 0 && <li className="text-slate-400">None</li>}
        </ul>
      </div>
      <div>
        <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Diagnoses</h3>
        <ul className="space-y-1.5 text-sm">
          {s.diagnoses.map((d) => (
            <li key={d.code + d.name} className="flex justify-between gap-2 rounded-lg bg-slate-50 px-3 py-1.5">
              <span className="text-slate-700">{d.name}</span><span className="text-[11px] text-slate-400">{d.code}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export function AccuracyStrip({ a }: { a: Accuracy }) {
  const items = [
    { v: `${Math.round(a.spikes_caught * 100)}%`, l: "of glucose spikes flagged in advance", s: "up to 2 h early" },
    { v: `${Math.round(a.lows_caught * 100)}%`, l: "of low-sugar episodes flagged in advance", s: "up to 2 h early" },
    { v: a.spike_auroc.toFixed(3), l: "spike-alert AUROC", s: `${a.spike_auroc_t2d.toFixed(3)} for T2D patients` },
    { v: `±${Math.round(a.mae60)}`, l: "mg/dL forecast error, 1 h ahead", s: `±${Math.round(a.mae120)} at 2 h` },
    { v: `${Math.round(a.twin_peak_err)}`, l: "mg/dL error predicting a new meal's peak", s: `vs ${Math.round(a.population_peak_err)} one-size-fits-all` },
    { v: a.cgm_only_auroc.toFixed(3), l: "AUROC with CGM only", s: "works without a smartwatch" },
  ];
  return (
    <div className="grid gap-3 sm:grid-cols-3 xl:grid-cols-6">
      {items.map((x) => (
        <div key={x.l} className="rounded-xl bg-white/70 p-3 ring-1 ring-slate-200">
          <div className="text-2xl font-extrabold text-brand-700">{x.v}</div>
          <div className="text-xs font-medium text-slate-600">{x.l}</div>
          <div className="text-[11px] text-slate-400">{x.s}</div>
        </div>
      ))}
    </div>
  );
}
