import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Minus, TrendingDown, TrendingUp } from "lucide-react";
import type { Accuracy, Summary } from "../api";
import { useChartTheme, useTheme } from "../theme";

export function EhrPanel({ s }: { s: Summary }) {
  const order = ["HbA1c", "Fasting plasma glucose", "LDL cholesterol", "Triglycerides", "eGFR (CKD-EPI 2021)", "Systolic blood pressure", "Body weight"];
  const labs = order.map((t) => s.labs.find((l) => l.test === t)).filter(Boolean) as Summary["labs"];
  const short: Record<string, string> = { "Fasting plasma glucose": "Fasting glucose", "eGFR (CKD-EPI 2021)": "eGFR (kidney)", "Systolic blood pressure": "BP (systolic)" };
  return (
    <div className="space-y-5">
    <div className="grid gap-5 lg:grid-cols-3">
      <div>
        <h3 className="mb-2 text-xs font-semibold text-slate-400">Latest labs</h3>
        <ul className="divide-y divide-slate-100 text-sm">
          {labs.map((l) => (
            <li key={l.test} className="flex justify-between py-1.5">
              <span className="text-slate-600">{short[l.test] ?? l.test}</span>
              <span className="font-semibold text-slate-800">{l.value} <span className="text-xs font-normal text-slate-400">{l.unit}</span></span>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <h3 className="mb-2 text-xs font-semibold text-slate-400">Current medicines</h3>
        <ul className="space-y-1.5 text-sm">
          {s.medications.map((m) => (
            <li key={m.drug + m.dose} className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5">
              <div className="font-semibold text-slate-800">{m.drug} <span className="font-normal text-slate-500">{m.dose}</span></div>
              <div className="text-[11px] text-slate-400">{m.class} · {m.for}</div>
            </li>
          ))}
          {s.medications.length === 0 && <li className="text-slate-400">None</li>}
        </ul>
      </div>
      <div>
        <h3 className="mb-2 text-xs font-semibold text-slate-400">Diagnoses</h3>
        <ul className="space-y-1.5 text-sm">
          {s.diagnoses.map((d) => (
            <li key={d.code + d.name} className="flex justify-between gap-2 rounded-lg border border-slate-200 bg-slate-50 px-3 py-1.5">
              <span className="text-slate-700">{d.name}</span><span className="text-[11px] text-slate-400">{d.code}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
    {s.hba1c_history.length > 1 && <HbA1cTrend history={s.hba1c_history} />}
    </div>
  );
}

const fmtMonth = new Intl.DateTimeFormat("en-IN", { month: "short", year: "numeric", timeZone: "UTC" });
const monthLabel = (ms: number) => fmtMonth.format(new Date(ms));

/** HbA1c control bands (ADA): the band carries the status, the line stays neutral ink. */
const BANDS_A1C = [
  { from: 0, to: 7, role: "inRange", label: "On target (under 7%)" },
  { from: 7, to: 8, role: "high", label: "Above target (7–8%)" },
  { from: 8, to: 20, role: "low", label: "Poorly controlled (over 8%)" },
] as const;
const bandOf = (v: number) => BANDS_A1C.find((b) => v < b.to) ?? BANDS_A1C[2];

/** HbA1c history with real axes, control bands, the 7% target and the change since the first test. */
function HbA1cTrend({ history }: { history: Summary["hba1c_history"] }) {
  const th = useChartTheme();
  const bandOpacity = useTheme().theme === "dark" ? 0.18 : 0.12;   // dark surface needs stronger tints to tell bands apart
  const pts = history.filter((h): h is { date: number; value: number } => h.value != null).sort((a, b) => a.date - b.date);
  if (pts.length < 2) return null;
  const first = pts[0], last = pts[pts.length - 1];
  const change = last.value - first.value;
  const lo = Math.floor(Math.min(6, ...pts.map((p) => p.value)) * 2) / 2 - 0.5;
  const hi = Math.ceil(Math.max(9, ...pts.map((p) => p.value)) * 2) / 2 + 0.5;
  const ticks: number[] = [];
  for (let v = Math.ceil(lo); v <= hi; v += 1) ticks.push(v);
  const band = bandOf(last.value);
  const Arrow = change > 0.05 ? TrendingUp : change < -0.05 ? TrendingDown : Minus;
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h3 className="text-xs font-semibold text-slate-400">HbA1c trend ({pts.length} tests)</h3>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-3xl font-extrabold text-slate-900">{last.value.toFixed(1)}%</span>
            <span className="text-xs text-slate-500">latest, {monthLabel(last.date)}</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="flex items-center gap-1.5 rounded-full bg-white px-3 py-1 font-semibold text-slate-700 ring-1 ring-slate-200">
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: th[band.role] }} />{band.label}
          </span>
          <span className="flex items-center gap-1 rounded-full bg-white px-3 py-1 font-semibold text-slate-700 ring-1 ring-slate-200"
            title="Lower is better">
            <Arrow className="h-4 w-4" />
            {change > 0 ? "+" : ""}{change.toFixed(1)} points since {monthLabel(first.date)}
            <span className="font-normal text-slate-500">({change > 0.05 ? "worse" : change < -0.05 ? "better" : "stable"})</span>
          </span>
        </div>
      </div>
      <div className="h-56">
        <ResponsiveContainer>
          <LineChart data={pts} margin={{ top: 10, right: 16, bottom: 4, left: -8 }}>
            {BANDS_A1C.map((b) => (
              <ReferenceArea key={b.label} y1={Math.max(b.from, lo)} y2={Math.min(b.to, hi)} fill={th[b.role]} fillOpacity={bandOpacity}
                ifOverflow="hidden" />
            ))}
            <CartesianGrid stroke={th.grid} strokeDasharray="2 4" vertical={false} />
            <ReferenceLine y={7} stroke={th.inRange} strokeDasharray="5 4" strokeWidth={1.5}
              label={{ value: "7% target", position: "insideTopLeft", fill: th.muted, fontSize: 11 }} />
            <XAxis dataKey="date" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={monthLabel}
              ticks={pts.map((p) => p.date)} interval="preserveStartEnd" minTickGap={24} />
            <YAxis domain={[lo, hi]} ticks={ticks} tickFormatter={(v) => `${v}%`} width={44} />
            <Tooltip labelFormatter={(v) => monthLabel(Number(v))}
              formatter={(v: number) => [`${v.toFixed(1)}% · ${bandOf(v).label}`, "HbA1c"]}
              contentStyle={{ borderRadius: 12, fontSize: 12, ...th.tooltip }} labelStyle={{ color: th.muted }}
              cursor={{ stroke: th.muted, strokeDasharray: "3 3" }} />
            <Line dataKey="value" stroke={th.ink} strokeWidth={2} isAnimationActive={false}
              dot={{ r: 4, fill: th.surface, stroke: th.ink, strokeWidth: 2 }}
              activeDot={{ r: 6, fill: th.ink, stroke: th.surface, strokeWidth: 2 }}
              label={(p: { x?: number; y?: number; index?: number }) => p.index === pts.length - 1
                ? <text key="last" x={p.x} y={(p.y ?? 0) - 12} textAnchor="middle" fontSize={12} fontWeight={700} fill={th.ink}>{last.value.toFixed(1)}%</text>
                : <g key={p.index} />} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
        {BANDS_A1C.map((b) => (
          <span key={b.label} className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm" style={{ background: th[b.role], opacity: 0.35 }} />{b.label}</span>
        ))}
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
