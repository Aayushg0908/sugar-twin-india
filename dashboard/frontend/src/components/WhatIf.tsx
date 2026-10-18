import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Footprints, Pill as PillIcon, RotateCcw, UtensilsCrossed } from "lucide-react";
import { api, isoNaive, type Change, type WhatIfResult } from "../api";
import { COLORS, hhmm } from "../ui";

export interface WhatIfState {
  meal_carbs: number;
  meal_gi: "low" | "medium" | "high";
  meal_in: number;
  walk_minutes: number;
  walk_in: number;
  skip: string[];
  ref: Change[] | null;          // optional comparison scenario (e.g. "usual meal") set by a suggestion
  refLabel: string | null;
}
export const EMPTY_WHATIF: WhatIfState = {
  meal_carbs: 0, meal_gi: "high", meal_in: 0, walk_minutes: 0, walk_in: 30, skip: [], ref: null, refLabel: null,
};

function toChanges(s: WhatIfState): Change[] {
  const out: Change[] = [];
  if (s.meal_carbs > 0) out.push(["meal", s.meal_in, s.meal_carbs, s.meal_gi]);
  if (s.walk_minutes > 0) out.push(["walk", s.walk_in, s.walk_minutes]);
  s.skip.forEach((c) => out.push(["skip_drug", c]));
  return out;
}

const FOODS: { gi: WhatIfState["meal_gi"]; label: string; sub: string }[] = [
  { gi: "low", label: "Roti / dal / sabzi", sub: "low GI, slow" },
  { gi: "medium", label: "Mixed thali / upma", sub: "medium GI" },
  { gi: "high", label: "Rice / idli / sweets", sub: "high GI, fast" },
];
const PRESETS: { label: string; v: Partial<WhatIfState> }[] = [
  { label: "Rice lunch (85 g)", v: { meal_carbs: 85, meal_gi: "high", meal_in: 0 } },
  { label: "Roti-dal (60 g)", v: { meal_carbs: 60, meal_gi: "low", meal_in: 0 } },
  { label: "Biryani (110 g)", v: { meal_carbs: 110, meal_gi: "high", meal_in: 0 } },
  { label: "Chai + biscuits (25 g)", v: { meal_carbs: 25, meal_gi: "high", meal_in: 0 } },
];
const DRUG_NAMES: Record<string, string> = {
  Biguanide: "Metformin", Sulfonylurea: "Sulfonylurea", "DPP-4 inhibitor": "DPP-4 inhibitor",
  "SGLT2 inhibitor": "SGLT2 inhibitor", "Basal insulin": "Basal insulin",
};

function Slider({ label, value, min, max, step, unit, onChange }: {
  label: string; value: number; min: number; max: number; step: number; unit: string; onChange: (v: number) => void;
}) {
  return (
    <label className="block">
      <div className="flex justify-between text-xs"><span className="font-medium text-slate-600">{label}</span>
        <span className="font-bold text-slate-800">{value} {unit}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} className="mt-1 w-full" />
    </label>
  );
}

export default function WhatIf({ pid, time, drugClasses, state, setState }: {
  pid: string; time: number; drugClasses: string[]; state: WhatIfState; setState: (s: WhatIfState) => void;
}) {
  const [res, setRes] = useState<WhatIfResult | null>(null);
  const set = (v: Partial<WhatIfState>) => setState({ ...state, ...v });
  const active = state.meal_carbs > 0 || state.walk_minutes > 0 || state.skip.length > 0;

  useEffect(() => {
    const ctl = new AbortController();
    const id = setTimeout(() => {
      api.whatif(pid, { t: isoNaive(time), changes: toChanges(state), reference: state.ref }, ctl.signal).then(setRes).catch(() => {});
    }, 200);
    return () => { clearTimeout(id); ctl.abort(); };
  }, [pid, time, state]);

  const s = res?.summary;
  const cmp = s?.reference ?? s?.baseline;                  // compare against the reference scenario if there is one
  const cmpLabel = s?.reference ? (state.refLabel ?? "reference") : "no change";
  const dPeak = s && cmp ? s.scenario.peak - cmp.peak : 0;
  const yMax = res ? Math.max(260, ...res.curve.map((c) => Math.max(c.base, c.scenario, c.reference ?? 0))) + 10 : 300;

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,320px)_1fr]">
      <div className="space-y-4">
        <div>
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700"><UtensilsCrossed className="h-4 w-4" /> Meal</div>
          <div className="mb-2 flex flex-wrap gap-1.5">
            {PRESETS.map((p) => (
              <button key={p.label} onClick={() => set(p.v)}
                className="rounded-full bg-amber-50 px-2.5 py-1 text-xs font-semibold text-amber-800 ring-1 ring-amber-200 hover:bg-amber-100">{p.label}</button>
            ))}
          </div>
          <Slider label="Carbohydrates" value={state.meal_carbs} min={0} max={150} step={5} unit="g" onChange={(v) => set({ meal_carbs: v })} />
          <div className="mt-2 grid grid-cols-3 gap-1.5">
            {FOODS.map((f) => (
              <button key={f.gi} onClick={() => set({ meal_gi: f.gi })}
                className={`rounded-xl p-2 text-left text-xs ring-1 transition ${state.meal_gi === f.gi ? "bg-brand-50 ring-brand-500" : "bg-white ring-slate-200 hover:bg-slate-50"}`}>
                <div className="font-semibold text-slate-700">{f.label}</div>
                <div className="text-[10px] text-slate-400">{f.sub}</div>
              </button>
            ))}
          </div>
          <div className="mt-2"><Slider label="Eat in" value={state.meal_in} min={0} max={90} step={15} unit="min" onChange={(v) => set({ meal_in: v })} /></div>
        </div>

        <div>
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700"><Footprints className="h-4 w-4" /> Walk</div>
          <Slider label="Duration" value={state.walk_minutes} min={0} max={45} step={5} unit="min" onChange={(v) => set({ walk_minutes: v })} />
          <Slider label="Start in" value={state.walk_in} min={0} max={90} step={15} unit="min" onChange={(v) => set({ walk_in: v })} />
        </div>

        {drugClasses.length > 0 && (
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-700"><PillIcon className="h-4 w-4" /> Skip today's dose</div>
            <div className="flex flex-wrap gap-1.5">
              {drugClasses.map((c) => {
                const on = state.skip.includes(c);
                return (
                  <button key={c} onClick={() => set({ skip: on ? state.skip.filter((x) => x !== c) : [...state.skip, c] })}
                    className={`rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ${on ? "bg-red-50 text-red-700 ring-red-300" : "bg-white text-slate-600 ring-slate-200 hover:bg-slate-50"}`}>
                    {on ? "Skipped: " : ""}{DRUG_NAMES[c] ?? c}
                  </button>
                );
              })}
            </div>
          </div>
        )}
        {active && (
          <button onClick={() => setState(EMPTY_WHATIF)} className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-slate-800">
            <RotateCcw className="h-3.5 w-3.5" /> Reset scenario
          </button>
        )}
      </div>

      <div>
        <div className="mb-3 grid grid-cols-3 gap-2">
          <div className="rounded-xl bg-slate-50 p-3">
            <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Highest glucose</div>
            <div className="text-lg font-bold text-slate-800">{s ? Math.round(s.scenario.peak) : "–"} <span className="text-xs font-medium text-slate-400">mg/dL</span></div>
            {active && s && <div className={`text-xs font-semibold ${dPeak > 0 ? "text-red-600" : "text-emerald-600"}`}>{dPeak > 0 ? "+" : ""}{Math.round(dPeak)} vs {cmpLabel}</div>}
          </div>
          <div className="rounded-xl bg-slate-50 p-3">
            <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Time above 180</div>
            <div className="text-lg font-bold text-slate-800">{s ? s.scenario.min_above_180 : "–"} <span className="text-xs font-medium text-slate-400">min</span></div>
            {active && cmp && <div className="text-xs text-slate-500">{cmp.min_above_180} min with {cmpLabel}</div>}
          </div>
          <div className="rounded-xl bg-slate-50 p-3">
            <div className="text-[11px] font-medium uppercase tracking-wide text-slate-400">Time below 70</div>
            <div className="text-lg font-bold text-slate-800">{s ? s.scenario.min_below_70 : "–"} <span className="text-xs font-medium text-slate-400">min</span></div>
            {active && cmp && <div className="text-xs text-slate-500">{cmp.min_below_70} min with {cmpLabel}</div>}
          </div>
        </div>
        <div className="h-[230px]">
          {res && (
            <ResponsiveContainer>
              <LineChart data={res.curve} margin={{ top: 8, right: 12, bottom: 0, left: -12 }}>
                <CartesianGrid stroke="#f1f5f9" vertical={false} />
                <ReferenceArea y1={70} y2={180} fill={COLORS.inRange} fillOpacity={0.07} />
                <ReferenceLine y={180} stroke={COLORS.high} strokeDasharray="4 4" />
                <ReferenceLine y={70} stroke={COLORS.low} strokeDasharray="4 4" />
                <ReferenceLine x={res.curve[24]?.t} stroke="#cbd5e1" label={{ value: "2 h", fontSize: 10, fill: "#94a3b8", position: "top" }} />
                <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={hhmm} scale="time"
                  ticks={res.curve.filter((_, i) => i % 6 === 0).map((c) => c.t)} />
                <YAxis domain={[40, yMax]} ticks={[70, 180, 250]} width={48} />
                <Tooltip labelFormatter={(v) => hhmm(Number(v))}
                  formatter={(v: number, n: string) => [`${Math.round(v)} mg/dL`, n === "base" ? "If nothing changes" : n === "reference" ? (state.refLabel ?? "Reference") : "With your changes"]}
                  contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                <Line dataKey="base" stroke="#94a3b8" strokeWidth={2} strokeDasharray="5 4" dot={false} isAnimationActive={false} />
                {s?.reference && <Line dataKey="reference" stroke="#d97706" strokeWidth={2} strokeDasharray="2 3" dot={false} isAnimationActive={false} />}
                <Line dataKey="scenario" stroke={COLORS.brand} strokeWidth={3} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
        <div className="mt-1 flex flex-wrap gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-5 border-t-2 border-dashed border-slate-400" />If nothing changes (ML forecast)</span>
          {s?.reference && <span className="flex items-center gap-1.5"><span className="h-0.5 w-5 border-t-2 border-dotted border-amber-600" />{state.refLabel}</span>}
          <span className="flex items-center gap-1.5"><span className="h-0.5 w-5 bg-teal-600" />With these changes (this patient's twin)</span>
        </div>
        <p className="mt-2 text-[11px] leading-snug text-slate-400">
          The first 2 h come from the ML forecast; the twin adds the effect of each change using this patient's own physiology.
          After 2 h the baseline is held flat. Walking effects are approximate.
        </p>
      </div>
    </div>
  );
}
