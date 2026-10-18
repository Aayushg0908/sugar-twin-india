import { useMemo } from "react";
import {
  Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceDot, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Snapshot } from "../api";
import { COLORS, hhmm } from "../ui";
import { useChartTheme } from "../theme";

type Pt = { t: number; g?: number | null; fc?: number; band?: [number, number]; actual?: number | null };

export default function GlucoseChart({ snap, reveal }: { snap: Snapshot; reveal: boolean }) {
  const th = useChartTheme();
  const now = snap.time;
  const data = useMemo(() => {
    const m = new Map<number, Pt>();
    const at = (t: number) => m.get(t) ?? (m.set(t, { t }), m.get(t)!);
    snap.history.forEach((p) => (at(p.t).g = p.g));
    snap.forecast.forEach((p) => {
      const d = at(p.t);
      d.fc = p.g;
      d.band = [p.lo, p.hi];
    });
    if (reveal) snap.actual_future.forEach((p) => (at(p.t).actual = p.g));
    return [...m.values()].sort((a, b) => a.t - b.t);
  }, [snap, reveal]);

  const start = snap.history[0]?.t ?? now - 12 * 3600e3;
  const end = snap.forecast[snap.forecast.length - 1]?.t ?? now + 3 * 3600e3;
  const ticks: number[] = [];
  for (let t = Math.ceil(start / 7200e3) * 7200e3; t <= end; t += 7200e3) ticks.push(t);
  const yMax = Math.max(300, ...data.map((d) => Math.max(d.g ?? 0, d.band?.[1] ?? 0, d.actual ?? 0)) ) + 10;

  const shown = snap.events.filter((e) => reveal || !e.future);
  const meals = shown.filter((e) => e.type === "meal");
  const meds = shown.filter((e) => e.type === "medication");
  const walks = shown.filter((e) => e.type === "exercise");

  return (
    <div className="h-[340px] w-full">
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 22, right: 18, bottom: 0, left: -12 }}>
          <CartesianGrid stroke={th.grid} vertical={false} />
          <ReferenceArea y1={70} y2={180} fill={COLORS.inRange} fillOpacity={0.07} />
          <ReferenceArea x1={now} x2={end} fill="#7c3aed" fillOpacity={0.035} />
          <ReferenceLine y={180} stroke={COLORS.high} strokeDasharray="4 4" strokeOpacity={0.7} />
          <ReferenceLine y={70} stroke={COLORS.low} strokeDasharray="4 4" strokeOpacity={0.7} />
          <ReferenceLine x={now} stroke={th.now} strokeWidth={1.5}
            label={{ value: "Now", position: "top", fill: th.now, fontSize: 11, fontWeight: 700 }} />
          <XAxis dataKey="t" type="number" domain={[start, end]} ticks={ticks} tickFormatter={hhmm} scale="time" />
          <YAxis domain={[40, yMax]} ticks={[54, 70, 180, 250]} width={48} />
          <Tooltip
            labelFormatter={(v) => hhmm(Number(v))}
            formatter={(v: number | number[], name: string) => {
              if (Array.isArray(v)) return [`${Math.round(v[0])}–${Math.round(v[1])} mg/dL`, "Likely range"];
              const label = name === "g" ? "Measured" : name === "fc" ? "Twin forecast" : "What really happened";
              return [`${Math.round(v)} mg/dL`, label];
            }}
            contentStyle={{ borderRadius: 12, fontSize: 12, ...th.tooltip }} labelStyle={{ color: th.muted }}
          />
          <Area dataKey="band" stroke="none" fill={COLORS.forecast} fillOpacity={0.12} isAnimationActive={false} connectNulls />
          <Line dataKey="g" stroke={th.ink} strokeWidth={2.2} dot={false} isAnimationActive={false} connectNulls={false} />
          <Line dataKey="fc" stroke={COLORS.forecast} strokeWidth={2.5} strokeDasharray="6 4" dot={false} isAnimationActive={false} connectNulls />
          {reveal && <Line dataKey="actual" stroke={th.actual} strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />}

          {meals.map((e) => (
            <ReferenceDot key={`m${e.t}`} x={e.t} y={48} r={9} fill={e.logged ? th.mealFill : th.unloggedFill} stroke={e.logged ? th.mealStroke : th.unloggedStroke}
              label={{ value: e.logged ? `${e.carbs ?? "?"}g` : "?", position: "top", fontSize: 10, fill: th.mealText }} />
          ))}
          {meds.map((e) => (
            <ReferenceDot key={`d${e.t}${e.label}`} x={e.t} y={yMax - 12} r={5} fill={e.taken ? "#16a34a" : "#dc2626"} stroke={th.surface} />
          ))}
          {walks.map((e) => (
            <ReferenceArea key={`w${e.t}`} x1={e.t} x2={e.t + (e.minutes ?? 20) * 60e3} y1={40} y2={60} fill="#0ea5e9" fillOpacity={0.35} />
          ))}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ChartLegend({ reveal }: { reveal: boolean }) {
  const item = (swatch: React.ReactNode, label: string) => (
    <span className="flex items-center gap-1.5">{swatch}{label}</span>
  );
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-500">
      {item(<span className="h-0.5 w-5 bg-slate-700" />, "Measured (CGM)")}
      {item(<span className="h-0.5 w-5 border-t-2 border-dashed border-violet-600" />, "Twin forecast")}
      {item(<span className="h-3 w-5 rounded bg-violet-200" />, "Likely range")}
      {reveal && item(<span className="h-0.5 w-5 bg-slate-400" />, "What really happened")}
      {item(<span className="h-3 w-5 rounded bg-green-100" />, "Target 70–180")}
      {item(<span className="h-3 w-3 rounded-full border border-amber-600 bg-amber-100" />, "Meal (g carbs, ? = not logged)")}
      {item(<span className="h-2.5 w-2.5 rounded-full bg-green-600" />, "Dose taken")}
      {item(<span className="h-2.5 w-2.5 rounded-full bg-red-600" />, "Dose missed")}
      {item(<span className="h-3 w-5 rounded bg-sky-300" />, "Exercise")}
    </div>
  );
}
