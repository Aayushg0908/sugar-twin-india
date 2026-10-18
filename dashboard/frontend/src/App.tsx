import { useCallback, useEffect, useState } from "react";
import { BarChart3, Brain, CalendarDays, Eye, EyeOff, FlaskConical, LineChart as LineIcon, ShieldCheck, Sparkles, Stethoscope, Wand2 } from "lucide-react";
import { api, type Meta, type PatientRow, type Snapshot, type Suggestion, type Summary } from "./api";
import TopBar from "./components/TopBar";
import PatientList from "./components/PatientList";
import GlucoseChart, { ChartLegend } from "./components/GlucoseChart";
import { CurrentGlucose, RiskCards, Suggestions } from "./components/RiskPanel";
import WhatIf, { EMPTY_WHATIF, type WhatIfState } from "./components/WhatIf";
import { ProfileCard, TwinCard } from "./components/Profile";
import { AccuracyStrip, CgmStats, DailyPattern, DailyTir, EhrPanel, TimeInRange } from "./components/Overview";
import { Card, hhmm } from "./ui";

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [time, setTime] = useState<number>(0);
  const [playing, setPlaying] = useState(false);
  const [patients, setPatients] = useState<PatientRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [reveal, setReveal] = useState(false);
  const [whatif, setWhatif] = useState<WhatIfState>(EMPTY_WHATIF);
  const [error, setError] = useState<string | null>(null);

  // the API loads models and data at startup (~20 s): keep retrying until it answers
  useEffect(() => {
    let tries = 0;
    let timer: ReturnType<typeof setTimeout>;
    const load = () => api.meta()
      .then((m) => { setMeta(m); setTime(m.default_time); setSelected(m.demo_patient); setError(null); })
      .catch(() => {
        tries += 1;
        if (tries > 45) setError("Cannot reach the API. Start the backend: uvicorn app:app --port 8000 (in dashboard/backend).");
        else timer = setTimeout(load, 2000);
      });
    load();
    return () => clearTimeout(timer);
  }, []);

  // patient list follows the simulated clock
  useEffect(() => {
    if (!time) return;
    const ctl = new AbortController();
    const id = setTimeout(() => api.patients(time, ctl.signal).then((r) => setPatients(r.patients)).catch(() => {}), 150);
    return () => { clearTimeout(id); ctl.abort(); };
  }, [time]);

  useEffect(() => {
    if (!selected) return;
    const ctl = new AbortController();
    setSummary(null);
    setWhatif(EMPTY_WHATIF);
    api.summary(selected, ctl.signal).then(setSummary).catch(() => {});
    return () => ctl.abort();
  }, [selected]);

  useEffect(() => {
    if (!selected || !time) return;
    const ctl = new AbortController();
    api.snapshot(selected, time, ctl.signal).then(setSnap).catch(() => {});
    return () => ctl.abort();
  }, [selected, time]);

  // replay: advance 30 simulated minutes per second
  useEffect(() => {
    if (!playing || !meta) return;
    const id = setInterval(() => setTime((t) => {
      const next = t + 30 * 60e3;
      if (next > meta.end - 2.5 * 3600e3) { setPlaying(false); return t; }
      return next;
    }), 1000);
    return () => clearInterval(id);
  }, [playing, meta]);

  const tryIt = useCallback((s: Suggestion) => {
    const next: WhatIfState = { ...EMPTY_WHATIF, ref: s.reference ?? null, refLabel: s.reference_label ?? null };
    for (const c of s.changes) {
      if (c[0] === "meal") Object.assign(next, { meal_in: Number(c[1]), meal_carbs: Number(c[2]), meal_gi: c[3] as WhatIfState["meal_gi"] });
      if (c[0] === "walk") Object.assign(next, { walk_in: Number(c[1]), walk_minutes: Number(c[2]) });
      if (c[0] === "skip_drug") next.skip = [...next.skip, String(c[1])];
    }
    setWhatif(next);
    document.getElementById("whatif")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  if (error) return <div className="flex h-full items-center justify-center p-8 text-center text-slate-600">{error}</div>;
  if (!meta) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-slate-500">
        <img src="/favicon.svg" alt="" className="h-12 w-12 animate-pulse" />
        <div className="font-semibold">Starting Sugar Twin…</div>
        <div className="text-xs text-slate-400">Loading models and 751 patients</div>
      </div>
    );
  }

  const drugClasses = [...new Set((summary?.medications ?? []).filter((m) => m.for === "Type 2 diabetes").map((m) => m.class))];

  return (
    <div className="flex h-full flex-col">
      <TopBar time={time} start={meta.start} end={meta.end} playing={playing} onTime={(t) => { setPlaying(false); setTime(t); }} onPlay={setPlaying} />
      <div className="flex min-h-0 flex-1">
        <PatientList patients={patients} selected={selected} onSelect={setSelected} />

        <main className="scroll-thin min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1400px] space-y-5 p-5">
            {!summary || !snap ? (
              <div className="py-24 text-center text-slate-400">Loading patient…</div>
            ) : (
              <>
                <Card><ProfileCard s={summary} /></Card>

                <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
                  <Card title="Glucose: last 12 hours and next 2 hours" icon={<LineIcon className="h-4 w-4" />}
                    hint={`Simulated now: ${hhmm(snap.time)}. The violet dashed line is the twin's forecast.`}
                    right={
                      <button onClick={() => setReveal(!reveal)}
                        className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold ring-1 transition ${reveal ? "bg-slate-800 text-white ring-slate-800" : "bg-white text-slate-600 ring-slate-200 hover:bg-slate-50"}`}>
                        {reveal ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                        {reveal ? "Hide what happened" : "Reveal what really happened"}
                      </button>
                    }>
                    <GlucoseChart snap={snap} reveal={reveal} />
                    <div className="mt-3"><ChartLegend reveal={reveal} /></div>
                  </Card>

                  <div className="space-y-5">
                    <Card title="Right now" icon={<Stethoscope className="h-4 w-4" />}>
                      <CurrentGlucose snap={snap} />
                      <div className="mt-3"><RiskCards snap={snap} /></div>
                    </Card>
                  </div>
                </div>

                <div className="grid gap-5 xl:grid-cols-[1fr_1fr]">
                  <Card title="What the twin suggests" icon={<Sparkles className="h-4 w-4" />}
                    hint="Each effect is simulated on this patient's own physiology. Click “Try it” to see it below.">
                    <Suggestions items={snap.suggestions} onTry={tryIt} />
                  </Card>
                  <Card title="This patient's twin" icon={<Brain className="h-4 w-4" />}
                    hint="Learned from the first week of logged meals and CGM, then checked on new meals.">
                    <TwinCard s={summary} />
                    <p className="mt-3 text-sm text-slate-600">{summary.twin.description}.</p>
                  </Card>
                </div>

                <div id="whatif">
                  <Card title="What if…? Try a meal, a walk or a missed dose" icon={<Wand2 className="h-4 w-4" />}
                    hint="Starts from the current moment. Change the controls to see how this patient's glucose would respond.">
                    <WhatIf pid={summary.id} time={snap.time} drugClasses={drugClasses} state={whatif} setState={setWhatif} />
                  </Card>
                </div>

                <div className="grid gap-5 xl:grid-cols-[360px_1fr]">
                  <Card title="14-day time in range" icon={<BarChart3 className="h-4 w-4" />}
                    hint="Targets for diabetes: >70% in range, <4% low, <5% very high (hover a row)">
                    <TimeInRange s={summary} />
                    <div className="mt-4"><CgmStats s={summary} /></div>
                  </Card>
                  <Card title="Typical day (14 days overlaid)" icon={<CalendarDays className="h-4 w-4" />}
                    hint="Line = typical glucose at each hour; shaded = how much it varies from day to day">
                    <DailyPattern s={summary} />
                    <div className="mt-3">
                      <div className="mb-1 text-xs font-semibold text-slate-500">Time in range, each day (dashed = 70% target)</div>
                      <DailyTir s={summary} />
                    </div>
                  </Card>
                </div>

                <Card title="Health record" icon={<FlaskConical className="h-4 w-4" />}>
                  <EhrPanel s={summary} />
                </Card>

                <Card title="How accurate is the twin?" icon={<ShieldCheck className="h-4 w-4" />} className="bg-gradient-to-br from-brand-50 to-white"
                  hint={`Measured on ${meta.accuracy.test_patients} test patients whose data was never used for training`}>
                  <AccuracyStrip a={meta.accuracy} />
                </Card>
                <p className="pb-4 text-center text-[11px] text-slate-400">
                  All patient data is synthetic. Decision-support prototype, not a medical device.
                </p>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
