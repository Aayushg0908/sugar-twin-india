import { useCallback, useEffect, useState } from "react";
import { Brain, ClipboardList, Dna, Eye, EyeOff, FlaskConical, LineChart as LineIcon, ShieldCheck, Sparkles, Stethoscope, Wand2 } from "lucide-react";
import { api, type Meta, type PatientRow, type Snapshot, type Suggestion, type Summary } from "./api";
import TopBar from "./components/TopBar";
import PatientList from "./components/PatientList";
import GlucoseChart, { ChartLegend } from "./components/GlucoseChart";
import { CurrentGlucose, RiskCards, Suggestions } from "./components/RiskPanel";
import WhatIf, { EMPTY_WHATIF, type WhatIfState } from "./components/WhatIf";
import { ProfileCard, TwinCard, twinSentence } from "./components/Profile";
import ClinicalSummary from "./components/ClinicalSummary";
import { AccuracyStrip, EhrPanel } from "./components/Overview";
import GeneticsPanel from "./components/Genetics";
import { Card, hhmm } from "./ui";

const LIST_KEY = "sugar-twin-list-open";

/** Patient list starts open on wide screens; the doctor's choice is remembered. */
function initialListOpen() {
  try {
    const saved = localStorage.getItem(LIST_KEY);
    if (saved === "true" || saved === "false") return saved === "true";
  } catch {
    /* storage blocked: fall back to screen width */
  }
  return window.innerWidth >= 1024;
}

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
  const [whatifOpen, setWhatifOpen] = useState(false);
  const [listOpen, setListOpen] = useState(initialListOpen);
  const [error, setError] = useState<string | null>(null);

  const toggleList = () => setListOpen((o) => {
    try { localStorage.setItem(LIST_KEY, String(!o)); } catch { /* ignore */ }
    return !o;
  });

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
    setWhatifOpen(true);   // the what-if dropdown opens so the doctor sees the result
    setTimeout(() => document.getElementById("whatif")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
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
      <TopBar time={time} start={meta.start} end={meta.end} playing={playing} onTime={(t) => { setPlaying(false); setTime(t); }} onPlay={setPlaying}
        listOpen={listOpen} onToggleList={toggleList} />
      <div className="flex min-h-0 flex-1">
        {listOpen && <PatientList patients={patients} selected={selected} onSelect={setSelected} />}

        <main className="scroll-thin min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-[1400px] space-y-5 p-5">
            {!summary || !snap ? (
              <div className="py-24 text-center text-slate-400">Loading patient…</div>
            ) : (
              <>
                {/* ---- always visible: what the doctor needs for this patient right now */}
                <Card><ProfileCard s={summary} /></Card>

                <Card title="Clinical summary" icon={<ClipboardList className="h-4 w-4" />}
                  hint="Read this first. Details for each point are in the sections below.">
                  <ClinicalSummary now={snap.headline} flags={summary.flags} />
                </Card>

                <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
                  <Card title="Glucose: last 12 hours and next 2 hours" icon={<LineIcon className="h-4 w-4" />}
                    hint={`Simulated now: ${hhmm(snap.time)}. The blue dashed line is the twin's forecast.`}
                    right={
                      <button onClick={() => setReveal(!reveal)}
                        className={`flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-semibold ring-1 transition ${reveal ? "bg-blue-600 text-[#fff] ring-blue-600" : "bg-white text-slate-600 ring-slate-200 hover:bg-slate-50"}`}>
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

                <Card title="What the twin suggests" icon={<Sparkles className="h-4 w-4" />}
                  hint="Each effect is simulated on this patient's own physiology. Click “Try it” to see it below.">
                  <Suggestions items={snap.suggestions} onTry={tryIt} />
                </Card>

                {/* ---- dropdowns: opened by the doctor when needed */}
                <Card id="whatif" collapsible open={whatifOpen} onToggle={() => setWhatifOpen((o) => !o)}
                  title="What if…? Try a meal, a walk or a missed dose" icon={<Wand2 className="h-4 w-4" />}
                  hint="Simulated on this patient's twin, starting from the current moment.">
                  {/* the twin's learned physiology explains why the simulated curves look the way they do */}
                  <div className="mb-5 rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <h3 className="mb-2 flex items-center gap-2 text-xs font-semibold text-slate-400">
                      <Brain className="h-3.5 w-3.5" />This patient's twin (learned from week 1 of logged meals and CGM)
                    </h3>
                    <TwinCard s={summary} />
                    <p className="mt-3 text-sm text-slate-600">{twinSentence(summary)}</p>
                  </div>
                  <WhatIf pid={summary.id} time={snap.time} drugClasses={drugClasses} state={whatif} setState={setWhatif} />
                </Card>

                <Card collapsible title="Health record: labs, medicines, diagnoses" icon={<FlaskConical className="h-4 w-4" />}>
                  <EhrPanel s={summary} />
                </Card>

                <Card collapsible title="Genetic markers" icon={<Dna className="h-4 w-4" />}
                  hint="Type 2 diabetes risk variants from the patient's genetic test">
                  <GeneticsPanel g={summary.genetics} />
                </Card>

                <Card collapsible title="How accurate is the twin?" icon={<ShieldCheck className="h-4 w-4" />}
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
