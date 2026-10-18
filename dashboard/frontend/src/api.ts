// Typed client for the FastAPI backend (dashboard/backend/app.py).

export type Level = "low" | "medium" | "high" | "now";
export type Trend = "rising_fast" | "rising" | "steady" | "falling" | "falling_fast" | "unknown";
export type RiskKey = "spike" | "severe" | "hypo";

export interface Accuracy {
  spike_auroc: number;
  spike_auroc_t2d: number;
  hypo_auroc: number;
  spikes_caught: number;
  lows_caught: number;
  mae60: number;
  mae120: number;
  twin_peak_err: number;
  population_peak_err: number;
  cgm_only_auroc: number;
  test_patients: number;
}

export interface Meta {
  start: number;
  end: number;
  default_time: number;
  grid_minutes: number;
  thresholds: Record<RiskKey, number>;
  accuracy: Accuracy;
  demo_patient: string;
}

export type Severity = "alert" | "warn" | "info" | "ok" | "tip";
export interface Finding {
  severity: Severity;
  text: string;
}

export interface PatientRow {
  id: string;
  name: string | null;
  age: number;
  sex: string;
  cohort: string;
  hba1c: number | null;
  glucose: number | null;
  trend: Trend;
  risks: Record<RiskKey, number>;
  levels: Record<RiskKey, Level>;
  urgency: "urgent" | "watch" | "ok";
  tir14: number;
  score: number;
}

export interface Summary {
  id: string;
  name: string | null;
  age: number;
  sex: string;
  cohort: string;
  region: string;
  setting: string;
  bmi: number | null;
  diet: string;
  staple: string;
  activity: string;
  smoking: string;
  diabetes_years: number | null;
  family_history: boolean;
  labs: { test: string; value: number | null; unit: string; date: string }[];
  hba1c_history: { date: number; value: number | null }[];
  medications: { drug: string; dose: string; class: string; for: string; since: string }[];
  diagnoses: { name: string; code: string; since: string }[];
  cgm14: {
    mean: number;
    gmi: number;
    cv: number;
    coverage: number;
    bands: { very_low: number; low: number; in_range: number; high: number; very_high: number };
  };
  agp: { hour: number; p5: number; p25: number; p50: number; p75: number; p95: number }[];
  daily: { date: number; tir: number | null; mean: number | null }[];
  twin: {
    p10: number;
    tau: number;
    w: number;
    pop_p10: number;
    pop_tau: number;
    description: string;
    carb_label: "high" | "low" | "average";
    recovery_label: "slow" | "fast" | "average";
    walk_label: "strong" | "weak" | "average";
    rise_per_roti: number;
    pop_rise_per_roti: number;
    recovery_hours: number;
    pop_recovery_hours: number;
  };
  flags: Finding[];
}

export interface Reason {
  text: string;
  weight: number;
  group: string;
}

export interface Risk {
  p: number;
  threshold: number;
  level: Level;
  reasons: Reason[];
}

export type Change = (string | number)[];

export interface Suggestion {
  action: string;
  effect: number | null;
  kind: "lower" | "raise" | "info";
  changes: Change[];
  reference?: Change[];
  reference_label?: string;
}

export interface TwinEvent {
  t: number;
  type: "meal" | "medication" | "exercise";
  label: string;
  logged: boolean;
  future: boolean;
  carbs: number | null;
  taken: boolean | null;
  minutes: number | null;
}

export interface Snapshot {
  time: number;
  glucose: number | null;
  delta15: number | null;
  trend: Trend;
  headline: Finding[];
  history: { t: number; g: number | null }[];
  actual_future: { t: number; g: number | null }[];
  forecast: { t: number; g: number; lo: number; hi: number }[];
  risks: Record<RiskKey, Risk>;
  events: TwinEvent[];
  suggestions: Suggestion[];
}

export interface WhatIfRequest {
  t: string;
  changes: Change[];
  reference?: Change[] | null;
}

export interface ScenarioStats {
  peak: number;
  peak_min: number;
  low: number;
  min_above_180: number;
  min_below_70: number;
}

export interface WhatIfResult {
  time: number;
  curve: { t: number; base: number; scenario: number; reference?: number }[];
  summary: { baseline: ScenarioStats; scenario: ScenarioStats; reference?: ScenarioStats };
}

// Timestamps from the backend are naive local times sent as epoch-ms; treat them as UTC everywhere.
export const isoNaive = (ms: number) => new Date(ms).toISOString().slice(0, 16);

async function get<T>(url: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(url, { signal });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json();
}

export const api = {
  meta: () => get<Meta>("/api/meta"),
  patients: (t: number, signal?: AbortSignal) =>
    get<{ time: number; patients: PatientRow[] }>(`/api/patients?t=${isoNaive(t)}`, signal),
  summary: (id: string, signal?: AbortSignal) => get<Summary>(`/api/patients/${id}/summary`, signal),
  snapshot: (id: string, t: number, signal?: AbortSignal) =>
    get<Snapshot>(`/api/patients/${id}/snapshot?t=${isoNaive(t)}`, signal),
  whatif: async (id: string, body: WhatIfRequest, signal?: AbortSignal): Promise<WhatIfResult> => {
    const r = await fetch(`/api/patients/${id}/whatif`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!r.ok) throw new Error(`${r.status} whatif`);
    return r.json();
  },
};
