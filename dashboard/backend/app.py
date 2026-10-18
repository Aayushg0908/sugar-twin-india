"""
Sugar Twin dashboard API (FastAPI).

Serves held-out TEST patients (never used for training) plus the demo patient NEW001 ("Sunita"):
  GET  /api/meta                         time range, thresholds, model accuracy
  GET  /api/patients?t=...               triage list at simulated time t, sorted by risk
  GET  /api/patients/{pid}/summary       profile, EHR, twin knobs, 14-day CGM overview
  GET  /api/patients/{pid}/snapshot?t=   glucose history, forecast, risks, reasons, suggestions at time t
  POST /api/patients/{pid}/whatif        twin what-if on top of the ML forecast

Run:  uvicorn app:app --port 8000   (from dashboard/backend)
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from twin_simulator import PatientTwin  # noqa: E402

DATA, EHR, W, RES, NEW = ROOT / "data", ROOT / "data" / "ehr", ROOT / "data" / "wearable", ROOT / "results", ROOT / "data" / "new_patient"
START = pd.Timestamp("2026-09-17")
STEP, BPD, T = 5, 288, 14 * 288
GRID = 30                                    # predictions every 30 min
HORIZONS = [30, 60, 90, 120]
DEMO_ID = "NEW001"
GI_NAMES = {"high": "high", "medium": "medium", "low": "low"}


def ms(ts) -> int:
    return int(pd.Timestamp(ts).value // 1_000_000)


def clean(v):
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 2)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# --------------------------------------------------------------------------
# Data store (loaded once at startup)
# --------------------------------------------------------------------------
class Store:
    def __init__(self):
        split = pd.read_csv(DATA / "splits.csv")
        self.ids = sorted(split[split.split == "test"].patient_id)
        ids = set(self.ids)

        pats = pd.read_csv(EHR / "patients.csv")
        feats = pd.read_csv(EHR / "patient_features.csv")
        meds = pd.read_csv(EHR / "medications.csv")
        dx = pd.read_csv(EHR / "diagnoses.csv")
        labs = pd.read_csv(EHR / "labs.csv")
        new = {n: pd.read_csv(NEW / f"{n}.csv") for n in ("patients", "medications", "diagnoses", "labs", "events", "predictions")}
        new_feats = pd.read_parquet(NEW / "features.parquet")
        self.patients = pd.concat([pats[pats.patient_id.isin(ids)], new["patients"]]).set_index("patient_id")
        self.patients["name"] = None
        self.patients.loc[DEMO_ID, "name"] = "Sunita (demo)"
        self.ehr_feats = feats.set_index("patient_id")
        self.meds = pd.concat([meds[meds.patient_id.isin(ids)], new["medications"]])
        self.dx = pd.concat([dx[dx.patient_id.isin(ids)], new["diagnoses"]])
        keep_tests = ["HbA1c", "Fasting plasma glucose", "LDL cholesterol", "eGFR (CKD-EPI 2021)", "Systolic blood pressure",
                      "Diastolic blood pressure", "Body weight", "Triglycerides"]
        labs = pd.concat([labs[labs.patient_id.isin(ids)], new["labs"]])
        self.labs = labs[labs.test.isin(keep_tests)]
        demo_last = new["labs"].sort_values("date").groupby("test").value.last()
        self.demo_hba1c = float(demo_last.get("HbA1c", np.nan))

        # 5-min CGM
        cgm = pd.read_parquet(W / "cgm.parquet", filters=[("patient_id", "in", self.ids)])
        self.cgm = {pid: g.glucose_mgdl.to_numpy(dtype=float, na_value=np.nan) for pid, g in cgm.groupby("patient_id", sort=False)}
        stream = pd.read_csv(NEW / "stream_5min.csv")
        self.cgm[DEMO_ID] = stream.glucose_mgdl.to_numpy(dtype=float)
        self.ids = self.ids + [DEMO_ID]

        ev = pd.read_csv(W / "events.csv", parse_dates=["timestamp"])
        ev = pd.concat([ev[ev.patient_id.isin(ids)], new["events"].assign(timestamp=pd.to_datetime(new["events"].timestamp))])
        ev["logged"] = ev.logged.astype(str).str.lower().eq("true")
        self.events = {pid: g.sort_values("timestamp") for pid, g in ev.groupby("patient_id")}

        # ML predictions: ensemble of LightGBM + GRU (30-min grid)
        keys = ["patient_id", "timestamp"]
        lg = pd.read_parquet(RES / "lightgbm" / "test_predictions.parquet")
        gr = pd.read_parquet(RES / "gru" / "test_predictions.parquet")
        cols = ["p_spike", "p_severe", "p_hypo"] + [f"pred_fc{h}" for h in HORIZONS]
        p = lg[keys + ["g_now"] + cols].merge(gr[keys + cols], on=keys, suffixes=("_l", "_g"))
        out = p[keys + ["g_now"]].copy()
        for c in cols:
            out[c] = (p[f"{c}_l"] + p[f"{c}_g"]) / 2
        demo = new["predictions"].assign(timestamp=lambda d: pd.to_datetime(d.timestamp)).rename(
            columns={"spike": "p_spike", "severe": "p_severe", "hypo": "p_hypo", **{f"fc{h}": f"pred_fc{h}" for h in HORIZONS}})
        out = pd.concat([out, demo[keys + ["g_now"] + cols]])
        self.preds = {pid: g.set_index("timestamp").sort_index() for pid, g in out.groupby("patient_id")}

        # LightGBM features (for the "why?" explanations)
        self.boosters = {k: lgb.Booster(model_file=str(ROOT / "models" / "lightgbm" / f"{k}.txt")) for k in ("spike", "hypo", "severe")}
        self.feat_cols = self.boosters["spike"].feature_name()
        f = pd.read_parquet(DATA / "model" / "features.parquet", columns=["patient_id", "timestamp"] + self.feat_cols,
                            filters=[("split", "==", "test")])
        f = pd.concat([f, new_feats[["patient_id", "timestamp"] + self.feat_cols]])
        self.features = {pid: g.set_index("timestamp") for pid, g in f.groupby("patient_id")}

        # thresholds (average of the two models' validation thresholds) and accuracy for the info strip
        ml = {m: json.loads((RES / m / "metrics.json").read_text()) for m in ("lightgbm", "gru")}
        self.thr = {k: (ml["lightgbm"][k]["threshold"] + ml["gru"][k]["threshold"]) / 2 for k in ("spike", "severe", "hypo")}
        comp = json.loads((RES / "comparison.json").read_text())
        self.mae = {h: comp[f"fc{h}"]["Ensemble"]["mae"] for h in HORIZONS}
        twin_m = json.loads((RES / "twin_simulator" / "metrics.json").read_text())
        rob = json.loads((RES / "robustness" / "results.json").read_text())
        self.accuracy = {
            "spike_auroc": comp["spike"]["Ensemble"]["auroc"], "spike_auroc_t2d": comp["spike"]["Ensemble"]["auroc_t2d"],
            "hypo_auroc": comp["hypo"]["Ensemble"]["auroc"],
            "spikes_caught": ml["gru"]["spike"]["event_detection"]["detection_rate"],
            "lows_caught": ml["gru"]["hypo"]["event_detection"]["detection_rate"],
            "mae60": self.mae[60], "mae120": self.mae[120],
            "twin_peak_err": twin_m["week2_test_comparison"]["personal_shrunk"]["peak_rise_mae"],
            "population_peak_err": twin_m["week2_test_comparison"]["population"]["peak_rise_mae"],
            "cgm_only_auroc": rob["no_smartwatch"]["Ensemble"]["spike_auroc"],
            "test_patients": len(self.ids) - 1}

        # twins
        tw = pd.read_csv(ROOT / "models" / "twin" / "twin_params_fitted.csv").set_index("patient_id")
        demo_tw = json.loads((NEW / "twin_fitted.json").read_text())
        self.twin_params = {pid: (r.tau, r.p10, r.w) for pid, r in tw.loc[tw.index.intersection(self.ids)].iterrows()}
        self.twin_params[DEMO_ID] = (demo_tw["tau"], demo_tw["p10"], demo_tw["w"])
        self.twin_pop = (float(tw.tau.median()), float(tw.p10.median()), float(tw.w.median()))

    # ---------------- helpers
    def active_meds(self, pid):
        m = self.meds[(self.meds.patient_id == pid) & self.meds.end_date.isna()]
        return m

    def twin(self, pid) -> PatientTwin:
        tau, p10, w = self.twin_params[pid]
        act = self.active_meds(pid)
        act = act[act.indication == "Type 2 diabetes"]
        return PatientTwin(pid, tau, p10, w, list(zip(act.drug, act.drug_class)))

    def snap_time(self, t) -> pd.Timestamp:
        t = pd.Timestamp(t) if t else pd.Timestamp("2026-09-26 19:30")
        t = t.floor(f"{GRID}min")
        return min(max(t, START + pd.Timedelta(hours=7)), START + pd.Timedelta(days=14) - pd.Timedelta(hours=2, minutes=30))

    def pred_row(self, pid, t):
        p = self.preds[pid]
        if t in p.index:
            return p.loc[t]
        i = p.index.searchsorted(t, side="right") - 1
        return p.iloc[max(i, 0)]


def jsonsafe(o):
    """Recursively turn NaN / inf / numpy types into JSON-friendly values."""
    if isinstance(o, dict):
        return {k: jsonsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonsafe(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if o is pd.NA or o is pd.NaT:
        return None
    return o


class SafeJSON(JSONResponse):
    def render(self, content) -> bytes:
        return super().render(jsonsafe(content))


store: Store | None = None


@asynccontextmanager
async def lifespan(_app):
    global store
    store = Store()          # loads data and models once (~20 s)
    yield


app = FastAPI(title="Sugar Twin API", default_response_class=SafeJSON, lifespan=lifespan)


def need(pid):
    if pid not in store.cgm:
        raise HTTPException(404, f"unknown patient {pid}")


# --------------------------------------------------------------------------
# Plain-language reasons from SHAP contributions
# --------------------------------------------------------------------------
def reason_text(f, v, row):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return None, None
    fam = f
    if f in ("g_delta_15", "g_delta_30", "g_delta_60", "g_delta_120", "g_accel_15"):
        mins = {"g_delta_15": 15, "g_delta_30": 30, "g_delta_60": 60, "g_delta_120": 120}.get(f)
        if f == "g_accel_15":
            return "trend", ("Rise is speeding up" if v > 0 else "Rise is slowing down")
        return "trend", f"Glucose {'up' if v > 0 else 'down'} {abs(v):.0f} mg/dL in last {mins} min"
    if f == "g_now":
        return "level", f"Glucose now {v:.0f} mg/dL"
    if f.startswith(("g_mean", "g_max", "g_min", "g_std")):
        span = f.split("_")[-1]
        kind = {"mean": "Average", "max": "Highest", "min": "Lowest", "std": "Swings"}[f.split("_")[1]]
        return "level", (f"{kind} glucose in last {span}: {v:.0f} mg/dL" if kind != "Swings" else f"Glucose swinging a lot in last {span}")
    if f == "g_tar_6h":
        return "level", f"{v * 100:.0f}% of last 6 h above 180"
    if f == "g_tir_6h":
        return "level", f"Only {v * 100:.0f}% of last 6 h in range" if v < 0.7 else f"{v * 100:.0f}% of last 6 h in range"
    if f in ("min_since_meal",):
        return "meal", "No meal logged in the last 12 h" if v >= 700 else f"Meal logged {v:.0f} min ago"
    if f in ("last_meal_carbs", "carbs_1h", "carbs_3h", "carbs_today"):
        span = {"last_meal_carbs": "last meal", "carbs_1h": "last 1 h", "carbs_3h": "last 3 h", "carbs_today": "today"}[f]
        return "meal", f"{v:.0f} g carbs logged ({span})"
    if f == "last_meal_gi":
        return "meal", {2: "Last meal was high-GI (rice / idli / sweets)", 1: "Last meal was medium-GI", 0: "Last meal was low-GI"}.get(int(round(v)))
    if f in ("doses_logged_today", "min_since_dose_logged", "prescribed_doses_per_day"):
        pres = row.get("prescribed_doses_per_day", np.nan)
        if f == "min_since_dose_logged":
            return "meds", "No diabetes dose logged in 24 h" if v >= 1400 else f"Last dose logged {v / 60:.1f} h ago"
        if np.isfinite(pres) and pres > 0:
            return "meds", f"{row.get('doses_logged_today', 0):.0f} of {pres:.0f} daily doses logged so far"
        return "meds", "No glucose-lowering medicine"
    if f.startswith("prof_"):
        if f == "prof_spike_rate_tod":
            return "habit", f"Went above 180 around this time on {v * 100:.0f}% of past days"
        if f == "prof_glucose_tod":
            return "habit", f"Usual glucose at this time: {v:.0f} mg/dL"
        return "habit", f"Usual 2-h peak from this time: {v:.0f} mg/dL"
    if f == "g_vs_profile":
        return "habit", f"{abs(v):.0f} mg/dL {'above' if v > 0 else 'below'} usual for this time"
    if f in ("hour", "hour_sin", "hour_cos"):
        return "time", f"Time of day ({int(row.get('hour', 0)):02d}:{int(round(row.get('hour', 0) % 1 * 60)):02d})"
    if f == "weekend":
        return "time", "Weekend" if v else "Weekday"
    if f in ("sleep_hours_last_night", "sleep_eff_last_night"):
        return "sleep", f"Slept {row.get('sleep_hours_last_night', v):.1f} h last night"
    if f in ("hrv_last_night", "resting_hr_last_night"):
        return "sleep", f"Overnight HRV {row.get('hrv_last_night', np.nan):.0f} ms / resting HR {row.get('resting_hr_last_night', np.nan):.0f}"
    if f.startswith("steps") or f in ("min_since_exercise", "exercise_min_3h"):
        if f == "min_since_exercise":
            return "activity", "No exercise in the last 24 h" if v >= 1400 else f"Exercised {v / 60:.1f} h ago"
        if f == "exercise_min_3h":
            return "activity", f"{v:.0f} min exercise in last 3 h"
        span = f.split("_")[-1]
        return "activity", f"{v:.0f} steps ({span.replace('yesterday', 'yesterday')})"
    if f.startswith("hr_"):
        return "activity", f"Heart rate {row.get('hr_15m', v):.0f} bpm"
    if f in ("last_hba1c", "hba1c_mean", "hba1c_change"):
        return "ehr", f"HbA1c {row.get('last_hba1c', v):.1f}%"
    if f == "on_sulfonylurea" and v:
        return "ehr", "On sulfonylurea (can cause lows)"
    if f == "on_basal_insulin" and v:
        return "ehr", "On basal insulin"
    if f == "diabetes_duration_yrs":
        return "ehr", f"Diabetes for {v:.0f} years"
    if f in ("bmi", "waist_cm"):
        return "ehr", f"BMI {row.get('bmi', v):.1f}"
    if f == "cohort":
        return "ehr", {0: "No diabetes", 1: "Prediabetes", 2: "Type 2 diabetes"}.get(int(v))
    return fam, None


def explain(pid, t, kind, top=4):
    feats = store.features[pid]
    if t not in feats.index:
        return []
    row = feats.loc[t]
    contrib = store.boosters[kind].predict(row[store.feat_cols].to_frame().T.astype(float), pred_contrib=True)[0][:-1]
    order = np.argsort(-contrib)
    out, seen = [], set()
    rowd = row.to_dict()
    for i in order:
        if contrib[i] <= 0 or len(out) >= top:
            break
        fam, text = reason_text(store.feat_cols[i], rowd[store.feat_cols[i]], rowd)
        if text and fam not in seen:
            seen.add(fam)
            out.append({"text": text, "weight": round(float(contrib[i]), 3), "group": fam})
    return out


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
def risk_level(p, thr):
    return "high" if p >= thr else "medium" if p >= thr * 0.6 else "low"


def current_glucose(pid, t):
    g = store.cgm[pid]
    i = int((t - START) / pd.Timedelta(minutes=STEP))
    seg = g[max(0, i - 3): i + 1]
    now = next((x for x in seg[::-1] if np.isfinite(x)), np.nan)
    prev = g[i - 3] if i >= 3 else np.nan
    return now, (now - prev) if np.isfinite(now) and np.isfinite(prev) else np.nan


def trend_arrow(d15):
    if not np.isfinite(d15):
        return "unknown"
    rate = d15 / 15
    return "rising_fast" if rate > 2 else "rising" if rate > 1 else "falling_fast" if rate < -2 else "falling" if rate < -1 else "steady"


@app.get("/api/meta")
def meta():
    return {"start": ms(START), "end": ms(START + pd.Timedelta(days=14)), "default_time": ms(store.snap_time(None)),
            "grid_minutes": GRID, "thresholds": store.thr, "accuracy": store.accuracy, "demo_patient": DEMO_ID}


@app.get("/api/patients")
def patients(t: str | None = None):
    t = store.snap_time(t)
    out = []
    for pid in store.ids:
        pr = store.pred_row(pid, t)
        info = store.patients.loc[pid]
        g, d15 = current_glucose(pid, t)
        g14 = store.cgm[pid]
        valid = g14[np.isfinite(g14)]
        hba1c = store.demo_hba1c if pid == DEMO_ID else store.ehr_feats.loc[pid, "last_hba1c"]
        risks = {k: float(pr[f"p_{k}"]) for k in ("spike", "severe", "hypo")}
        score = max(risks["spike"] / store.thr["spike"], risks["hypo"] / store.thr["hypo"])
        out.append({"id": pid, "name": info["name"], "age": int(info.age), "sex": info.sex, "cohort": info.cohort,
                    "hba1c": clean(hba1c), "glucose": clean(g), "trend": trend_arrow(d15),
                    "risks": {k: round(v, 3) for k, v in risks.items()},
                    "levels": {k: risk_level(v, store.thr[k]) for k, v in risks.items()},
                    "tir14": round(float(((valid >= 70) & (valid <= 180)).mean() * 100), 1), "score": round(score, 3)})
    out.sort(key=lambda r: (r["id"] != DEMO_ID, -r["score"]))
    return {"time": ms(t), "patients": out}


@app.get("/api/patients/{pid}/summary")
def summary(pid: str):
    need(pid)
    info = store.patients.loc[pid]
    labs = store.labs[store.labs.patient_id == pid]
    hba1c = labs[labs.test == "HbA1c"][["date", "value"]]
    latest = labs.sort_values("date").groupby("test").agg(value=("value", "last"), unit=("unit", "last"), date=("date", "last"))
    act = store.active_meds(pid)
    dx = store.dx[store.dx.patient_id == pid].sort_values("onset_date")
    g = store.cgm[pid]
    valid = g[np.isfinite(g)]
    bands = {"very_low": (valid < 54).mean(), "low": ((valid >= 54) & (valid < 70)).mean(),
             "in_range": ((valid >= 70) & (valid <= 180)).mean(), "high": ((valid > 180) & (valid <= 250)).mean(),
             "very_high": (valid > 250).mean()}
    # AGP: percentiles of glucose by time of day (hourly)
    day = g.reshape(14, BPD)
    agp = []
    for h in range(24):
        v = day[:, h * 12:(h + 1) * 12].ravel()
        v = v[np.isfinite(v)]
        if len(v):
            q = np.percentile(v, [5, 25, 50, 75, 95])
            agp.append({"hour": h, "p5": q[0], "p25": q[1], "p50": q[2], "p75": q[3], "p95": q[4]})
    daily = []
    for d in range(14):
        v = day[d][np.isfinite(day[d])]
        daily.append({"date": ms(START + pd.Timedelta(days=d)), "tir": float(((v >= 70) & (v <= 180)).mean() * 100) if len(v) else None,
                      "mean": float(v.mean()) if len(v) else None})
    tw = store.twin(pid)
    tau, p10, w = store.twin_params[pid]
    pop = store.twin_pop
    return {
        "id": pid, "name": info["name"], "age": int(info.age), "sex": info.sex, "cohort": info.cohort, "region": info.region,
        "setting": info.setting, "bmi": clean(info.bmi), "diet": info.diet, "staple": info.staple, "activity": info.activity_level,
        "smoking": info.smoking, "diabetes_years": clean(info.diabetes_duration_yrs),
        "family_history": bool(info.family_history_diabetes in (True, "True", "true", 1)),
        "labs": [{"test": k, "value": clean(r.value), "unit": r.unit, "date": r.date} for k, r in latest.iterrows()],
        "hba1c_history": [{"date": ms(d), "value": clean(v)} for d, v in zip(hba1c.date, hba1c.value)],
        "medications": [{"drug": r.drug, "dose": r.dose, "class": r.drug_class, "for": r.indication, "since": r.start_date}
                        for r in act.itertuples()],
        "diagnoses": [{"name": r.description, "code": r.icd10, "since": r.onset_date} for r in dx.itertuples()],
        "cgm14": {"mean": float(valid.mean()), "gmi": float(3.31 + 0.02392 * valid.mean()), "cv": float(valid.std() / valid.mean() * 100),
                  "bands": {k: round(float(v) * 100, 1) for k, v in bands.items()}, "coverage": float(np.isfinite(g).mean() * 100)},
        "agp": [{k: round(float(v), 1) for k, v in a.items()} for a in agp], "daily": daily,
        "twin": {"p10": round(p10, 1), "tau": round(tau), "w": round(w, 2), "pop_p10": round(pop[1], 1), "pop_tau": round(pop[0]),
                 "description": tw.describe(),
                 "carb_label": "high" if p10 > pop[1] * 1.2 else "low" if p10 < pop[1] * 0.8 else "average",
                 "recovery_label": "slow" if tau > pop[0] * 1.2 else "fast" if tau < pop[0] * 0.8 else "average",
                 "walk_label": "strong" if w > 1.2 else "weak" if w < 0.8 else "average"},
    }


def recent_context(pid, t, hours=4):
    ev = store.events.get(pid)
    if ev is None:
        return [], []
    w = ev[(ev.timestamp <= t) & (ev.timestamp > t - pd.Timedelta(hours=hours)) & ev.logged]
    meals = [(int((t - r.timestamp).total_seconds() // 60), float(r.logged_carbs_g), GI_NAMES.get(r.glycaemic_index, "medium"))
             for r in w[w.event_type == "meal"].itertuples() if np.isfinite(r.logged_carbs_g)]
    ex = [(int((t - r.timestamp).total_seconds() // 60), int(r.duration_min), r.detail) for r in w[w.event_type == "exercise"].itertuples()]
    return meals, ex


def typical_meal(pid, t):
    """The patient's usual logged main meal (median carbs, most common GI) from their history before t."""
    ev = store.events.get(pid)
    if ev is None:
        return None
    m = ev[(ev.timestamp < t) & (ev.event_type == "meal") & ev.logged & ev.meal_slot.isin(["lunch", "dinner"])]
    m = m[np.isfinite(m.logged_carbs_g)]
    if len(m) < 2:
        return None
    return float(m.logged_carbs_g.median()), str(m.glycaemic_index.mode().iloc[0])


def baseline_curve(pid, t, horizon=180):
    """ML forecast on a 5-min grid: anchored at current glucose, through the 30/60/90/120-min ensemble points,
    held flat after 2 h (beyond the models' horizon)."""
    pr = store.pred_row(pid, t)
    g, _ = current_glucose(pid, t)
    g = g if np.isfinite(g) else pr.g_now
    xs = [0] + HORIZONS
    ys = [g] + [float(pr[f"pred_fc{h}"]) for h in HORIZONS]
    mins = np.arange(0, horizon + 1, STEP)
    curve = np.interp(mins, xs, ys)
    band = np.interp(mins, [0] + HORIZONS, [0] + [store.mae[h] for h in HORIZONS])
    band[mins > 120] = store.mae[120]
    return mins, curve, band


def scenario_summary(base, alt, mins):
    def stats(c):
        return {"peak": float(c.max()), "peak_min": int(mins[int(np.argmax(c))]), "low": float(c.min()),
                "min_above_180": int((c > 180).sum() * STEP), "min_below_70": int((c < 70).sum() * STEP)}
    return {"baseline": stats(base), "scenario": stats(alt)}


@app.get("/api/patients/{pid}/snapshot")
def snapshot(pid: str, t: str | None = None, history_hours: int = 12):
    need(pid)
    t = store.snap_time(t)
    i = int((t - START) / pd.Timedelta(minutes=STEP))
    g = store.cgm[pid]
    lo = max(0, i - history_hours * 12)
    hist = [{"t": ms(START + pd.Timedelta(minutes=STEP * k)), "g": clean(g[k])} for k in range(lo, i + 1)]
    future = [{"t": ms(START + pd.Timedelta(minutes=STEP * k)), "g": clean(g[k])} for k in range(i + 1, min(T, i + 37))]
    pr = store.pred_row(pid, t)
    mins, curve, band = baseline_curve(pid, t)
    forecast = [{"t": ms(t + pd.Timedelta(minutes=int(m))), "g": round(float(c), 1), "lo": round(float(c - b), 1),
                 "hi": round(float(c + b), 1)} for m, c, b in zip(mins, curve, band)]
    now_g, d15 = current_glucose(pid, t)
    risks = {}
    for k in ("spike", "severe", "hypo"):
        p = float(pr[f"p_{k}"])
        applicable = not ((k == "spike" and now_g >= 180) or (k == "severe" and now_g >= 250) or (k == "hypo" and now_g < 70))
        risks[k] = {"p": round(p, 3), "threshold": store.thr[k], "level": risk_level(p, store.thr[k]) if applicable else "now",
                    "reasons": explain(pid, t, k) if applicable else []}

    ev = store.events.get(pid, pd.DataFrame(columns=["timestamp"]))
    ev = ev[(ev.timestamp > t - pd.Timedelta(hours=history_hours)) & (ev.timestamp <= t + pd.Timedelta(hours=3))]
    events = [{"t": ms(r.timestamp), "type": r.event_type, "label": r.detail, "logged": bool(r.logged),
               "future": bool(r.timestamp > t),
               "carbs": clean(r.logged_carbs_g) if r.event_type == "meal" else None,
               "taken": (None if r.event_type != "medication" else bool(str(r.dose_taken).lower() == "true")),
               "minutes": clean(r.duration_min) if r.event_type == "exercise" else None}
              for r in ev.itertuples() if r.event_type != "medication" or r.timestamp <= t]

    # twin suggestions for the current situation
    twin = store.twin(pid)
    meals, ex = recent_context(pid, t)
    sugg = []
    spike_risky = risks["spike"]["level"] in ("high", "medium", "now")
    if spike_risky:
        ate_recently = any(m <= 90 for m, _, _ in meals)
        # effects are measured exactly like the what-if panel: ML baseline + twin change, peak vs peak
        _, base_c, _ = baseline_curve(pid, t)
        peak = lambda ch: float((base_c + twin.whatif_delta(meals, ex, ch, horizon_min=180)[:len(base_c)]).max())
        if ate_recently:
            sugg.append({"action": "Walk 20 min now (meal still digesting)", "effect": round(peak([("walk", 0, 20)]) - peak([]), 0),
                         "kind": "lower", "changes": [["walk", 0, 20]]})
        else:
            usual = typical_meal(pid, t)
            if usual:
                carbs, gi = usual
                base_peak = peak([("meal", 0, carbs, gi)])
                swap = peak([("meal", 0, carbs * 0.8, "low")])
                walk = peak([("meal", 0, carbs, gi), ("walk", 30, 20)])
                ref = {"reference": [["meal", 0, round(carbs), gi]], "reference_label": f"Usual meal ({carbs:.0f} g, {gi} GI)"}
                sugg.append({"action": f"Next meal: swap usual {carbs:.0f} g {gi}-GI meal for a smaller low-GI one (roti, dal, salad)",
                             "effect": round(swap - base_peak, 0), "kind": "lower",
                             "changes": [["meal", 0, round(carbs * 0.8), "low"]], **ref})
                sugg.append({"action": "Next meal: walk 20 min, starting 30 min after eating",
                             "effect": round(walk - base_peak, 0), "kind": "lower",
                             "changes": [["meal", 0, round(carbs), gi], ["walk", 30, 20]], **ref})
        fe = store.features[pid].loc[t] if t in store.features[pid].index else None
        if (twin.regimen and fe is not None and t.hour >= 10
                and fe.get("doses_logged_today", 0) < fe.get("prescribed_doses_per_day", 0)):
            sugg.append({"action": "Confirm today's diabetes medicine was taken (not logged yet)", "effect": None,
                         "kind": "info", "changes": []})
    if risks["hypo"]["level"] in ("high", "medium", "now"):
        _, base_c, _ = baseline_curve(pid, t)
        d = twin.whatif_delta(meals, ex, [("meal", 0, 15, "high")], horizon_min=180)[:len(base_c)]
        low_now, low_after = float(base_c[:13].min()), float((base_c + d)[:13].min())
        sugg.append({"action": "Take 15 g fast sugar (glucose tablets / juice) and recheck in 15 min",
                     "effect": round(low_after - low_now, 0), "kind": "raise", "changes": [["meal", 0, 15, "high"]]})
    return {"time": ms(t), "glucose": clean(now_g), "delta15": clean(d15), "trend": trend_arrow(d15),
            "history": hist, "actual_future": future, "forecast": forecast, "risks": risks, "events": events,
            "suggestions": sugg, "recent_meals": [{"minutes_ago": m, "carbs": c, "gi": gi} for m, c, gi in meals]}


class WhatIf(BaseModel):
    t: str | None = None
    changes: list[list] = []            # [["meal", in_min, carbs, gi], ["walk", in_min, minutes], ["skip_drug", drug_class]]
    reference: list[list] | None = None  # optional comparison scenario, same format


def parse_changes(chs):
    out = []
    for c in chs or []:
        if c[0] == "meal" and float(c[2]) > 0:
            out.append(("meal", int(c[1]), float(c[2]), str(c[3])))
        elif c[0] == "walk" and int(c[2]) > 0:
            out.append(("walk", int(c[1]), int(c[2])))
        elif c[0] == "skip_drug":
            out.append(("skip_drug", str(c[1])))
    return out


@app.post("/api/patients/{pid}/whatif")
def whatif(pid: str, q: WhatIf):
    need(pid)
    t = store.snap_time(q.t)
    mins, base, _ = baseline_curve(pid, t)
    meals, ex = recent_context(pid, t)
    twin = store.twin(pid)

    def run(chs):
        d = twin.whatif_delta(meals, ex, parse_changes(chs), horizon_min=int(mins[-1]))
        return np.clip(base + d[:len(base)], 40, 450)

    alt = run(q.changes)
    ref = run(q.reference) if q.reference else None
    curve = [{"t": ms(t + pd.Timedelta(minutes=int(m))), "base": round(float(b), 1), "scenario": round(float(a), 1),
              **({"reference": round(float(ref[k]), 1)} if ref is not None else {})}
             for k, (m, b, a) in enumerate(zip(mins, base, alt))]
    summ = scenario_summary(base, alt, mins)
    if ref is not None:
        summ["reference"] = scenario_summary(base, ref, mins)["scenario"]
    return {"time": ms(t), "curve": curve, "summary": summ}
