"""
End-to-end check on a BRAND-NEW patient who is not in the 5,000-patient dataset.

1. Create the person (a persona you can edit below) -> full synthetic EHR + 14 days of CGM / smartwatch / events,
   using the same generators as the main dataset.
2. Run the trained ML models (LightGBM + GRU ensemble) over the 14 days: alerts, forecasts, spikes caught.
3. Build the personalized twin from week 1 only, compare its knobs with the hidden truth, and test it on week-2 meals.
4. Ask the twin what-if questions.

Usage:
    python src/new_patient_demo.py   # -> data/new_patient/*.csv, results/new_patient/report.html
"""
from __future__ import annotations

import base64
import io
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_features as bf  # noqa: E402
import generate_ehr as ge  # noqa: E402
import generate_wearable as gw  # noqa: E402
import train_gru as tg  # noqa: E402
import twin_simulator as tw  # noqa: E402
from train_lightgbm import event_detection  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_OUT = ROOT / "data" / "new_patient"
OUT = ROOT / "results" / "new_patient"
PID = "NEW001"

# ---- The new person. Edit freely.
PERSONA = {"name": "Sunita", "age": 48, "sex": "F", "region": "South", "setting": "Urban", "height_cm": 155.0, "bmi": 27.5,
           "diet": "Non-vegetarian", "activity_level": "Sedentary", "smoking": "Never", "alcohol": "None",
           "family_history_diabetes": True, "family_history_cvd": False, "diabetes_duration_yrs": 3.0}
ADHERENCE = 0.65
WANT_DRUGS = {"Biguanide", "Sulfonylurea"}      # regenerate until the simulated doctor has prescribed these


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# --------------------------------------------------------------------------
# 1. Create the patient
# --------------------------------------------------------------------------
def make_ehr(seed):
    rng = np.random.default_rng(seed)
    p = ge.sample_patients(1, rng)
    p["patient_id"], p["cohort"], p["staple"] = PID, "T2D", ge.STAPLE[PERSONA["region"]]
    for k, v in PERSONA.items():
        if k in p.columns:
            p[k] = v
    p["weight_kg"] = round(p.bmi.iloc[0] * (p.height_cm.iloc[0] / 100) ** 2, 1)
    p["waist_cm"] = round(38 + 1.95 * p.bmi.iloc[0] + rng.normal(0, 3), 1)
    gen = ge.sample_genetics(p, rng)
    tp = ge.sample_twin_params(p, gen, rng)
    tp["med_adherence"] = ADHERENCE
    st = ge.sample_comorbidity_state(p, tp, rng)
    sim = ge.PatientSim(p.iloc[0].to_dict(), tp.iloc[0].to_dict(), st.iloc[0].to_dict(), rng).run()

    labs = pd.DataFrame(sim.labs)
    labs["date"] = ge.day_to_date(labs.pop("day"))
    meds = pd.DataFrame(sim.meds).drop(columns="_effect")
    meds["start_date"] = ge.day_to_date(meds.pop("start_day"))
    end = meds.pop("end_day")
    meds["end_date"] = ge.day_to_date(end.fillna(0)).where(end.notna(), None)
    dx = pd.DataFrame(sim.dx)
    dx["onset_date"] = ge.day_to_date(dx.pop("onset_day"))
    p["reference_date"] = ge.REF_DATE.date()
    feats = ge.build_features(p, labs, meds, dx, gen)
    for c in bf.EHR_FLAGS:                                        # flags for drugs / diagnoses she doesn't have
        if c not in feats:
            feats[c] = 0
    for c in bf.EHR_NUMERIC:                                      # labs never measured for her
        if c not in feats:
            feats[c] = np.nan
    return p, gen, tp, labs, meds, dx, feats


def make_wearable(p, tp, feats, meds, seed):
    rng = np.random.default_rng(seed)
    df = p.merge(tp, on="patient_id").merge(feats[["patient_id", "last_hba1c", "hba1c_mean"]], on="patient_id")
    act = meds[meds.end_date.isna() & (meds.indication == "Type 2 diabetes")]
    df["regimen"] = [list(act[["drug", "drug_class", "dose"]].itertuples(index=False, name=None))]
    a1c = gw.target_a1c(df, rng)
    meals, doses, exercise, stress, nights, days = gw.simulate_behaviour(df, rng)
    sim = gw.simulate_chunk(df, meals, doses, exercise, stress, nights, days, a1c, rng)
    daily = gw.daily_summary(df, sim, nights, days, rng)
    to_ts = lambda t: gw.START + pd.to_timedelta(t * gw.STEP_MIN, unit="min")
    events = pd.concat([
        pd.DataFrame({"patient_id": PID, "timestamp": to_ts(meals.t), "event_type": "meal", "meal_slot": meals.slot,
                      "detail": meals.food, "carbs_g": meals.carbs_g.round(0), "glycaemic_index": [gw.GI_NAME[g] for g in meals.gi],
                      "logged": meals.logged, "logged_carbs_g": meals.logged_carbs_g.round(0)}),
        pd.DataFrame({"patient_id": PID, "timestamp": to_ts(doses.t), "event_type": "medication", "detail": doses.drug + " " + doses.dose,
                      "drug_class": doses.drug_class, "dose_taken": doses.taken, "logged": doses.logged}),
        pd.DataFrame({"patient_id": PID, "timestamp": to_ts(exercise.t), "event_type": "exercise",
                      "detail": [gw.EXERCISE[c][0] for c in exercise.code], "duration_min": exercise.n_bins * gw.STEP_MIN, "logged": True}),
    ]).sort_values("timestamp").reset_index(drop=True)
    day_ctx = days.assign(date=[(gw.START + pd.Timedelta(days=int(d))).date() for d in days.day])
    return sim, daily, events, day_ctx


# --------------------------------------------------------------------------
# 2. ML models
# --------------------------------------------------------------------------
def gru_sequence(sim, events):
    """Same 11 channels as train_gru.build_sequences, for one patient."""
    T = bf.T
    g, hr, st = sim["cgm"][:, 0], sim["hr"][:, 0], sim["steps"][:, 0]
    asleep = (sim["stage"][:, 0] >= 0).astype(np.float32)
    ev = events[events.logged.astype(bool)].copy()
    ev["bin"] = ((ev.timestamp - bf.START) / pd.Timedelta(minutes=bf.STEP)).astype(int)
    ev = ev[(ev.bin >= 0) & (ev.bin < T)]
    carbs, dose, exm = np.zeros(T, np.float32), np.zeros(T, np.float32), np.zeros(T, np.float32)
    m = ev[ev.event_type == "meal"]
    np.add.at(carbs, m.bin.to_numpy(), m.logged_carbs_g.fillna(0).to_numpy())
    np.add.at(dose, ev[ev.event_type == "medication"].bin.to_numpy(), 1.0)
    for b, d in ev[ev.event_type == "exercise"][["bin", "duration_min"]].itertuples(index=False):
        exm[b:b + int(d // bf.STEP)] = 1.0
    hour = (np.arange(T) % 288) * bf.STEP / 60
    ch = [np.nan_to_num((g - 140) / 50), np.isnan(g), np.nan_to_num((hr - 75) / 12), np.isnan(hr), np.log1p(np.nan_to_num(st)) / 6,
          asleep, carbs / 100, dose, exm, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
    return np.stack([np.asarray(c, np.float32) for c in ch], axis=1)[None]           # (1, T, 11)


def run_models(rows, seq):
    preds = rows[["patient_id", "timestamp", "g_now", "y_spike", "y_hypo", "y_severe", "g_p30", "g_p60", "g_p90", "g_p120"]].copy()
    feat_cols = lgb.Booster(model_file=str(ROOT / "models" / "lightgbm" / "spike.txt")).feature_name()
    X = rows[feat_cols]
    lg = {k: lgb.Booster(model_file=str(ROOT / "models" / "lightgbm" / f"{k}.txt")).predict(X)
          for k in ("spike", "hypo", "severe", "fc30", "fc60", "fc90", "fc120")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / "models" / "gru" / "gru.pt", map_location=device, weights_only=False)
    tg.SEQ_LEN = ck["seq_len"]
    model = tg.TwinGRU(seq.shape[2], len(ck["static_features"]), ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    d = rows.copy()
    d["p_idx"] = 0
    d["t_idx"] = ((d.timestamp - bf.START) / pd.Timedelta(minutes=bf.STEP)).astype(np.int64)
    tens = tg.to_tensors(d, np.array(ck["static_mean"], np.float32), np.array(ck["static_std"], np.float32), device)
    c, r = tg.predict(model, torch.tensor(seq, device=device, dtype=torch.float16), tens)
    g = rows.g_now.to_numpy()
    for name, lgb_p, gru_p in [("spike", lg["spike"], c[:, 0]), ("severe", lg["severe"], c[:, 1]), ("hypo", lg["hypo"], c[:, 2]),
                               ("fc30", g + lg["fc30"], g + r[:, 0]), ("fc60", g + lg["fc60"], g + r[:, 1]),
                               ("fc90", g + lg["fc90"], g + r[:, 2]), ("fc120", g + lg["fc120"], g + r[:, 3])]:
        preds[f"{name}_lgb"], preds[f"{name}_gru"], preds[name] = lgb_p, gru_p, (lgb_p + gru_p) / 2
    thr = {}
    for k in ("spike", "hypo", "severe"):
        a = json.loads((ROOT / "results" / "lightgbm" / "metrics.json").read_text())[k]["threshold"]
        b = json.loads((ROOT / "results" / "gru" / "metrics.json").read_text())[k]["threshold"]
        thr[k] = (a + b) / 2
    return preds, thr


# --------------------------------------------------------------------------
# 3. Twin
# --------------------------------------------------------------------------
def build_twin(sim, events, tp):
    g = sim["cgm"][:, 0].astype(float)
    ev = events[events.logged.astype(bool)].copy()
    ev["step"] = ((ev.timestamp - tw.START) / pd.Timedelta(minutes=tw.STEP_MIN)).astype(int)
    meals, ex = ev[ev.event_type == "meal"], ev[ev.event_type == "exercise"]
    wins = tw.meal_windows(g, meals, ex, 0, tw.FIT_DAYS)
    clean = [w for w in wins if w["ex"].sum() == 0]
    raw = tw.fit_patient(clean)
    ex_wins = [w for w in wins if w["ex"].sum() > 0]
    w_raw = tw.fit_walk(ex_wins, raw["tau"], raw["p10"]) if len(ex_wins) >= 2 else None

    fitted = pd.read_csv(tw.PARAMS_OUT)
    pri = fitted[(fitted.split == "train") & (fitted.cohort == "T2D")]
    k = float(fitted.shrinkage_k.iloc[0])
    pop_w = float(fitted[fitted.split == "train"].w_raw.median())
    a = raw["n_meals"] / (raw["n_meals"] + k)
    aw = len(ex_wins) / (len(ex_wins) + k) if w_raw is not None else 0.0
    twin = {"tau": a * raw["tau"] + (1 - a) * pri.tau_raw.median(), "p10": a * raw["p10"] + (1 - a) * pri.p10_raw.median(),
            "w": aw * (w_raw if w_raw is not None else pop_w) + (1 - aw) * pop_w}
    truth = {"tau": float(35 / (tp.insulin_sensitivity.iloc[0] * tp.beta_cell_function.iloc[0]) ** 0.35),
             "p10": float(min(tp.carb_sensitivity_mgdl_per_10g.iloc[0] * 1.5, 15.0)), "w": 1.0}
    population = {"tau": float(fitted[fitted.split == "train"].tau_raw.median()),
                  "p10": float(fitted[fitted.split == "train"].p10_raw.median()), "w": pop_w}

    val = []
    n = tw.PRE + tw.POST + 1
    for wd in tw.meal_windows(g, meals, ex, tw.FIT_DAYS, 14):
        actual, pre = tw.peak_rise(wd["y"])
        if not np.isfinite(actual):
            continue
        row = {"step": wd["step"], "carbs_logged": wd["carbs"], "carbs_true": wd["true_carbs"], "gi": wd["gi"],
               "actual_rise": actual, "y": wd["y"], "pre": pre}
        for name, prm in (("twin", twin), ("population", population), ("truth", truth)):
            curve = tw.simulate([(tw.PRE, wd["carbs"], wd["gi"])], wd["ex"], n, prm["tau"], prm["p10"], prm["w"])
            row[f"{name}_rise"], row[f"{name}_curve"] = float(curve.max()), curve
        val.append(row)
    return {"raw": raw, "twin": twin, "truth": truth, "population": population, "n_walk_meals": len(ex_wins),
            "week1_windows": len(clean), "val": val}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    for seed in range(100, 600):
        p, gen, tp, labs, meds, dx, feats = make_ehr(seed)
        act = set(meds[meds.end_date.isna() & (meds.indication == "Type 2 diabetes")].drug_class)
        if WANT_DRUGS <= act:
            break
    print(f"Created {PERSONA['name']} ({PID}) with seed {seed}: HbA1c {feats.last_hba1c.iloc[0]}, active meds "
          f"{sorted(meds[meds.end_date.isna()].drug)}, diagnoses {sorted(dx.description)}")
    sim, daily, events, day_ctx = make_wearable(p, tp, feats, meds, seed)
    for name, d in [("patients", p), ("labs", labs), ("medications", meds), ("diagnoses", dx), ("genetics", gen),
                    ("twin_params_hidden", tp), ("events", events), ("daily_summary", daily), ("day_context_hidden", day_ctx)]:
        d.to_csv(DATA_OUT / f"{name}.csv", index=False)
    times = bf.START + pd.to_timedelta(np.arange(bf.T) * bf.STEP, unit="min")
    pd.DataFrame({"timestamp": times, "glucose_mgdl": sim["cgm"][:, 0], "heart_rate_bpm": sim["hr"][:, 0],
                  "steps": sim["steps"][:, 0], "sleep_stage": gw.STAGES[sim["stage"][:, 0] + 1]}).to_csv(DATA_OUT / "stream_5min.csv", index=False)
    g = sim["cgm"][:, 0]
    cgm_stats = {"mean": float(np.nanmean(g)), "tir": float(np.nanmean((g >= 70) & (g <= 180)) / np.mean(~np.isnan(g)) * 100),
                 "tar": float(np.nansum(g > 180) / (~np.isnan(g)).sum() * 100), "tbr": float(np.nansum(g < 70) / (~np.isnan(g)).sum() * 100)}
    print(f"14-day CGM: mean {cgm_stats['mean']:.0f} mg/dL, in range {cgm_stats['tir']:.0f}%, above 180 {cgm_stats['tar']:.0f}%, "
          f"below 70 {cgm_stats['tbr']:.1f}%")

    # ---- ML
    ev = events.copy()
    ev["bin"] = ((ev.timestamp - bf.START) / pd.Timedelta(minutes=bf.STEP)).astype(int)
    ev = ev[(ev.bin >= 0) & (ev.bin < bf.T)]
    ev["logged"] = ev.logged.astype(bool)
    ehr = bf.prepare_ehr(feats, meds, pd.DataFrame({"patient_id": [PID], "split": ["new"]}))
    rows = bf.build_chunk([PID], sim["cgm"].T, sim["hr"].T, sim["steps"].T, sim["stage"].T >= 0, ev, daily, ehr, 30)
    preds, thr = run_models(rows, gru_sequence(sim, events))
    m = preds.y_spike.notna()
    ml = {"spike_auroc": float(roc_auc_score(preds.y_spike[m], preds.spike[m])) if 0 < preds.y_spike[m].sum() < m.sum() else None,
          "spike_events": event_detection(preds.assign(spike=preds.spike.fillna(0)), "spike", thr["spike"], "spike"),
          "hypo_events": event_detection(preds.assign(hypo=preds.hypo.fillna(0)), "hypo", thr["hypo"], "hypo"),
          "alerts_per_day": float(((preds.spike >= thr["spike"]) & ~(preds.spike.shift(1) >= thr["spike"])).sum() / 14),
          "mae60": float(np.nanmean(np.abs(preds.fc60 - preds.g_p60))), "mae120": float(np.nanmean(np.abs(preds.fc120 - preds.g_p120)))}
    print(f"ML: spike AUROC {ml['spike_auroc']:.3f}, spikes caught {ml['spike_events']['detection_rate']:.0%} of "
          f"{ml['spike_events']['events']}, lows caught {ml['hypo_events']['detection_rate']} of {ml['hypo_events']['events']}, "
          f"MAE60 {ml['mae60']:.1f}")

    # ---- Twin
    T = build_twin(sim, events, tp)
    val = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in T["val"]])
    errs = {n: float((val[f"{n}_rise"] - val.actual_rise).abs().mean()) for n in ("twin", "population", "truth")}
    print(f"Twin learned from {T['week1_windows']} week-1 meals: p10 {T['twin']['p10']:.1f} (truth {T['truth']['p10']:.1f}), "
          f"tau {T['twin']['tau']:.0f} (truth {T['truth']['tau']:.0f}), walk {T['twin']['w']:.2f}")
    print(f"Week-2 peak-rise error over {len(val)} meals: twin {errs['twin']:.1f} | population {errs['population']:.1f} | "
          f"perfect knowledge {errs['truth']:.1f} mg/dL")

    twin = tw.PatientTwin(PID, T["twin"]["tau"], T["twin"]["p10"], T["twin"]["w"],
                          list(zip(meds[meds.end_date.isna()].drug, meds[meds.end_date.isna()].drug_class)))
    whatifs = {"2 roti + dal (60 g, low GI)": [("meal", 0, 60, "low")],
               "Idli-sambar (60 g, high GI)": [("meal", 0, 60, "high")],
               "Rice meal 85 g (her usual lunch)": [("meal", 0, 85, "high")],
               "Rice 85 g + 20 min walk after 30 min": [("meal", 0, 85, "high"), ("walk", 30, 20)],
               "Rice 85 g + 30 min walk after 15 min": [("meal", 0, 85, "high"), ("walk", 15, 30)],
               "Skip metformin today": [("skip_drug", "Biguanide")],
               "Skip gliclazide/glimepiride today": [("skip_drug", "Sulfonylurea")]}
    wi = []
    for name, ch in whatifs.items():
        d = twin.whatif_delta(changes=ch)
        wi.append({"question": name, "peak": float(d.max()), "peak_min": int(np.argmax(d)) * 5, "at_2h": float(d[24])})
    for r in wi:
        print(f"  what-if {r['question']:40s} peak {r['peak']:+5.0f} mg/dL at {r['peak_min']} min, at 2 h {r['at_2h']:+5.0f}")

    # files the dashboard uses to show her like any other patient
    preds.to_csv(DATA_OUT / "predictions.csv", index=False)
    rows.to_parquet(DATA_OUT / "features.parquet", index=False)
    (DATA_OUT / "twin_fitted.json").write_text(json.dumps({"patient_id": PID, **T["twin"], "n_meals": T["raw"]["n_meals"]}, indent=2))

    summary = {"persona": PERSONA, "seed": seed, "hba1c": float(feats.last_hba1c.iloc[0]),
               "active_meds": sorted(meds[meds.end_date.isna()].drug), "diagnoses": sorted(dx.description), "cgm": cgm_stats,
               "ml": ml, "twin": {k: T[k] for k in ("raw", "twin", "truth", "population", "n_walk_meals", "week1_windows")},
               "twin_week2_peak_error": errs, "week2_meals": len(val), "whatif": wi}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, preds, thr, events, sim, T, twin)
    print(f"Report -> {OUT / 'report.html'}")


def write_report(s, preds, thr, events, sim, T, twin):
    charts = []
    # 3-day view with alerts
    times = bf.START + pd.to_timedelta(np.arange(bf.T) * bf.STEP, unit="min")
    d0 = bf.START + pd.Timedelta(days=8)
    d1 = d0 + pd.Timedelta(days=3)
    win = (times >= d0) & (times < d1)
    pw = preds[(preds.timestamp >= d0) & (preds.timestamp < d1)]
    ew = events[(events.timestamp >= d0) & (events.timestamp < d1)]
    fig, ax = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax[0].plot(times[win], sim["cgm"][win, 0], color="#334155", lw=1.1, label="CGM")
    ax[0].scatter(pw.timestamp + pd.Timedelta("60min"), pw.fc60, s=14, color="#c2410c", label="forecast made 60 min earlier")
    ax[0].axhspan(70, 180, color="#16a34a", alpha=0.07)
    for r in ew[ew.event_type == "meal"].itertuples():
        ax[0].axvline(r.timestamp, color="#94a3b8", ls=":", lw=0.7)
        ax[0].annotate(f"{r.detail} {int(r.carbs_g)}g{'' if r.logged else ' (not logged)'}", (r.timestamp, 45), fontsize=6.5,
                       rotation=90, va="bottom", color="#475569")
    for r in ew[ew.event_type == "medication"].itertuples():
        ax[0].plot(r.timestamp, 395, marker="v", color="#16a34a" if r.dose_taken else "#dc2626", ms=6)
    alerts = pw[(pw.spike >= thr["spike"]) & (pw.g_now < 180)]
    ax[0].scatter(alerts.timestamp, np.full(len(alerts), 375), marker="^", color="#c2410c", s=30, label="spike alert")
    lows = pw[pw.hypo >= thr["hypo"]]
    ax[0].scatter(lows.timestamp, np.full(len(lows), 360), marker="^", color="#2563eb", s=30, label="hypo alert")
    ax[0].set(ylim=(35, 410), ylabel="mg/dL", title=f"{s['persona']['name']} ({PID}), days 9-11: unseen by any model during training")
    ax[0].legend(loc="upper left", fontsize=8, ncol=4)
    ax[1].plot(pw.timestamp, pw.spike, color="#c2410c", marker="o", ms=3, label="spike risk")
    ax[1].plot(pw.timestamp, pw.hypo, color="#2563eb", marker="o", ms=3, label="hypo risk")
    ax[1].axhline(thr["spike"], color="#c2410c", ls="--", lw=0.7)
    ax[1].axhline(thr["hypo"], color="#2563eb", ls="--", lw=0.7)
    ax[1].set(ylim=(0, 1), ylabel="risk")
    ax[1].legend(loc="upper left", fontsize=8)
    charts.append(img(fig))

    # twin on week-2 meals
    vals = T["val"][:4]
    if vals:
        fig, axes = plt.subplots(1, len(vals), figsize=(4 * len(vals), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        tmin = (np.arange(tw.PRE + tw.POST + 1) - tw.PRE) * tw.STEP_MIN
        for a, r in zip(axes, vals):
            a.plot(tmin, r["y"], color="#334155", label="actual CGM")
            a.plot(tmin, r["pre"] + r["twin_curve"], color="#15803d", label="her twin (learned week 1)")
            a.plot(tmin, r["pre"] + r["population_curve"], color="#94a3b8", ls="--", label="population average")
            a.set(title=f"logged {int(r['carbs_logged'])} g (true {int(r['carbs_true'])} g), {['high', 'medium', 'low'][r['gi']]} GI",
                  xlabel="minutes from meal")
        axes[0].set_ylabel("mg/dL")
        axes[0].legend(fontsize=7)
        charts.append(img(fig))

    tw_rows = "".join(f"<tr><td>{lab}</td><td>{T['twin'][k]:.{dp}f}</td><td>{T['population'][k]:.{dp}f}</td><td>{T['truth'][k]:.{dp}f}</td></tr>"
                      for lab, k, dp in [("Carb sensitivity (mg/dL per 10 g)", "p10", 1), ("Recovery time tau (min)", "tau", 0),
                                         ("Walking effect (x average)", "w", 2)])
    wi_rows = "".join(f"<tr><td>{r['question']}</td><td>{r['peak']:+.0f} mg/dL</td><td>{r['peak_min']} min</td><td>{r['at_2h']:+.0f} mg/dL</td></tr>"
                      for r in s["whatif"])
    ml, e = s["ml"], s["twin_week2_peak_error"]
    lows = ml["hypo_events"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>New Patient Check</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.muted{{color:#64748b}} img{{max-width:100%}} .card{{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 16px}}</style></head><body>
<h1>New patient through the digital twin: {s['persona']['name']} ({PID})</h1>
<div class="card"><b>{s['persona']['age']}{s['persona']['sex']}</b>, {s['persona']['region']} India ({s['persona']['setting']}), rice-based diet,
{s['persona']['activity_level'].lower()}, BMI {s['persona']['bmi']}, T2D for {s['persona']['diabetes_duration_yrs']:.0f} years,
HbA1c <b>{s['hba1c']}%</b>, takes her medicines ~{ADHERENCE:.0%} of the time.<br>
Medicines: {', '.join(s['active_meds'])}<br>Diagnoses: {', '.join(s['diagnoses'])}<br>
14-day CGM: mean {s['cgm']['mean']:.0f} mg/dL, in range {s['cgm']['tir']:.0f}%, above 180 {s['cgm']['tar']:.0f}%, below 70 {s['cgm']['tbr']:.1f}%</div>
<p class="muted">She is not part of the 5,000-patient dataset: no model or prior was ever trained on her data.</p>
<h2>1. ML alerts and forecasts (LightGBM + GRU ensemble)</h2>
<table><tr><td>Spike-alert AUROC on her 14 days</td><td>{ml['spike_auroc']:.3f}</td></tr>
<tr><td>Spikes (&gt;180) caught in advance</td><td>{ml['spike_events']['detection_rate']:.0%} of {ml['spike_events']['events']}</td></tr>
<tr><td>Lows (&lt;70) caught in advance</td><td>{'&ndash;' if lows['detection_rate'] is None else f"{lows['detection_rate']:.0%}"} of {lows['events']}</td></tr>
<tr><td>Spike alerts per day</td><td>{ml['alerts_per_day']:.1f}</td></tr>
<tr><td>Forecast error, 60 / 120 min ahead</td><td>{ml['mae60']:.1f} / {ml['mae120']:.1f} mg/dL</td></tr></table>
{charts[0]}
<h2>2. Her personalized twin (learned from week 1 only)</h2>
<table><tr><th>Knob</th><th>Her twin</th><th>Population average</th><th>Hidden truth</th></tr>{tw_rows}</table>
<p>Week 2, {s['week2_meals']} new meals: error in predicted post-meal peak = <b>{e['twin']:.1f} mg/dL</b> with her twin vs
{e['population']:.1f} with population values ({e['truth']:.1f} even with perfect knowledge of her physiology, because of
carb-estimate errors, unlogged snacks and daily variation).</p>
{charts[1] if len(charts) > 1 else ''}
<h2>3. What-if questions for {s['persona']['name']}</h2>
<p class="muted">{twin.describe()}</p>
<table><tr><th>If she...</th><th>Peak change</th><th>Peak at</th><th>Change after 2 h</th></tr>{wi_rows}</table>
</body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
