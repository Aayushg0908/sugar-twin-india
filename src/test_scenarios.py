"""
Scenario tests for the trained models (LightGBM, GRU and their average) before building the dashboard.
Everything runs on held-out TEST patients.

A. Scenario performance: how well each model does in specific real-life situations
   (feast, missed medicines, illness, fasting on sulfonylurea/insulin, post-meal walk, night,
   poor sleep, healthy person after a big meal, sensor gaps), with PASS/FAIL criteria.
B. What-if (counterfactual) tests: change one input and check the models react like physiology
   (bigger meal -> higher risk, walk -> lower, missing sensor data -> still sensible output).
C. Visual walkthrough of three real test patient-days.

Usage:
    python src/test_scenarios.py   # -> results/scenario_tests/{report.html, results.json}
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
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
from build_features import START, STEP, T  # noqa: E402
import train_gru as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "scenario_tests"
W = ROOT / "data" / "wearable"
FEATURES = ROOT / "data" / "model" / "features.parquet"
ID_COLS = ["patient_id", "timestamp", "split"]
TARGETS = ["y_spike", "y_severe", "y_hypo", "g_p30", "g_p60", "g_p90", "g_p120"]
MODELS = ["LightGBM", "GRU", "Ensemble"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# --------------------------------------------------------------------------
# Data: test predictions of both models + context of each moment
# --------------------------------------------------------------------------
def load_predictions():
    keys = ["patient_id", "timestamp"]
    lg = pd.read_parquet(RES / "lightgbm" / "test_predictions.parquet")
    gr = pd.read_parquet(RES / "gru" / "test_predictions.parquet")
    pcols = [c for c in gr.columns if c.startswith(("p_", "pred_"))]
    df = lg.rename(columns={c: f"{c}_LightGBM" for c in pcols}).merge(
        gr[keys + pcols].rename(columns={c: f"{c}_GRU" for c in pcols}), on=keys)
    for c in pcols:
        df[f"{c}_Ensemble"] = (df[f"{c}_LightGBM"] + df[f"{c}_GRU"]) / 2
    thr = {}
    for model, folder in [("LightGBM", "lightgbm"), ("GRU", "gru")]:
        m = json.loads((RES / folder / "metrics.json").read_text())
        thr[model] = {k: m[k]["threshold"] for k in ("spike", "severe", "hypo")}
    thr["Ensemble"] = {k: (thr["LightGBM"][k] + thr["GRU"][k]) / 2 for k in ("spike", "severe", "hypo")}
    return df, thr


def add_context(df):
    """Attach day context (hidden truth, used only to SELECT scenarios) and true meal / exercise timing."""
    ctx = pd.read_csv(W / "day_context.csv")
    df["date"] = df.timestamp.dt.date.astype(str)
    df = df.merge(ctx, on=["patient_id", "date"], how="left")
    feats = pd.read_parquet(FEATURES, columns=["patient_id", "timestamp", "g_coverage_6h", "sleep_hours_last_night"],
                            filters=[("split", "==", "test")])
    df = df.merge(feats, on=["patient_id", "timestamp"], how="left")

    ev = pd.read_csv(W / "events.csv", parse_dates=["timestamp"],
                     usecols=["patient_id", "timestamp", "event_type", "meal_slot", "carbs_g", "duration_min"])
    code = {p: i for i, p in enumerate(sorted(df.patient_id.unique()))}
    ev = ev[ev.patient_id.isin(code)]
    key = lambda pid, ts: pid.map(code).to_numpy() * T + ((ts - START) / pd.Timedelta(minutes=STEP)).astype(int).to_numpy()
    r_key = key(df.patient_id, df.timestamp)

    meals = ev[(ev.event_type == "meal") & ev.meal_slot.isin(["breakfast", "lunch", "dinner"])]
    mk = key(meals.patient_id, meals.timestamp)
    o = np.argsort(mk)
    mk, carbs = mk[o], meals.carbs_g.to_numpy()[o]
    i = np.searchsorted(mk, r_key, side="right") - 1                       # last main meal at or before t
    ok = (i >= 0) & (mk[np.maximum(i, 0)] // T == r_key // T)
    df["min_since_true_meal"] = np.where(ok, (r_key - mk[np.maximum(i, 0)]) * STEP, np.nan)
    df["true_meal_carbs"] = np.where(ok, carbs[np.maximum(i, 0)], np.nan)

    ex = ev[ev.event_type == "exercise"]
    ek = key(ex.patient_id, ex.timestamp)
    o = np.argsort(ek)
    ek = ek[o]
    # was there exercise starting between the last meal and 60 min after it?
    meal_key = np.where(ok, mk[np.maximum(i, 0)], -1)
    j = np.searchsorted(ek, meal_key, side="left")
    jj = np.minimum(j, len(ek) - 1)
    df["walk_after_meal"] = ok & (ek[jj] - meal_key <= 12) & (ek[jj] >= meal_key) & (ek[jj] // T == meal_key // T)
    df["hour"] = df.timestamp.dt.hour + df.timestamp.dt.minute / 60
    return df


# --------------------------------------------------------------------------
# A. Scenario performance
# --------------------------------------------------------------------------
def scenario_masks(df):
    t2d = df.cohort == 2
    risky = (df.on_sulfonylurea + df.on_basal_insulin) > 0
    after_meal = df.min_since_true_meal.between(0, 120)
    normal_day = df.day_type.isin(["weekday", "weekend"]) & (df.missed_dose_frac == 0) & ~df.poor_sleep_prev.astype(bool)
    return {
        "1. Normal day, after a meal (T2D)": ("spike", t2d & normal_day & after_meal),
        "2. Feast / wedding evening (T2D)": ("spike", t2d & (df.day_type == "feast") & (df.hour >= 18)),
        "3. Missed medicine day (T2D)": ("spike", t2d & (df.missed_dose_frac >= 0.5)),
        "4. Illness day (T2D)": ("spike", t2d & (df.day_type == "illness")),
        "5. Fasting day on SU/insulin (hypo risk)": ("hypo", t2d & risky & (df.day_type == "fasting")),
        "6a. After meal WITH a walk (T2D)": ("spike", t2d & after_meal & df.walk_after_meal),
        "6b. After meal, NO walk (T2D)": ("spike", t2d & after_meal & ~df.walk_after_meal),
        "7. Night 00:00-05:00, SU/insulin (hypo)": ("hypo", t2d & risky & (df.hour < 5)),
        "8. Poor sleep last night (T2D)": ("spike", t2d & df.poor_sleep_prev.astype(bool)),
        "9. Healthy person after a big meal (false alarms)": ("spike", (df.cohort == 0) & after_meal & (df.true_meal_carbs >= 70)),
        "10. Sensor gaps (<80% CGM data in last 6 h)": ("spike", df.g_coverage_6h < 0.8),
    }


def scenario_tests(df, thr):
    rows, results = [], {}
    for name, (alert, mask) in scenario_masks(df).items():
        target = f"y_{alert}"
        d = df[mask & df[target].notna()]
        y = d[target].to_numpy()
        res = {"alert": alert, "n": int(len(d)), "patients": int(d.patient_id.nunique()), "event_rate": float(y.mean()) if len(y) else None}
        for model in MODELS:
            p = d[f"p_{alert}_{model}"].to_numpy()
            pred = p >= thr[model][alert]
            r = {"mean_risk": float(p.mean()) if len(p) else None,
                 "sensitivity": float(pred[y == 1].mean()) if (y == 1).sum() else None,
                 "false_alarm_rate": float(pred[y == 0].mean()) if (y == 0).sum() else None,
                 "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}
            m60 = d.g_p60.notna().to_numpy()
            r["mae60"] = float(np.abs(d[f"pred_fc60_{model}"].to_numpy()[m60] - d.g_p60.to_numpy()[m60]).mean()) if m60.any() else None
            res[model] = r
        results[name] = res

    # pass / fail rules
    checks = []
    for name, res in results.items():
        e = res["Ensemble"]
        if name.startswith("9."):
            ok = e["false_alarm_rate"] is not None and e["false_alarm_rate"] <= 0.05
            rule = f"false alarms <= 5% (got {e['false_alarm_rate']:.1%})"
        elif name.startswith(("5.", "7.")):
            ok = e["sensitivity"] is not None and e["sensitivity"] >= 0.6
            rule = f"catches >= 60% of lows (got {e['sensitivity']:.0%})" if e["sensitivity"] is not None else "no lows in this scenario"
        elif name.startswith("10."):
            ok = e["auroc"] is not None and e["auroc"] >= 0.85
            rule = f"AUROC >= 0.85 despite gaps (got {e['auroc']:.3f})"
        else:
            ok = e["sensitivity"] is not None and e["sensitivity"] >= 0.7 and (e["auroc"] or 0) >= 0.85
            rule = f"catches >= 70% of spikes and AUROC >= 0.85 (got {e['sensitivity']:.0%}, {e['auroc']:.3f})" \
                if e["sensitivity"] is not None else "no spikes in this scenario"
        checks.append((name, ok, rule))
    # Walkers and non-walkers differ in many ways (walks follow late dinners, walkers are more active), so the
    # observed groups are not a clean experiment. The fair check: does the model match what ACTUALLY happened in each group?
    sel = (df.cohort == 2) & df.min_since_true_meal.between(30, 90) & df.g_p60.notna()
    gaps = []
    for w in (True, False):
        d = df[sel & (df.walk_after_meal == w)]
        gaps.append((w, float((d.g_p60 - d.g_now).mean()), float((d.pred_fc60_Ensemble - d.g_now).mean())))
    ok = all(abs(a_ - p_) <= 3 for _, a_, p_ in gaps)
    checks.append(("6. Post-meal forecast matches reality with and without a walk", ok,
                   "; ".join(f"{'walk' if w else 'no walk'}: actual {a_:+.1f} vs predicted {p_:+.1f} mg/dL in 60 min"
                             for w, a_, p_ in gaps) + " (need within 3 mg/dL)"))
    return results, checks


# --------------------------------------------------------------------------
# B. What-if tests (change inputs, check direction of the response)
# --------------------------------------------------------------------------
def whatif_tests(n_rows=400):
    feats = pd.read_parquet(FEATURES, filters=[("split", "==", "test")])
    cand = feats[(feats.cohort == 2) & feats.g_now.between(90, 160) & (feats.min_since_meal >= 180)
                 & (feats.min_since_exercise >= 120) & (feats.asleep_now == 0) & feats.hour.between(8, 21)]
    rows = cand.sample(min(n_rows, len(cand)), random_state=0).reset_index(drop=True)
    feat_cols = [c for c in feats.columns if c not in ID_COLS + TARGETS]
    del feats

    # ---- LightGBM: edit the engineered features
    boosters = {k: lgb.Booster(model_file=str(ROOT / "models" / "lightgbm" / f"{k}.txt")) for k in ("spike", "hypo", "fc60", "fc120")}

    def lgb_predict(X):
        return {"spike": boosters["spike"].predict(X[feat_cols]), "hypo": boosters["hypo"].predict(X[feat_cols]),
                "fc60": X.g_now.to_numpy() + boosters["fc60"].predict(X[feat_cols]),
                "fc120": X.g_now.to_numpy() + boosters["fc120"].predict(X[feat_cols])}

    def lgb_meal(X, carbs, walk=False):
        X = X.copy()
        X["min_since_meal"], X["last_meal_carbs"], X["last_meal_gi"] = 0.0, carbs, 1.0
        for c in ("carbs_1h", "carbs_3h", "carbs_today"):
            X[c] = X[c] + carbs
        if walk:
            X["min_since_exercise"], X["exercise_min_3h"] = 0.0, 30.0
            X["steps_30m"], X["steps_1h"], X["steps_3h"] = X.steps_30m + 2900, X.steps_1h + 2900, X.steps_3h + 2900
        return X

    def lgb_gap(X):
        X = X.copy()
        for c in ("g_delta_15", "g_delta_30", "g_delta_60", "g_accel_15", "g_mean_1h", "g_std_1h", "g_max_1h", "g_min_1h"):
            X[c] = np.nan
        X["g_coverage_6h"] = X.g_coverage_6h - 12 / 72
        return X

    # ---- GRU: edit the raw input sequence
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ROOT / "models" / "gru" / "gru.pt", map_location=device, weights_only=False)
    tg.SEQ_LEN = ck["seq_len"]
    pids, seq_all = tg.build_sequences(device)
    model = tg.TwinGRU(seq_all.shape[2], len(ck["static_features"]), ck["config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    p_idx = torch.tensor(rows.patient_id.map({p: i for i, p in enumerate(pids)}).to_numpy(), device=device)
    t_idx = torch.tensor(((rows.timestamp - START) / pd.Timedelta(minutes=STEP)).astype(int).to_numpy(), device=device)
    base_seq = tg.batch_seq(seq_all, p_idx, t_idx).float()
    mean, std = np.array(ck["static_mean"], np.float32), np.array(ck["static_std"], np.float32)

    def gru_predict(seq, X):
        x = np.nan_to_num((X[ck["static_features"]].to_numpy(np.float32) - mean) / std)
        with torch.no_grad():
            c, r = model(seq, torch.tensor(x, device=device))
        c, r = torch.sigmoid(c).cpu().numpy(), r.cpu().numpy() * ck["reg_scale"]
        g = X.g_now.to_numpy()
        return {"spike": c[:, 0], "hypo": c[:, 2], "fc60": g + r[:, 1], "fc120": g + r[:, 3]}

    def gru_meal(carbs, walk=False):
        s = base_seq.clone()
        s[:, -1, 6] += carbs / 100                                   # logged carbs channel at t
        if walk:
            s[:, -6:, 8] = 1.0                                       # exercise channel, last 30 min
            s[:, -6:, 4] = float(np.log1p(480) / 6)                  # steps channel
        return s

    def gru_gap():
        s = base_seq.clone()
        s[:, -12:-1, 0] = 0.0                                        # glucose values missing for the last hour
        s[:, -12:-1, 1] = 1.0                                        # (keep only the current reading)
        return s

    scen = {}
    scen["LightGBM"] = {"base": lgb_predict(rows), "meal60": lgb_predict(lgb_meal(rows, 60)),
                        "meal120": lgb_predict(lgb_meal(rows, 120)), "meal60_walk": lgb_predict(lgb_meal(rows, 60, True)),
                        "gap": lgb_predict(lgb_gap(rows))}
    X_meal = {c: lgb_meal(rows, c) for c in (60, 120)}
    scen["GRU"] = {"base": gru_predict(base_seq, rows), "meal60": gru_predict(gru_meal(60), X_meal[60]),
                   "meal120": gru_predict(gru_meal(120), X_meal[120]),
                   "meal60_walk": gru_predict(gru_meal(60, True), lgb_meal(rows, 60, True)),
                   "gap": gru_predict(gru_gap(), rows)}
    del seq_all
    if device.type == "cuda":
        torch.cuda.empty_cache()

    checks, table = [], []
    for model_name, s in scen.items():
        def frac(a, b, key):
            return float((s[a][key] > s[b][key]).mean())
        tests = [
            ("60 g meal raises spike risk", frac("meal60", "base", "spike"), 0.9),
            ("60 g meal raises 2-h forecast", frac("meal60", "base", "fc120"), 0.9),
            ("120 g meal raises forecast more than 60 g", frac("meal120", "meal60", "fc120"), 0.9),
            ("Walk after meal lowers 2-h forecast", 1 - frac("meal60_walk", "meal60", "fc120"), 0.8),
            ("Walk after meal lowers spike risk", 1 - frac("meal60_walk", "meal60", "spike"), 0.8),
        ]
        for label, share, need in tests:
            checks.append((f"{model_name}: {label}", share >= need, f"true for {share:.0%} of {len(rows)} moments (need {need:.0%})"))
        gap_ok = np.isfinite(s["gap"]["fc120"]).all() and (s["gap"]["fc120"] > 40).all() and (s["gap"]["fc120"] < 400).all()
        shift = float(np.median(np.abs(s["gap"]["fc120"] - s["base"]["fc120"])))
        checks.append((f"{model_name}: 1 h sensor gap -> still sensible output", bool(gap_ok and shift < 25),
                       f"all forecasts within 40-400 mg/dL, median change {shift:.1f} mg/dL (need < 25)"))
        table.append({"model": model_name,
                      "base_risk": float(np.median(s["base"]["spike"])), "meal60_risk": float(np.median(s["meal60"]["spike"])),
                      "meal120_risk": float(np.median(s["meal120"]["spike"])), "meal60_walk_risk": float(np.median(s["meal60_walk"]["spike"])),
                      "base_fc120": float(np.median(s["base"]["fc120"])), "meal60_fc120": float(np.median(s["meal60"]["fc120"])),
                      "meal120_fc120": float(np.median(s["meal120"]["fc120"])), "meal60_walk_fc120": float(np.median(s["meal60_walk"]["fc120"]))})
    return checks, table, len(rows)


# --------------------------------------------------------------------------
# C. Visual walkthrough
# --------------------------------------------------------------------------
def walkthrough(df, thr):
    cgm = pd.read_parquet(W / "cgm.parquet", filters=[("patient_id", "in", list(df.patient_id.unique()))])
    ev = pd.read_csv(W / "events.csv", parse_dates=["timestamp"])
    picks = []
    for label, mask in [("Feast day, T2D", (df.cohort == 2) & (df.day_type == "feast")),
                        ("Fasting day on sulfonylurea/insulin", (df.cohort == 2) & (df.day_type == "fasting")
                         & ((df.on_sulfonylurea + df.on_basal_insulin) > 0)),
                        ("Normal day, healthy person", (df.cohort == 0) & (df.day_type == "weekday"))]:
        cands = df[mask].groupby(["patient_id", "date"]).y_spike.mean().sort_values(ascending=False)
        if len(cands):
            picks.append((label, *cands.index[0 if label != "Fasting day on sulfonylurea/insulin" else -1]))
    charts = []
    for label, pid, date in picks:
        day0, day1 = pd.Timestamp(date), pd.Timestamp(date) + pd.Timedelta("1D")
        c = cgm[(cgm.patient_id == pid) & cgm.timestamp.between(day0, day1)]
        p = df[(df.patient_id == pid) & df.timestamp.between(day0, day1)]
        e = ev[(ev.patient_id == pid) & ev.timestamp.between(day0, day1)]
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
        ax[0].plot(c.timestamp, c.glucose_mgdl, color="#334155", lw=1.2, label="actual CGM")
        ax[0].scatter(p.timestamp + pd.Timedelta("60min"), p.pred_fc60_Ensemble, s=14, color="#c2410c", label="forecast made 60 min earlier")
        ax[0].axhspan(70, 180, color="#16a34a", alpha=0.07)
        for r in e[e.event_type == "meal"].itertuples():
            ax[0].axvline(r.timestamp, color="#94a3b8", ls=":", lw=0.8)
            ax[0].annotate(f"{r.detail} {int(r.carbs_g)}g", (r.timestamp, 45), fontsize=7, rotation=90, va="bottom", color="#475569")
        for r in e[e.event_type == "medication"].itertuples():
            ax[0].plot(r.timestamp, 395, marker="v", color="#16a34a" if r.dose_taken else "#dc2626", ms=6)
        ax[0].set(ylim=(35, 410), ylabel="mg/dL", title=f"{label}: {pid}, {date}")
        ax[0].legend(loc="upper left", fontsize=8)
        ax[1].plot(p.timestamp, p.p_spike_Ensemble, color="#c2410c", marker="o", ms=3, label="spike risk (>180 in 2 h)")
        ax[1].plot(p.timestamp, p.p_hypo_Ensemble, color="#2563eb", marker="o", ms=3, label="hypo risk (<70 in 2 h)")
        ax[1].axhline(thr["Ensemble"]["spike"], color="#c2410c", ls="--", lw=0.7)
        ax[1].axhline(thr["Ensemble"]["hypo"], color="#2563eb", ls="--", lw=0.7)
        ax[1].set(ylim=(0, 1), ylabel="risk")
        ax[1].legend(loc="upper left", fontsize=8)
        charts.append(img(fig))
    return charts


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log("Loading test predictions")
    df, thr = load_predictions()
    df = add_context(df)
    log("A. Scenario tests")
    results, a_checks = scenario_tests(df, thr)
    log("B. What-if tests")
    b_checks, b_table, n_whatif = whatif_tests()
    log("C. Walkthrough charts")
    charts = walkthrough(df, thr)

    all_checks = a_checks + b_checks
    passed = sum(ok for _, ok, _ in all_checks)
    (OUT / "results.json").write_text(json.dumps({"scenarios": results, "whatif": b_table,
                                                  "checks": [{"test": n, "pass": bool(o), "detail": d} for n, o, d in all_checks]},
                                                 indent=2, default=float))

    def fmt(v, pct=False):
        return "&ndash;" if v is None else (f"{v:.0%}" if pct else f"{v:.3f}")
    def fmt_mae(v):
        return "&ndash;" if v is None else f"{v:.1f}"
    sc_rows = ""
    for name, r in results.items():
        sc_rows += (f"<tr><td>{name}</td><td>{r['n']:,} / {r['patients']}</td><td>{fmt(r['event_rate'], True)}</td>" +
                    "".join(f"<td>{fmt(r[m]['auroc'])}</td><td>{fmt(r[m]['sensitivity'], True)}</td><td>{fmt(r[m]['false_alarm_rate'], True)}</td>"
                            for m in MODELS) +
                    f"<td>{fmt_mae(r['Ensemble']['mae60'])}</td></tr>")
    chk = lambda cs: "".join(f"<tr><td>{n}</td><td class='{'ok' if o else 'bad'}'>{'PASS' if o else 'FAIL'}</td><td>{d}</td></tr>" for n, o, d in cs)
    wt = "".join(f"<tr><td>{r['model']}</td><td>{r['base_risk']:.2f} / {r['base_fc120']:.0f}</td><td>{r['meal60_risk']:.2f} / {r['meal60_fc120']:.0f}</td>"
                 f"<td>{r['meal120_risk']:.2f} / {r['meal120_fc120']:.0f}</td><td>{r['meal60_walk_risk']:.2f} / {r['meal60_walk_fc120']:.0f}</td></tr>"
                 for r in b_table)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Scenario Tests</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1200px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}} td,th{{border-bottom:1px solid #e2e8f0;padding:5px 7px;text-align:left}}
.ok{{color:#15803d;font-weight:700}} .bad{{color:#b91c1c;font-weight:700}} .muted{{color:#64748b}} img{{max-width:100%}}</style></head><body>
<h1>Model Scenario Tests &mdash; {passed}/{len(all_checks)} passed</h1>
<p class="muted">Held-out test patients only. Ensemble = average of LightGBM and GRU. Alert thresholds were set on validation
patients to catch 80% of events overall.</p>
<h2>A. Real-life scenarios</h2>
<table><tr><th rowspan=2>Scenario</th><th rowspan=2>Moments / patients</th><th rowspan=2>Event rate</th>
<th colspan=3>LightGBM</th><th colspan=3>GRU</th><th colspan=3>Ensemble</th><th rowspan=2>MAE 60 min</th></tr>
<tr>{'<th>AUROC</th><th>Caught</th><th>False alarms</th>' * 3}</tr>{sc_rows}</table>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{chk(a_checks)}</table>
<h2>B. What-if tests ({n_whatif} T2D moments, glucose 90&ndash;160, no meal for 3 h)</h2>
<p class="muted">Median spike risk / median 2-hour forecast (mg/dL) after changing one input.</p>
<table><tr><th>Model</th><th>As is</th><th>+ 60 g meal now</th><th>+ 120 g meal now</th><th>+ 60 g meal + 30 min walk</th></tr>{wt}</table>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{chk(b_checks)}</table>
<h2>C. Walkthrough: three real test patient-days</h2>
<p class="muted">Grey = actual CGM. Orange dots = the ensemble's forecast made 60 minutes earlier. Triangles = medicine dose taken (green) / missed (red).
Lower panel: risk scores with alert thresholds (dashed).</p>
{''.join(charts)}
</body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")
    for n, o, d in all_checks:
        print(f"  {'PASS' if o else 'FAIL'}  {n}: {d}")
    log(f"{passed}/{len(all_checks)} checks passed. Report -> {OUT / 'report.html'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
