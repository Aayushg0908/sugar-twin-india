"""
Personalized physiology twin: answers WHAT-IF questions ("what if this patient eats 60 g more /
walks 20 min / skips metformin?") that the ML models cannot answer reliably (they learn correlation,
not cause and effect; see results/scenario_tests).

Physiology (same structure as the data simulator, 5-min steps):
    meal carbs -> 2-compartment gut absorption (speed by glycaemic index)
    excursion X:  dX/dt = -X / tau * (1 + w * exercise) + c * Ra(t) - 0.4 * w * exercise
Each patient has their own three knobs, LEARNED FROM THEIR OWN OBSERVABLE DATA (logged meals,
logged exercise and CGM from week 1). Hidden simulator truth is used only to score the fit afterwards.
    p10   peak glucose rise per 10 g of medium-GI carbs (mg/dL)        "carb sensitivity"
    tau   clearance time constant (min)                                 "recovery time"
    w     exercise effect multiplier (1 = average)                      "walk effect"

Fitting (per patient, week 1): for every clean logged meal window (-30..+180 min, no other logged meal
or exercise inside), model CGM as  a + b*t + p10 * R(t; tau, GI, carbs)  where R is the unit response.
For a grid of tau values the problem is linear in (p10, a_i, b_i), solved by least squares with
outlier windows (unlogged snacks etc.) removed; the tau with the lowest error wins. w is then fitted
on meal windows followed by a logged walk; patients with fewer than 2 such windows get the population w.

Validation:
  1. Week 2: predict the glucose rise after each new logged meal from the pre-meal value.
     Personalized twin vs one-size-fits-all population parameters.
  2. Recovery: fitted knobs vs the hidden true physiology of each synthetic patient.

What-if (used by the dashboard):
    twin = PatientTwin.load("P00042")
    delta = twin.whatif_delta(history_events, changes=[("meal", 0, 60, "medium"), ("walk", 30, 20)])
    what_if_curve = ml_forecast + delta

Usage:
    python src/twin_simulator.py   # fit all patients -> models/twin/twin_params_fitted.csv,
                                   # report -> results/twin_simulator/report.html
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_wearable import DRUG_EFFECT, GI_KABS, STEP_MIN, unit_peak  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data" / "wearable"
EHR = ROOT / "data" / "ehr"
PARAMS_OUT = ROOT / "models" / "twin" / "twin_params_fitted.csv"
OUT = ROOT / "results" / "twin_simulator"

START = pd.Timestamp("2026-09-17")
FIT_DAYS = 7                              # week 1 = learn, week 2 = validate
PRE, POST = 6, 36                         # window: 30 min before to 180 min after a meal (5-min steps)
GI = {"high": 0, "medium": 1, "low": 2}   # index into GI_KABS
EX_INTENSITY = {"Yoga": 0.6, "Walk": 1.0, "Brisk walk / gym": 1.8}
TAU_GRID = np.arange(25, 181, 5, dtype=float)
K_GRID = [0, 2, 4, 8, 16, 32]            # shrinkage strength (pseudo-meals of prior evidence)
P10_BOUNDS = (1.0, 25.0)
W_GRID = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])


# --------------------------------------------------------------------------
# Physiology
# --------------------------------------------------------------------------
_PEAK_CACHE: dict[float, float] = {}


def _unit_peak(tau: float) -> float:
    if tau not in _PEAK_CACHE:
        _PEAK_CACHE[tau] = float(unit_peak(np.array([tau]))[0])
    return _PEAK_CACHE[tau]


def simulate(meals, exercise, n_steps, tau, p10, w=1.0, x0=0.0, q0=None):
    """Glucose excursion (mg/dL above baseline) on a 5-min grid.
    meals: list of (step, carbs_g, gi_index); exercise: array (n_steps,) of intensities."""
    c = p10 / _unit_peak(float(tau))
    q1 = np.zeros(3) if q0 is None else q0[0].copy()
    q2 = np.zeros(3) if q0 is None else q0[1].copy()
    meal_in = {}
    for s, carbs, gi in meals:
        meal_in.setdefault(int(s), []).append((carbs, gi))
    x, out = x0, np.empty(n_steps)
    for t in range(n_steps):
        for carbs, gi in meal_in.get(t, ()):
            q1[gi] += carbs
        ra = (GI_KABS * q2).sum()
        q1, q2 = q1 - STEP_MIN * GI_KABS * q1, q2 + STEP_MIN * GI_KABS * (q1 - q2)
        e = w * exercise[t]
        x = max(x + STEP_MIN * (-x / tau * (1 + e) + c * ra - 0.4 * e), -40.0)
        out[t] = x
    return out


def unit_responses():
    """R(t) for 10 g carbs eaten at step PRE, for every (tau, GI), scaled so p10 multiplies it directly."""
    n = PRE + POST + 1
    resp = np.zeros((len(TAU_GRID), 3, n))
    zero = np.zeros(n)
    for i, tau in enumerate(TAU_GRID):
        for gi in range(3):
            resp[i, gi] = simulate([(PRE, 10.0, gi)], zero, n, tau, p10=1.0)
    return resp


UNIT = unit_responses()


# --------------------------------------------------------------------------
# Data helpers
# --------------------------------------------------------------------------
def load_observed():
    """CGM matrix + LOGGED events only (what a real app would have)."""
    cgm = pd.read_parquet(W / "cgm.parquet")
    pids = cgm.patient_id.drop_duplicates().tolist()
    T = len(cgm) // len(pids)
    G = cgm.glucose_mgdl.to_numpy(dtype=float, na_value=np.nan).reshape(len(pids), T)
    ev = pd.read_csv(W / "events.csv", parse_dates=["timestamp"],
                     usecols=["patient_id", "timestamp", "event_type", "detail", "logged", "logged_carbs_g", "glycaemic_index",
                              "duration_min", "carbs_g"])
    ev["step"] = ((ev.timestamp - START) / pd.Timedelta(minutes=STEP_MIN)).astype(int)
    ev = ev[ev.logged.astype(bool)]
    return pids, G, ev


def meal_windows(g, meals, exer, day_from, day_to):
    """Clean windows around logged meals: no other logged meal in [-3 h, +3 h], no logged exercise inside."""
    out = []
    steps_meal = meals.step.to_numpy()
    for m in meals.itertuples():
        s = m.step
        if not (day_from * 288 <= s < day_to * 288) or s - PRE < 0 or s + POST >= len(g):
            continue
        if ((np.abs(steps_meal - s) <= 36) & (steps_meal != s)).any():
            continue
        if not (m.logged_carbs_g >= 20):
            continue
        ex_mask = np.zeros(PRE + POST + 1)
        for e in exer.itertuples():
            lo, hi = e.step - (s - PRE), e.step - (s - PRE) + int(e.duration_min // STEP_MIN)
            if hi > 0 and lo < len(ex_mask):
                ex_mask[max(lo, 0):min(hi, len(ex_mask))] = EX_INTENSITY.get(e.detail, 1.0)
        y = g[s - PRE: s + POST + 1]
        if np.isnan(y).mean() > 0.2 or np.isnan(y[:PRE + 1]).all():
            continue
        out.append({"step": s, "carbs": float(m.logged_carbs_g), "gi": GI[m.glycaemic_index], "y": y, "ex": ex_mask,
                    "true_carbs": float(m.carbs_g)})
    return out


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_p10_tau(wins):
    """Grid over tau; for each, linear least squares in p10 + per-window intercept and slope."""
    n = PRE + POST + 1
    tt = (np.arange(n) - PRE) / 36.0
    best = None
    for i, tau in enumerate(TAU_GRID):
        rows, ys, k = [], [], len(wins)
        for j, wd in enumerate(wins):
            ok = ~np.isnan(wd["y"])
            r = UNIT[i, wd["gi"]] * wd["carbs"] / 10.0
            X = np.zeros((ok.sum(), 1 + 2 * k))
            X[:, 0] = r[ok]
            X[:, 1 + 2 * j] = 1.0
            X[:, 2 + 2 * j] = tt[ok]
            rows.append(X)
            ys.append(wd["y"][ok])
        X, y = np.vstack(rows), np.concatenate(ys)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        p10 = float(np.clip(coef[0], *P10_BOUNDS))
        sse = float(((y - X @ np.r_[p10, coef[1:]]) ** 2).mean())
        if best is None or sse < best[2]:
            best = (tau, p10, sse)
    return best


def fit_patient(wins):
    """Fit, drop badly explained windows (likely unlogged food / stress), refit."""
    if len(wins) < 3:
        return None
    tau, p10, _ = fit_p10_tau(wins)
    i = int(np.where(TAU_GRID == tau)[0][0])
    errs = []
    for wd in wins:
        r = UNIT[i, wd["gi"]] * wd["carbs"] / 10.0 * p10
        ok = ~np.isnan(wd["y"])
        A = np.c_[np.ones(ok.sum()), (np.arange(len(r))[ok] - PRE) / 36.0]
        resid = wd["y"][ok] - r[ok]
        coef, *_ = np.linalg.lstsq(A, resid, rcond=None)
        errs.append(float(np.sqrt(((resid - A @ coef) ** 2).mean())))
    errs = np.array(errs)
    keep = [wd for wd, e in zip(wins, errs) if e <= 2 * np.median(errs)]
    if len(keep) >= 3:
        tau, p10, _ = fit_p10_tau(keep)
    return {"tau": tau, "p10": p10, "n_meals": len(keep), "n_meals_raw": len(wins), "fit_rmse": float(np.median(errs))}


def fit_walk(wins_ex, tau, p10):
    """Grid search for the exercise multiplier on meal windows that contain a logged walk."""
    best_w, best_sse = None, np.inf
    for w in W_GRID:
        sse = 0.0
        for wd in wins_ex:
            pred = simulate([(PRE, wd["carbs"], wd["gi"])], wd["ex"], PRE + POST + 1, tau, p10, w)
            ok = ~np.isnan(wd["y"])
            A = np.c_[np.ones(ok.sum()), (np.arange(len(pred))[ok] - PRE) / 36.0]
            coef, *_ = np.linalg.lstsq(A, wd["y"][ok] - pred[ok], rcond=None)
            sse += float(((wd["y"][ok] - pred[ok] - A @ coef) ** 2).sum())
        if sse < best_sse:
            best_w, best_sse = w, sse
    return float(best_w)


# --------------------------------------------------------------------------
# What-if interface (used by the dashboard)
# --------------------------------------------------------------------------
class PatientTwin:
    """A patient's fitted physiology. All times are in minutes relative to 'now'."""

    def __init__(self, patient_id, tau, p10, w, regimen=()):
        self.patient_id, self.tau, self.p10, self.w = patient_id, tau, p10, w
        self.regimen = list(regimen)          # [(drug, drug_class)] currently prescribed

    @classmethod
    def load(cls, patient_id):
        p = pd.read_csv(PARAMS_OUT).set_index("patient_id").loc[patient_id]
        meds = pd.read_csv(EHR / "medications.csv")
        act = meds[(meds.patient_id == patient_id) & meds.end_date.isna() & (meds.indication == "Type 2 diabetes")]
        return cls(patient_id, p.tau, p.p10, p.w, list(zip(act.drug, act.drug_class)))

    def whatif_delta(self, recent_meals=(), recent_exercise=(), changes=(), horizon_min=240):
        """Change in glucose (mg/dL) over the next `horizon_min` minutes caused by `changes`.
        recent_meals / recent_exercise: what already happened, as (minutes_ago, carbs, gi_name) / (minutes_ago, duration, type)
        changes: ("meal", in_minutes, carbs, gi_name) | ("walk", in_minutes, duration_min) | ("skip_drug", drug_class)"""
        back = 48                                                     # simulate the last 4 h so digestion in progress counts
        n = back + horizon_min // STEP_MIN + 1
        base_meals = [(back - m // STEP_MIN, c, GI[g]) for m, c, g in recent_meals if m // STEP_MIN <= back]
        ex = np.zeros(n)
        for m, d, kind in recent_exercise:
            s = back - m // STEP_MIN
            ex[max(s, 0):max(s + d // STEP_MIN, 0)] = EX_INTENSITY.get(kind, 1.0)
        new_meals, new_ex, offset = list(base_meals), ex.copy(), 0.0
        for ch in changes:
            if ch[0] == "meal":
                new_meals.append((back + ch[1] // STEP_MIN, float(ch[2]), GI[ch[3]]))
            elif ch[0] == "walk":
                s = back + ch[1] // STEP_MIN
                new_ex[s:s + ch[2] // STEP_MIN] = EX_INTENSITY["Walk"]
            elif ch[0] == "skip_drug":
                # pharmacology prior (same table as the data simulator): missing a dose raises the day's baseline
                offset += DRUG_EFFECT.get(ch[1], 0.0) * 28.7 * 0.6
        base = simulate(base_meals, ex, n, self.tau, self.p10, self.w)
        alt = simulate(new_meals, new_ex, n, self.tau, self.p10, self.w)
        ramp = np.clip(np.arange(n - back) * STEP_MIN / 120, 0, 1)    # missed-dose effect builds up over ~2 h
        return (alt - base)[back:] + offset * ramp

    def describe(self):
        speed = "slow" if self.tau > 70 else "moderate" if self.tau > 45 else "fast"
        return (f"10 g carbs raise glucose by ~{self.p10:.1f} mg/dL at peak; recovery is {speed} (tau {self.tau:.0f} min); "
                f"walking effect {self.w:.2f}x average")


# --------------------------------------------------------------------------
# Main: fit everyone, validate, report
# --------------------------------------------------------------------------
def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def peak_rise(y):
    pre = np.nanmean(y[PRE - 2:PRE + 1])
    return float(np.nanmax(y[PRE:]) - pre), pre


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PARAMS_OUT.parent.mkdir(parents=True, exist_ok=True)
    pids, G, ev = load_observed()
    meals_all = ev[ev.event_type == "meal"]
    ex_all = ev[ev.event_type == "exercise"]
    meals_by = dict(tuple(meals_all.groupby("patient_id")))
    ex_by = dict(tuple(ex_all.groupby("patient_id")))
    empty = meals_all.iloc[:0]

    rows, walk_fits, val_cache = [], {}, {}
    for k, pid in enumerate(pids):
        m, e = meals_by.get(pid, empty), ex_by.get(pid, ex_all.iloc[:0])
        wins = meal_windows(G[k], m, e, 0, FIT_DAYS)
        clean = [wd for wd in wins if wd["ex"].sum() == 0]
        fit = fit_patient(clean)
        if fit is None:
            continue
        ex_wins = [wd for wd in wins if wd["ex"].sum() > 0]
        if len(ex_wins) >= 2:
            walk_fits[pid] = fit_walk(ex_wins, fit["tau"], fit["p10"])
        rows.append({"patient_id": pid, **fit, "n_walk_meals": len(ex_wins)})
        val_cache[pid] = meal_windows(G[k], m, e, FIT_DAYS, 14)
        if k % 500 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] fitted {k}/{len(pids)} patients", flush=True)

    # ---- Shrinkage: blend each patient's own estimate with their cohort's typical value.
    # Few or noisy meals -> trust the cohort prior more; many meals -> trust the patient's own data.
    fit = pd.DataFrame(rows).rename(columns={"tau": "tau_raw", "p10": "p10_raw"})
    fit["w_raw"] = fit.patient_id.map(walk_fits)
    meta = pd.read_csv(EHR / "patients.csv").set_index("patient_id")
    split = pd.read_csv(ROOT / "data" / "splits.csv").set_index("patient_id").split
    fit["cohort"], fit["split"] = fit.patient_id.map(meta.cohort), fit.patient_id.map(split)
    train = fit[fit.split == "train"]
    prior = train.groupby("cohort")[["tau_raw", "p10_raw"]].median()          # priors from TRAIN patients only
    pop_w = float(train.w_raw.median()) if train.w_raw.notna().any() else 1.0
    fit["tau_prior"], fit["p10_prior"] = fit.cohort.map(prior.tau_raw), fit.cohort.map(prior.p10_raw)
    pop_tau, pop_p10 = float(train.tau_raw.median()), float(train.p10_raw.median())

    def shrink(k):
        a = fit.n_meals / (fit.n_meals + k)
        aw = np.where(fit.w_raw.notna(), fit.n_walk_meals / (fit.n_walk_meals + k), 0.0)
        return pd.DataFrame({"patient_id": fit.patient_id,
                             "tau": a * fit.tau_raw + (1 - a) * fit.tau_prior,
                             "p10": a * fit.p10_raw + (1 - a) * fit.p10_prior,
                             "w": aw * fit.w_raw.fillna(pop_w) + (1 - aw) * pop_w}).set_index("patient_id")

    windows = [(pid, wd) for pid, wins in val_cache.items() for wd in wins]
    n = PRE + POST + 1

    def curve(wd, tau, p10, w):
        if wd["ex"].sum() == 0:                                     # no exercise: linear -> use precomputed responses
            i = int(np.abs(TAU_GRID - tau).argmin())
            return p10 * wd["carbs"] / 10.0 * UNIT[i, wd["gi"]]
        return simulate([(PRE, wd["carbs"], wd["gi"])], wd["ex"], n, tau, p10, w)

    def evaluate(params, ids=None):
        out = []
        for pid, wd in windows:
            if ids is not None and pid not in ids:
                continue
            actual, pre = peak_rise(wd["y"])
            if not np.isfinite(actual):
                continue
            prm = params if isinstance(params, tuple) else tuple(params.loc[pid, ["tau", "p10", "w"]])
            c = curve(wd, *prm)
            ok = ~np.isnan(wd["y"])
            out.append({"patient_id": pid, "cohort": meta.cohort[pid], "actual_rise": actual, "pred_rise": float(c.max()),
                        "peak_err": abs(float(c.max()) - actual), "curve_mae": float(np.abs(wd["y"][ok] - pre - c[ok]).mean())})
        return pd.DataFrame(out)

    train_ids, test_ids = set(train.patient_id), set(fit.patient_id[fit.split == "test"])
    k_scores = {}
    for k in K_GRID:                                               # pick k on TRAIN patients' week 2
        k_scores[k] = float(evaluate(shrink(k), train_ids).peak_err.mean())
        print(f"[{time.strftime('%H:%M:%S')}] shrinkage k={k}: train week-2 peak error {k_scores[k]:.2f}", flush=True)
    best_k = min(k_scores, key=k_scores.get)
    final = shrink(best_k)
    fit["tau"], fit["p10"], fit["w"] = fit.patient_id.map(final.tau), fit.patient_id.map(final.p10), fit.patient_id.map(final.w)
    fit["w_source"] = np.where(fit.w_raw.notna(), "own data (shrunk)", "population")
    fit["shrinkage_k"] = best_k
    fit.round(3).to_csv(PARAMS_OUT, index=False)
    print(f"Fitted {len(fit)} / {len(pids)} patients. Best shrinkage k={best_k}. Population: tau {pop_tau:.0f}, p10 {pop_p10:.1f}, w {pop_w:.2f}")

    # ---- Validation 1: week-2 meals of TEST patients
    variants = {"population": (pop_tau, pop_p10, pop_w),
                "cohort_average": shrink(1e9),
                "personal_raw": shrink(0),
                "personal_shrunk": final}
    evals = {name: evaluate(prm, test_ids) for name, prm in variants.items()}
    val = evals["personal_shrunk"].rename(columns={"peak_err": "pers_err", "curve_mae": "pers_curve_mae", "pred_rise": "pers_rise"})
    val["pop_err"] = evals["population"].peak_err.to_numpy()
    val["pop_curve_mae"] = evals["population"].curve_mae.to_numpy()
    val["pop_rise"] = evals["population"].pred_rise.to_numpy()

    # ---- Validation 2: recovery of hidden physiology (truth used ONLY for scoring)
    tp = pd.read_csv(EHR / "twin_params.csv").set_index("patient_id")
    true_tau = 35 / (tp.insulin_sensitivity * tp.beta_cell_function) ** 0.35
    true_p10 = np.minimum(tp.carb_sensitivity_mgdl_per_10g * 1.5, 15.0)
    fit["true_tau"], fit["true_p10"] = fit.patient_id.map(true_tau), fit.patient_id.map(true_p10)
    r_tau = float(np.corrcoef(fit.tau, fit.true_tau)[0, 1])
    r_p10 = float(np.corrcoef(fit.p10, fit.true_p10)[0, 1])

    summary = {"patients_fitted": int(len(fit)), "patients_total": len(pids),
               "population": {"tau": pop_tau, "p10": pop_p10, "w": pop_w},
               "cohort_priors": prior.round(2).to_dict("index"),
               "shrinkage": {"k_scores_train": k_scores, "best_k": best_k},
               "week2_test_comparison": {name: {"peak_rise_mae": float(e.peak_err.mean()), "curve_mae": float(e.curve_mae.mean()),
                                                "peak_rise_mae_by_cohort": e.groupby("cohort").peak_err.mean().round(2).to_dict()}
                                         for name, e in evals.items()},
               "walk_fitted_from_own_data": int(len(walk_fits)),
               "week2_meals_test_patients": int(len(val)),
               "week2_peak_rise_mae": {"personalized": float(val.pers_err.mean()), "population": float(val.pop_err.mean())},
               "week2_curve_mae": {"personalized": float(val.pers_curve_mae.mean()), "population": float(val.pop_curve_mae.mean())},
               "week2_peak_rise_mae_by_cohort": val.groupby("cohort")[["pers_err", "pop_err"]].mean().round(1).to_dict("index"),
               "recovery_correlation": {"tau": r_tau, "p10": r_p10},
               "minutes": round((time.time() - t0) / 60, 1)}
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    write_report(summary, fit, val, pids, G, meals_by, ex_by)
    print(json.dumps(summary, indent=2, default=float))


LABELS = {"population": "One-size-fits-all", "cohort_average": "Cohort average (from EHR diagnosis)",
          "personal_raw": "Personal, own data only", "personal_shrunk": "Personalized twin (blended)"}


def write_report(s, fit, val, pids, G, meals_by, ex_by):
    charts = []
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].scatter(fit.true_p10, fit.p10, s=4, alpha=0.3, color="#c2410c")
    ax[0].plot([0, 16], [0, 16], "k--", lw=0.7)
    ax[0].set(title=f"Carb sensitivity: learned vs hidden truth (r={s['recovery_correlation']['p10']:.2f})",
              xlabel="true mg/dL per 10 g", ylabel="learned from week 1")
    ax[1].scatter(fit.true_tau, fit.tau, s=4, alpha=0.3, color="#2563eb")
    ax[1].plot([20, 180], [20, 180], "k--", lw=0.7)
    ax[1].set(title=f"Recovery time: learned vs hidden truth (r={s['recovery_correlation']['tau']:.2f})",
              xlabel="true tau (min)", ylabel="learned tau (min)")
    vals = [s["week2_peak_rise_mae"]["population"], s["week2_peak_rise_mae"]["personalized"]]
    ax[2].bar(["One-size-fits-all", "Personalized twin"], vals, color=["#94a3b8", "#15803d"])
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.5, f"{v:.1f}", ha="center")
    ax[2].set(title="Week 2: error in predicted post-meal peak (mg/dL)")
    charts.append(img(fig))

    # example: one T2D patient's week-2 meals, actual vs predicted
    example = val.groupby("patient_id").size().loc[lambda x: x >= 4].index
    ex_pid = next((p for p in example if val[val.patient_id == p].cohort.iloc[0] == "T2D"), example[0] if len(example) else None)
    if ex_pid is not None:
        k = pids.index(ex_pid)
        prm = fit.set_index("patient_id").loc[ex_pid]
        wins = meal_windows(G[k], meals_by[ex_pid], ex_by.get(ex_pid, meals_by[ex_pid].iloc[:0]), FIT_DAYS, 14)[:4]
        fig, axes = plt.subplots(1, len(wins), figsize=(4 * len(wins), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        tmin = (np.arange(PRE + POST + 1) - PRE) * STEP_MIN
        for a, wd in zip(axes, wins):
            _, pre = peak_rise(wd["y"])
            n = PRE + POST + 1
            a.plot(tmin, wd["y"], color="#334155", label="actual CGM")
            a.plot(tmin, pre + simulate([(PRE, wd["carbs"], wd["gi"])], wd["ex"], n, prm.tau, prm.p10, prm.w), color="#15803d", label="personalized twin")
            a.plot(tmin, pre + simulate([(PRE, wd["carbs"], wd["gi"])], wd["ex"], n, s["population"]["tau"], s["population"]["p10"], s["population"]["w"]),
                   color="#94a3b8", ls="--", label="population average")
            a.set(title=f"{int(wd['carbs'])} g logged ({['high', 'medium', 'low'][wd['gi']]} GI)", xlabel="minutes from meal")
        axes[0].set_ylabel("mg/dL")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{ex_pid}: unseen week-2 meals ({PatientTwin(ex_pid, prm.tau, prm.p10, prm.w).describe()})", fontsize=9)
        charts.append(img(fig))

    coh = "".join(f"<tr><td>{c}</td><td>{v['pop_err']:.1f}</td><td>{v['pers_err']:.1f}</td></tr>"
                  for c, v in s["week2_peak_rise_mae_by_cohort"].items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Personalized Twin Simulator</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.muted{{color:#64748b}} img{{max-width:100%}}</style></head><body>
<h1>Personalized Physiology Twin</h1>
<p class="muted">Each patient's carb sensitivity, recovery time and walking effect are learned from week 1
(logged meals + CGM only) and tested on unseen week-2 meals. {s['patients_fitted']:,} of {s['patients_total']:,} patients had
enough clean logged meals to fit; walking effect learned from own data for {s['walk_fitted_from_own_data']:,}, population value otherwise.</p>
<h2>Week 2: predicting the glucose rise after a new meal ({s['week2_meals_test_patients']:,} meals of test patients)</h2>
<p class="muted">Each patient's own estimate is blended with their cohort's typical value; the blend strength
(k = {s['shrinkage']['best_k']} pseudo-meals) was chosen on training patients only.</p>
<table><tr><th>Approach</th><th>Peak-rise error</th><th>Whole-curve error</th><th>Normoglycemic</th><th>Prediabetes</th><th>T2D</th></tr>
{''.join(f"<tr><td>{LABELS[k]}</td><td>{v['peak_rise_mae']:.1f}</td><td>{v['curve_mae']:.1f}</td>" + "".join(f"<td>{v['peak_rise_mae_by_cohort'].get(c, float('nan')):.1f}</td>" for c in ('Normoglycemic', 'Prediabetes', 'T2D')) + "</tr>" for k, v in s['week2_test_comparison'].items())}</table>
<table><tr><th></th><th>One-size-fits-all</th><th>Personalized twin</th></tr>
<tr><td>Peak-rise error (mg/dL)</td><td>{s['week2_peak_rise_mae']['population']:.1f}</td><td>{s['week2_peak_rise_mae']['personalized']:.1f}</td></tr>
<tr><td>Whole-curve error (mg/dL)</td><td>{s['week2_curve_mae']['population']:.1f}</td><td>{s['week2_curve_mae']['personalized']:.1f}</td></tr></table>
<table><tr><th>Cohort</th><th>Peak-rise error, population</th><th>Peak-rise error, personalized</th></tr>{coh}</table>
<h2>Does the twin recover each patient's hidden physiology?</h2>
<p>Correlation between learned and true values: carb sensitivity <b>r = {s['recovery_correlation']['p10']:.2f}</b>,
recovery time <b>r = {s['recovery_correlation']['tau']:.2f}</b>.</p>
{''.join(charts)}</body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
