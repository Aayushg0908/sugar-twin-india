"""
Experiment: how much better does the twin predict if the patient ANNOUNCES meals before eating?

The oracle experiment showed that knowing PAST meals perfectly barely helps (AUROC 0.958 -> 0.961):
most of the remaining error comes from meals that have not happened yet. An app can capture this
with a "I'm about to eat" button. We compare three scenarios with identical LightGBM settings:

  baseline       today's features (what the app knows now)
  announce_15m   patient taps "about to eat" with an estimated carb amount up to 15 min before each meal
  meal_plan_2h   patient has entered today's meal plan: every meal in the next 2 h is known

Announced carbs are the patient's ESTIMATE (true carbs x lognormal error, sd 0.3), not the true value.

Models: spike alert, 60-min and 120-min forecast. Evaluated on held-out test patients, overall, for T2D,
and on "pre-meal moments" (a meal starts within 30 min), where announcements matter most.

Usage:
    python src/experiment_meal_announcement.py   # -> results/meal_announcement/{metrics.json, report.html}
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
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import START, STEP, T  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "model" / "features.parquet"
EVENTS = ROOT / "data" / "wearable" / "events.csv"
OUT = ROOT / "results" / "meal_announcement"
ID_COLS = ["patient_id", "timestamp", "split"]
TARGETS = ["y_spike", "y_severe", "y_hypo", "g_p30", "g_p60", "g_p90", "g_p120"]
SCENARIOS = {"baseline": 0, "announce_15m": 3, "meal_plan_2h": 24}   # look-ahead window in 5-min bins
GI_CODE = {"low": 0, "medium": 1, "high": 2}
PARAMS = dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def announced_meal_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Features about meals starting in (t, t + window] that the patient has announced."""
    ev = pd.read_csv(EVENTS, parse_dates=["timestamp"], usecols=["patient_id", "timestamp", "event_type", "carbs_g",
                                                                  "glycaemic_index"])
    meals = ev[ev.event_type == "meal"].copy()
    rng = np.random.default_rng(11)
    meals["est_carbs"] = meals.carbs_g * rng.lognormal(0, 0.3, len(meals))   # patient's guess, not the truth
    pid_code = {p: i for i, p in enumerate(sorted(df.patient_id.unique()))}
    meals = meals[meals.patient_id.isin(pid_code)]
    m_key = meals.patient_id.map(pid_code).to_numpy() * T + ((meals.timestamp - START) / pd.Timedelta(minutes=STEP)).astype(int).to_numpy()
    order = np.argsort(m_key)
    m_key, carbs, gi = m_key[order], meals.est_carbs.to_numpy()[order], meals.glycaemic_index.map(GI_CODE).to_numpy()[order]
    cs = np.concatenate([[0.0], np.cumsum(carbs)])

    r_key = df.patient_id.map(pid_code).to_numpy() * T + ((df.timestamp - START) / pd.Timedelta(minutes=STEP)).astype(int).to_numpy()
    first = np.searchsorted(m_key, r_key, side="right")              # first meal strictly after t
    last = np.searchsorted(m_key, r_key + window, side="right")      # meals up to t + window
    has = last > first
    nxt = np.minimum(first, len(m_key) - 1)
    return pd.DataFrame({
        "ann_min_to_meal": np.where(has, (m_key[nxt] - r_key) * STEP, np.nan),
        "ann_next_meal_carbs": np.where(has, carbs[nxt], np.nan),
        "ann_next_meal_gi": np.where(has, gi[nxt], np.nan),
        "ann_carbs_window": np.where(has, cs[last] - cs[first], 0.0),
    }, index=df.index).astype(np.float32)


def premeal_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows where a (true) meal starts within the next 30 min: the moments announcements are meant for."""
    return announced_meal_features(df, 6)["ann_min_to_meal"].notna().to_numpy()


def train_eval(tr, va, te, feats, target, kind):
    params = dict(PARAMS, objective="binary", metric="auc") if kind == "cls" else dict(PARAMS, objective="huber", alpha=30.0, metric="l1")
    m_tr, m_va = tr[target].notna(), va[target].notna()
    y_tr = tr.loc[m_tr, target] if kind == "cls" else tr.loc[m_tr, target] - tr.loc[m_tr, "g_now"]
    y_va = va.loc[m_va, target] if kind == "cls" else va.loc[m_va, target] - va.loc[m_va, "g_now"]
    dtr = lgb.Dataset(tr.loc[m_tr, feats], y_tr, free_raw_data=True)
    dva = lgb.Dataset(va.loc[m_va, feats], y_va, reference=dtr)
    model = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    m_te = te[target].notna().to_numpy()
    p = model.predict(te.loc[m_te, feats], num_iteration=model.best_iteration)
    if kind == "reg":
        p = te.loc[m_te, "g_now"].to_numpy() + p
    return m_te, p


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(FEATURES)
    base_feats = [c for c in df.columns if c not in ID_COLS + TARGETS]
    te_mask = (df.split == "test").to_numpy()
    premeal = premeal_mask(df)[te_mask]
    t2d = (df.loc[te_mask, "cohort"] == 2).to_numpy()
    log(f"{len(df):,} rows; {premeal.mean():.1%} of test rows are pre-meal moments (meal within 30 min)")

    results = {}
    for scen, window in SCENARIOS.items():
        feats = list(base_feats)
        if window:   # add / overwrite the announcement columns in place (no copy of the big table)
            ann = announced_meal_features(df, window)
            for c in ann.columns:
                df[c] = ann[c]
            feats += list(ann.columns)
            del ann
        tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
        res = {}
        log(f"Scenario '{scen}': training spike model")
        m, p = train_eval(tr, va, te, feats, "y_spike", "cls")
        y = te.loc[m, "y_spike"].to_numpy()
        pm, tm = premeal[m], t2d[m]
        res["spike"] = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                        "auroc_t2d": roc_auc_score(y[tm], p[tm]), "auroc_premeal": roc_auc_score(y[pm], p[pm]),
                        "auprc_premeal": average_precision_score(y[pm], p[pm])}
        for target in ("g_p60", "g_p120"):
            log(f"Scenario '{scen}': training {target} forecaster")
            m, p = train_eval(tr, va, te, feats, target, "reg")
            err = np.abs(p - te.loc[m, target].to_numpy())
            pm, tm = premeal[m], t2d[m]
            res[target] = {"mae": float(err.mean()), "mae_t2d": float(err[tm].mean()), "mae_premeal": float(err[pm].mean()),
                           "mae_premeal_t2d": float(err[pm & tm].mean())}
        results[scen] = res
        log(f"  {scen}: spike AUROC {res['spike']['auroc']:.3f} (pre-meal {res['spike']['auroc_premeal']:.3f}) | "
            f"MAE60 {res['g_p60']['mae']:.1f} (pre-meal {res['g_p60']['mae_premeal']:.1f}) | "
            f"MAE120 {res['g_p120']['mae']:.1f} (pre-meal {res['g_p120']['mae_premeal']:.1f})")
        del tr, va, te

    results["_meta"] = {"premeal_share_of_test_rows": float(premeal.mean()), "minutes": round((time.time() - t0) / 60, 1)}
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    write_report(results)
    log(f"Done in {results['_meta']['minutes']} min -> {OUT}")


def write_report(r):
    scen = list(SCENARIOS)
    labels = {"baseline": "Today (no announcement)", "announce_15m": "Announce 15 min before eating",
              "meal_plan_2h": "Meal plan for next 2 h"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(scen))
    ax[0].bar(x - 0.2, [r[s]["spike"]["auroc"] for s in scen], 0.4, label="all moments", color="#94a3b8")
    ax[0].bar(x + 0.2, [r[s]["spike"]["auroc_premeal"] for s in scen], 0.4, label="pre-meal moments", color="#c2410c")
    ax[0].set(xticks=x, xticklabels=[labels[s] for s in scen], ylim=(0.8, 1.0), title="Spike alert AUROC")
    ax[0].legend()
    ax[1].bar(x - 0.2, [r[s]["g_p60"]["mae_premeal"] for s in scen], 0.4, label="60-min forecast", color="#2563eb")
    ax[1].bar(x + 0.2, [r[s]["g_p120"]["mae_premeal"] for s in scen], 0.4, label="120-min forecast", color="#1e3a8a")
    ax[1].set(xticks=x, xticklabels=[labels[s] for s in scen], title="Forecast error at pre-meal moments (mg/dL, lower is better)")
    ax[1].legend()
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    chart = base64.b64encode(buf.getvalue()).decode()
    rows = "".join(
        f"<tr><td>{labels[s]}</td><td>{r[s]['spike']['auroc']:.3f}</td><td>{r[s]['spike']['auroc_t2d']:.3f}</td>"
        f"<td>{r[s]['spike']['auroc_premeal']:.3f}</td><td>{r[s]['g_p60']['mae_premeal']:.1f}</td>"
        f"<td>{r[s]['g_p120']['mae_premeal']:.1f}</td><td>{r[s]['g_p120']['mae_premeal_t2d']:.1f}</td></tr>" for s in scen)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Meal Announcement Experiment</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.muted{{color:#64748b}} img{{max-width:100%}}</style></head><body>
<h1>What if patients announce meals before eating?</h1>
<p class="muted">LightGBM, identical settings per scenario, held-out test patients. Announced carbs are the patient's
estimate (&plusmn;30% error). Pre-meal moments = a meal starts within 30 min ({r['_meta']['premeal_share_of_test_rows']:.0%} of test rows).</p>
<table><tr><th>Scenario</th><th>Spike AUROC</th><th>Spike AUROC (T2D)</th><th>Spike AUROC (pre-meal)</th>
<th>MAE 60 min (pre-meal)</th><th>MAE 120 min (pre-meal)</th><th>MAE 120 min (pre-meal, T2D)</th></tr>{rows}</table>
<img src="data:image/png;base64,{chart}"></body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
