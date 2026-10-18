"""
Experiment: what does each data stream add? (fusion ablation)

The challenge asks for a model that FUSES static EHR data with dynamic wearable data. This trains the same
LightGBM models on growing feature sets so the contribution of each stream is measured, not assumed:

  cgm_only          CGM history + time of day                         (what a CGM app alone knows)
  cgm_wearable      + smartwatch (HR, HRV, steps, sleep, exercise) + app logs (meals, doses)
  cgm_ehr           CGM + time + EHR (labs, diagnoses, medicines, genetics, demographics)
  full_fusion       everything (the feature set used by the main models)

Models: spike alert, hypo alert, 60-min forecast. Same settings for every variant (faster than the main
training: learning rate 0.1), trained on train patients, early-stopped on validation patients, scored on
held-out test patients, overall and for T2D only.

Usage:
    python src/experiment_fusion_ablation.py   # -> results/fusion_ablation/{metrics.json, report.html}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import EHR_CATEG, EHR_FLAGS, EHR_NUMERIC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "model" / "features.parquet"
OUT = ROOT / "results" / "fusion_ablation"
ID_COLS = ["patient_id", "timestamp", "split"]
TARGETS = ["y_spike", "y_severe", "y_hypo", "g_p30", "g_p60", "g_p90", "g_p120"]
PARAMS = dict(learning_rate=0.1, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)

# Feature groups (names from build_features.py)
TIME = ["hour", "hour_sin", "hour_cos", "weekend"]
SMARTWATCH = ["hr_15m", "hr_1h", "steps_30m", "steps_1h", "steps_3h", "asleep_now", "min_since_sleep",
              "sleep_hours_last_night", "hrv_last_night", "sleep_eff_last_night", "resting_hr_last_night",
              "steps_yesterday", "hr_vs_resting", "min_since_exercise", "exercise_min_3h"]
APP_LOGS = ["min_since_meal", "last_meal_carbs", "last_meal_gi", "carbs_1h", "carbs_3h", "carbs_today",
            "min_since_dose_logged", "doses_logged_today"]
EHR = EHR_NUMERIC + EHR_FLAGS + list(EHR_CATEG) + ["prescribed_doses_per_day"]
LABELS = {"cgm_only": "CGM only", "cgm_wearable": "CGM + smartwatch + app logs", "cgm_ehr": "CGM + EHR",
          "full_fusion": "Full fusion (CGM + smartwatch + app logs + EHR)"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    all_feats = [c for c in df.columns if c not in ID_COLS + TARGETS]
    cgm = [c for c in all_feats if c not in TIME + SMARTWATCH + APP_LOGS + EHR]
    assert set(TIME + SMARTWATCH + APP_LOGS + EHR) <= set(all_feats), "feature group names out of date"
    variants = {"cgm_only": cgm + TIME, "cgm_wearable": cgm + TIME + SMARTWATCH + APP_LOGS,
                "cgm_ehr": cgm + TIME + EHR, "full_fusion": all_feats}
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    del df
    t2d = (te.cohort == 2).to_numpy()
    log(f"train {len(tr):,} / val {len(va):,} / test {len(te):,} rows; CGM features: {len(cgm)}")

    results = {"feature_groups": {"cgm": cgm, "time": TIME, "smartwatch": SMARTWATCH, "app_logs": APP_LOGS, "ehr": EHR}}
    for name, feats in variants.items():
        res = {"n_features": len(feats)}
        for target, key in (("y_spike", "spike"), ("y_hypo", "hypo")):
            log(f"{name}: training {key} alert ({len(feats)} features)")
            m, p = train_eval(tr, va, te, feats, target, "cls")
            y, tm = te.loc[m, target].to_numpy(), t2d[m]
            res[key] = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                        "auroc_t2d": roc_auc_score(y[tm], p[tm]), "auprc_t2d": average_precision_score(y[tm], p[tm])}
        log(f"{name}: training 60-min forecast")
        m, p = train_eval(tr, va, te, feats, "g_p60", "reg")
        err = np.abs(p - te.loc[m, "g_p60"].to_numpy())
        res["fc60"] = {"mae": float(err.mean()), "mae_t2d": float(err[t2d[m]].mean())}
        results[name] = res
        log(f"  {name}: spike AUROC {res['spike']['auroc']:.3f} (T2D {res['spike']['auroc_t2d']:.3f}) | "
            f"hypo AUPRC {res['hypo']['auprc']:.3f} | MAE60 {res['fc60']['mae']:.1f}")

    results["minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    write_report(results, variants)
    log(f"Done in {results['minutes']} min -> {OUT}")


def write_report(r, variants):
    base = r["cgm_only"]
    def delta(v, b, better_high=True):
        d = v - b
        good = d > 0 if better_high else d < 0
        return f'<span style="color:{"#15803d" if good else "#b91c1c"}">{"+" if d > 0 else ""}{d:.3f}</span>' if abs(d) >= 0.0005 else ""
    rows = "".join(
        f"<tr><td>{LABELS[k]}</td><td>{r[k]['n_features']}</td>"
        f"<td>{r[k]['spike']['auroc']:.3f} {delta(r[k]['spike']['auroc'], base['spike']['auroc'])}</td>"
        f"<td>{r[k]['spike']['auroc_t2d']:.3f} {delta(r[k]['spike']['auroc_t2d'], base['spike']['auroc_t2d'])}</td>"
        f"<td>{r[k]['hypo']['auroc']:.3f}</td>"
        f"<td>{r[k]['hypo']['auprc']:.3f} {delta(r[k]['hypo']['auprc'], base['hypo']['auprc'])}</td>"
        f"<td>{r[k]['fc60']['mae']:.2f} {delta(r[k]['fc60']['mae'], base['fc60']['mae'], better_high=False)}</td>"
        f"<td>{r[k]['fc60']['mae_t2d']:.2f}</td></tr>" for k in variants)
    groups = "".join(f"<tr><td>{g}</td><td>{len(f)}</td><td>{', '.join(f)}</td></tr>" for g, f in r["feature_groups"].items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Fusion Ablation</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left;vertical-align:top}}
.muted{{color:#64748b}}</style></head><body>
<h1>What does each data stream add?</h1>
<p class="muted">Same LightGBM settings for every row (learning rate 0.1, early stopping on validation patients).
All numbers on held-out test patients. Coloured numbers are the change against CGM only.</p>
<table><tr><th>Inputs</th><th>Features</th><th>Spike AUROC</th><th>Spike AUROC (T2D)</th><th>Hypo AUROC</th><th>Hypo AUPRC</th>
<th>60-min MAE (mg/dL)</th><th>60-min MAE, T2D</th></tr>{rows}</table>
<h2>Feature groups</h2><table><tr><th>Group</th><th>#</th><th>Features</th></tr>{groups}</table>
<p class="muted">Run time {r['minutes']} min.</p></body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
