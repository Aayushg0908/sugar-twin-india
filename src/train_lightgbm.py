"""
Train and evaluate the LightGBM models of the digital twin.

Models (all use the fused EHR + wearable features from build_features.py):
  spike    P(glucose > 180 mg/dL in the next 2 h | glucose now < 180)   <- primary alert
  severe   P(glucose > 250 mg/dL in the next 2 h)
  hypo     P(glucose <  70 mg/dL for >= 10 min in the next 2 h | glucose now >= 70)
  fc30 .. fc120   glucose forecast 30 / 60 / 90 / 120 min ahead (predicts the change from now)

Evaluation is on held-out TEST patients never seen in training (patient-level split in data/splits.csv):
AUROC, AUPRC, sensitivity / specificity / PPV at an alert threshold chosen on the validation set,
false alerts per patient-day, event detection rate and lead time, subgroup fairness, calibration,
forecast MAE / RMSE and Clarke Error Grid, and SHAP explanations.

Outputs:
  models/lightgbm/*.txt                 trained models (load with lightgbm.Booster(model_file=...))
  results/lightgbm/metrics.json         every number in the report + run info
  results/lightgbm/report.html          charts and tables
  results/lightgbm/test_predictions.parquet   predictions for the dashboard / ensembling with the GRU
  results/lightgbm/feature_importance.csv

Usage:
    python src/train_lightgbm.py            # full training
    python src/train_lightgbm.py --quick    # smoke test: 20% of train patients, 150 rounds
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import subprocess
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "model" / "features.parquet"
MODELS = ROOT / "models" / "lightgbm"
RESULTS = ROOT / "results" / "lightgbm"

ID_COLS = ["patient_id", "timestamp", "split"]
TARGETS = ["y_spike", "y_severe", "y_hypo", "g_p30", "g_p60", "g_p90", "g_p120"]
CLASSIFIERS = {"spike": "y_spike", "severe": "y_severe", "hypo": "y_hypo"}
FORECASTS = {"fc30": "g_p30", "fc60": "g_p60", "fc90": "g_p90", "fc120": "g_p120"}
TARGET_SENSITIVITY = {"spike": 0.80, "severe": 0.80, "hypo": 0.80}

PARAMS_CLS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  metric=["auc", "average_precision"], verbose=-1, seed=42)
PARAMS_REG = dict(objective="huber", alpha=30.0, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  metric=["l1", "l2"], verbose=-1, seed=42)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def threshold_for_sensitivity(y, p, target):
    """Highest threshold that still catches `target` of positives (chosen on validation)."""
    pos = np.sort(p[y == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    return float(pos[min(int(np.ceil(target * len(pos))) - 1, len(pos) - 1)])


def binary_report(y, p, thr, pid, ts):
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    df = pd.DataFrame({"pid": pid, "ts": ts, "alert": pred}).sort_values(["pid", "ts"])
    onset = df.alert & ~df.groupby("pid").alert.shift(1, fill_value=False)
    patient_days = df.groupby("pid").ts.agg(lambda s: (s.max() - s.min()).total_seconds() / 86400).sum()
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "prevalence": float(y.mean()), "threshold": thr,
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1), "ppv": tp / max(tp + fp, 1),
            "alerts_per_patient_day": float(onset.sum() / max(patient_days, 1e-9)),
            "false_alert_points_per_patient_day": float(fp / max(patient_days, 1e-9)), "n": int(len(y))}


def event_detection(df_pred, score_col, thr, kind):
    """For each new excursion (spike above 180 / drop below 70) in test patients, was there an alert in the
    preceding 2 h, and how early? Uses the prediction grid (every 15-30 min)."""
    rows, lead = [], []
    for pid, g in df_pred.groupby("patient_id"):
        g = g.sort_values("timestamp")
        now = g.g_now.to_numpy()
        crossed = (now > 180) if kind == "spike" else (now < 70)
        before_ok = (now <= 180) if kind == "spike" else (now >= 70)
        t = g.timestamp.to_numpy()
        s = g[score_col].to_numpy()
        for k in np.where(crossed[1:] & before_ok[:-1])[0] + 1:
            w = (t >= t[k] - np.timedelta64(120, "m")) & (t < t[k]) & before_ok
            if not w.any():
                continue
            hits = np.where(w & (s >= thr))[0]
            rows.append(len(hits) > 0)
            if len(hits):
                lead.append((t[k] - t[hits[0]]) / np.timedelta64(1, "m"))
    return {"events": len(rows), "detection_rate": float(np.mean(rows)) if rows else None,
            "median_lead_time_min": float(np.median(lead)) if lead else None,
            "mean_lead_time_min": float(np.mean(lead)) if lead else None}


def clarke_zones(ref, pred):
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    zone = np.full(len(ref), "B", dtype=object)
    a = ((ref <= 70) & (pred <= 70)) | ((pred <= 1.2 * ref) & (pred >= 0.8 * ref))
    e = ((ref >= 180) & (pred <= 70)) | ((ref <= 70) & (pred >= 180))
    c = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | ((ref >= 130) & (ref <= 180) & (pred <= (7 / 5) * ref - 182))
    d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | ((ref <= 175 / 3) & (pred >= 70) & (pred <= 180)) | \
        ((ref >= 175 / 3) & (ref <= 70) & (pred >= (6 / 5) * ref))
    zone[d] = "D"
    zone[c] = "C"
    zone[e] = "E"
    zone[a] = "A"
    return zone


def calibration(y, p, bins=10):
    q = np.clip((p * bins).astype(int), 0, bins - 1)
    return pd.DataFrame({"q": q, "y": y, "p": p}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(quick: bool):
    t_start = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    log(f"Loading {FEATURES}")
    df = pd.read_parquet(FEATURES)
    feat_cols = [c for c in df.columns if c not in ID_COLS + TARGETS]
    df[feat_cols] = df[feat_cols].astype(np.float32)
    if quick:
        keep = df.loc[df.split == "train", "patient_id"].drop_duplicates().sample(frac=0.2, random_state=0)
        df = df[(df.split != "train") | df.patient_id.isin(keep)]
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    n_rows = len(df)
    del df  # free memory
    log(f"{n_rows:,} rows, {len(feat_cols)} features | train {len(tr):,} / val {len(va):,} / test {len(te):,} rows "
        f"| patients {tr.patient_id.nunique()} / {va.patient_id.nunique()} / {te.patient_id.nunique()}")

    rounds = 150 if quick else 3000
    metrics = {"run": {"machine": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                       "lightgbm": lgb.__version__, "quick": quick, "rows": n_rows, "n_features": len(feat_cols),
                       "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                    capture_output=True, text=True).stdout.strip() or None}}
    preds = te[["patient_id", "timestamp", "g_now", "cohort", "setting", "on_sulfonylurea", "on_basal_insulin"]
               + list(CLASSIFIERS.values()) + list(FORECASTS.values())].copy()
    charts, curves = [], {}

    # ---------------- classifiers
    for name, target in CLASSIFIERS.items():
        log(f"Training '{name}' classifier")
        m_tr, m_va, m_te = tr[target].notna(), va[target].notna(), te[target].notna()
        dtr = lgb.Dataset(tr.loc[m_tr, feat_cols], tr.loc[m_tr, target], free_raw_data=True)
        dva = lgb.Dataset(va.loc[m_va, feat_cols], va.loc[m_va, target], reference=dtr, free_raw_data=True)
        t0 = time.time()
        model = lgb.train(PARAMS_CLS, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False), lgb.log_evaluation(200)])
        model.save_model(MODELS / f"{name}.txt")
        p_va = model.predict(va.loc[m_va, feat_cols], num_iteration=model.best_iteration)
        p_te = model.predict(te.loc[m_te, feat_cols], num_iteration=model.best_iteration)
        thr = threshold_for_sensitivity(va.loc[m_va, target].to_numpy(), p_va, TARGET_SENSITIVITY[name])
        y_te = te.loc[m_te, target].to_numpy()
        res = binary_report(y_te, p_te, thr, te.loc[m_te, "patient_id"].to_numpy(), te.loc[m_te, "timestamp"])
        res.update(best_iteration=model.best_iteration, train_seconds=round(time.time() - t0, 1))
        preds[f"p_{name}"] = np.nan
        preds.loc[m_te, f"p_{name}"] = p_te

        # baselines: simple trend rule ("where will glucose be in 1 h if the 30-min trend continues?")
        proj = (te.loc[m_te, "g_now"] + 2 * te.loc[m_te, "g_delta_30"]).fillna(te.loc[m_te, "g_now"]).to_numpy()
        base_score = -proj if name == "hypo" else proj
        res["baseline_trend_auroc"] = float(roc_auc_score(y_te, base_score))
        res["baseline_trend_auprc"] = float(average_precision_score(y_te, base_score))
        now_score = -te.loc[m_te, "g_now"].to_numpy() if name == "hypo" else te.loc[m_te, "g_now"].to_numpy()
        res["baseline_current_glucose_auroc"] = float(roc_auc_score(y_te, now_score))

        # subgroups (fairness)
        sub = {}
        g = te.loc[m_te]
        groups = {"cohort": {0: "Normoglycemic", 1: "Prediabetes", 2: "T2D"}, "setting": {0: "Rural", 1: "Urban"}}
        for col, labels in groups.items():
            for code, lab in labels.items():
                mask = (g[col] == code).to_numpy()
                if mask.sum() > 100 and 0 < y_te[mask].sum() < mask.sum():
                    sub[f"{col}={lab}"] = {"auroc": float(roc_auc_score(y_te[mask], p_te[mask])),
                                           "sensitivity": float(((p_te[mask] >= thr) & (y_te[mask] == 1)).sum() / max((y_te[mask] == 1).sum(), 1)),
                                           "n": int(mask.sum())}
        risky = ((g.on_sulfonylurea + g.on_basal_insulin) > 0).to_numpy()
        for lab, mask in [("on SU/insulin", risky), ("not on SU/insulin", ~risky)]:
            if mask.sum() > 100 and 0 < y_te[mask].sum() < mask.sum():
                sub[lab] = {"auroc": float(roc_auc_score(y_te[mask], p_te[mask])), "n": int(mask.sum())}
        res["subgroups"] = sub
        if name in ("spike", "hypo"):
            scored = preds.assign(**{f"p_{name}": preds[f"p_{name}"].fillna(0)})  # rows already past the threshold score 0
            res["event_detection"] = event_detection(scored, f"p_{name}", thr, name)
        res["calibration"] = calibration(y_te, p_te).round(4).reset_index().to_dict("records")
        metrics[name] = res
        curves[name] = (y_te, p_te, base_score)
        log(f"  {name}: AUROC {res['auroc']:.3f} (trend rule {res['baseline_trend_auroc']:.3f}), AUPRC {res['auprc']:.3f}, "
            f"sens {res['sensitivity']:.2f}, spec {res['specificity']:.2f}, PPV {res['ppv']:.2f}, "
            f"alerts/patient-day {res['alerts_per_patient_day']:.2f}")

        if name == "spike":
            contrib = model.predict(te.loc[m_te, feat_cols].sample(min(20000, int(m_te.sum())), random_state=0),
                                    num_iteration=model.best_iteration, pred_contrib=True)
            shap_imp = pd.Series(np.abs(contrib[:, :-1]).mean(0), index=feat_cols).sort_values(ascending=False)
            gain = pd.Series(model.feature_importance("gain"), index=feat_cols)
            pd.DataFrame({"mean_abs_shap": shap_imp, "gain": gain}).sort_values("mean_abs_shap", ascending=False) \
                .to_csv(RESULTS / "feature_importance.csv")
            metrics["spike"]["top_features"] = shap_imp.head(15).round(4).to_dict()
            fig, ax = plt.subplots(figsize=(8, 6))
            shap_imp.head(20)[::-1].plot.barh(ax=ax, color="#c2410c")
            ax.set_title("Spike model: top 20 features (mean |SHAP|)")
            charts.append(img(fig))

    # ---------------- forecasts
    for name, target in FORECASTS.items():
        log(f"Training '{name}' forecaster")
        m_tr, m_va, m_te = tr[target].notna(), va[target].notna(), te[target].notna()
        dtr = lgb.Dataset(tr.loc[m_tr, feat_cols], (tr.loc[m_tr, target] - tr.loc[m_tr, "g_now"]), free_raw_data=True)
        dva = lgb.Dataset(va.loc[m_va, feat_cols], (va.loc[m_va, target] - va.loc[m_va, "g_now"]), reference=dtr)
        t0 = time.time()
        model = lgb.train(PARAMS_REG, dtr, rounds, valid_sets=[dva], valid_names=["val"],
                          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False), lgb.log_evaluation(200)])
        model.save_model(MODELS / f"{name}.txt")
        y = te.loc[m_te, target].to_numpy()
        now = te.loc[m_te, "g_now"].to_numpy()
        p = now + model.predict(te.loc[m_te, feat_cols], num_iteration=model.best_iteration)
        h = int(name[2:])
        trend = np.clip(now + te.loc[m_te, "g_delta_30"].fillna(0).to_numpy() * h / 30, 40, 400)
        zones = pd.Series(clarke_zones(y, p)).value_counts(normalize=True).reindex(list("ABCDE"), fill_value=0)
        res = {"mae": float(np.mean(np.abs(p - y))), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
               "mard_pct": float(np.mean(np.abs(p - y) / y) * 100),
               "baseline_persistence_mae": float(np.mean(np.abs(now - y))),
               "baseline_trend_mae": float(np.mean(np.abs(trend - y))),
               "clarke_zones_pct": (zones * 100).round(2).to_dict(), "best_iteration": model.best_iteration,
               "train_seconds": round(time.time() - t0, 1), "n": int(len(y))}
        preds[f"pred_{name}"] = np.nan
        preds.loc[m_te, f"pred_{name}"] = p
        metrics[name] = res
        log(f"  {name}: MAE {res['mae']:.1f} mg/dL (persistence {res['baseline_persistence_mae']:.1f}, "
            f"trend {res['baseline_trend_mae']:.1f}), Clarke A+B {zones['A'] * 100 + zones['B'] * 100:.1f}%")
        if name in ("fc60", "fc120"):
            fig, ax = plt.subplots(figsize=(6, 6))
            sample = np.random.default_rng(0).choice(len(y), min(8000, len(y)), replace=False)
            ax.scatter(y[sample], p[sample], s=2, alpha=0.3, color="#2563eb")
            ax.plot([0, 400], [0, 400], "k--", lw=0.8)
            for x0, y0, x1, y1 in [(0, 70, 58.3, 70), (70, 84, 70, 400), (70, 56, 400, 320), (180, 0, 180, 70),
                                   (240, 70, 240, 180), (240, 70, 400, 70), (130, 0, 180, 70), (70, 180, 290, 400)]:
                ax.plot([x0, x1], [y0, y1], color="#64748b", lw=0.6)
            ax.set(xlim=(40, 400), ylim=(40, 400), xlabel="actual glucose (mg/dL)", ylabel="predicted (mg/dL)",
                   title=f"Clarke Error Grid, {h}-min forecast (A {zones['A']:.0%}, B {zones['B']:.0%})")
            charts.append(img(fig))

    # ---------------- curves
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, (y, p, b) in curves.items():
        if name == "severe":
            continue
        fpr, tpr, _ = roc_curve(y, p)
        bfpr, btpr, _ = roc_curve(y, b)
        ax[0].plot(fpr, tpr, label=f"{name} LightGBM")
        ax[0].plot(bfpr, btpr, ls="--", label=f"{name} trend rule")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{name} LightGBM")
        cal = calibration(y, p)
        ax[2].plot(cal.pred, cal.obs, marker="o", label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.6)
    ax[0].set(title="ROC (test patients)", xlabel="false positive rate", ylabel="true positive rate"); ax[0].legend(fontsize=8)
    ax[1].set(title="Precision-recall", xlabel="recall", ylabel="precision"); ax[1].legend(fontsize=8)
    ax[2].plot([0, 1], [0, 1], "k:", lw=0.6)
    ax[2].set(title="Calibration", xlabel="predicted risk", ylabel="observed rate"); ax[2].legend(fontsize=8)
    charts.insert(0, img(fig))

    preds.to_parquet(RESULTS / "test_predictions.parquet", index=False)
    metrics["run"]["total_minutes"] = round((time.time() - t_start) / 60, 1)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(metrics, charts)
    log(f"Done in {metrics['run']['total_minutes']} min. Results -> {RESULTS}")


def write_report(m, charts):
    def cls_row(name, label):
        r = m[name]
        ed = r.get("event_detection", {})
        caught = f"{ed['detection_rate']:.0%}" if ed.get("detection_rate") is not None else ""
        warn = f"{ed['median_lead_time_min']:.0f} min" if ed.get("median_lead_time_min") is not None else ""
        return (f"<tr><td>{label}</td><td>{r['auroc']:.3f}</td><td>{r['baseline_trend_auroc']:.3f}</td>"
                f"<td>{r['auprc']:.3f}</td><td>{r['sensitivity']:.0%}</td><td>{r['specificity']:.0%}</td>"
                f"<td>{r['ppv']:.0%}</td><td>{r['alerts_per_patient_day']:.2f}</td>"
                f"<td>{caught}</td><td>{warn}</td></tr>")

    def fc_row(name, label):
        r = m[name]
        z = r["clarke_zones_pct"]
        return (f"<tr><td>{label}</td><td>{r['mae']:.1f}</td><td>{r['rmse']:.1f}</td><td>{r['mard_pct']:.1f}%</td>"
                f"<td>{r['baseline_persistence_mae']:.1f}</td><td>{r['baseline_trend_mae']:.1f}</td>"
                f"<td>{z['A'] + z['B']:.1f}%</td></tr>")

    sub_rows = "".join(f"<tr><td>{k}</td><td>{v['auroc']:.3f}</td><td>{v['n']:,}</td></tr>"
                       for k, v in m["spike"]["subgroups"].items())
    top = "".join(f"<li>{k}</li>" for k in list(m["spike"]["top_features"])[:10])
    run = m["run"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>LightGBM Results</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.muted{{color:#64748b}} img{{max-width:100%}}</style></head><body>
<h1>LightGBM Digital Twin &mdash; Test Results</h1>
<p class="muted">Machine {run['machine']} &middot; LightGBM {run['lightgbm']} &middot; {run['rows']:,} rows &middot;
{run['n_features']} features &middot; {run['total_minutes']} min &middot; commit {run['git_commit']}
{'&middot; <b>QUICK MODE</b>' if run['quick'] else ''}. All numbers are on held-out test patients.</p>
<h2>Adverse-event alerts (next 2 hours)</h2>
<p class="muted">Alert threshold set on validation patients to catch 80% of events.</p>
<table><tr><th>Event</th><th>AUROC</th><th>Trend-rule AUROC</th><th>AUPRC</th><th>Sensitivity</th><th>Specificity</th>
<th>PPV</th><th>Alerts / patient-day</th><th>Events caught</th><th>Median warning</th></tr>
{cls_row('spike', 'Spike &gt; 180')}{cls_row('severe', 'Severe &gt; 250')}{cls_row('hypo', 'Hypo &lt; 70')}</table>
<h2>Glucose forecast</h2>
<table><tr><th>Horizon</th><th>MAE</th><th>RMSE</th><th>MARD</th><th>Persistence MAE</th><th>Trend MAE</th><th>Clarke A+B</th></tr>
{fc_row('fc30', '30 min')}{fc_row('fc60', '60 min')}{fc_row('fc90', '90 min')}{fc_row('fc120', '120 min')}</table>
<h2>Fairness: spike model by subgroup</h2><table><tr><th>Group</th><th>AUROC</th><th>Rows</th></tr>{sub_rows}</table>
<h2>What drives the spike predictions</h2><ol>{top}</ol>
<h2>Charts</h2>{''.join(charts)}
</body></html>"""
    (RESULTS / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fast smoke test on a subset")
    main(ap.parse_args().quick)
