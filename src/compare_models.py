"""
Compare LightGBM vs GRU vs their ensemble (simple average) on the same held-out test rows.

The ensemble is an unweighted average, so nothing is tuned on the test set.
Reports AUROC / AUPRC for every alert (all patients and T2D only) and forecast MAE,
plus how often the two models agree on an alert.

Usage:
    python src/compare_models.py   # -> results/comparison.html, results/comparison.json
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
ALERTS = {"spike": "y_spike", "severe": "y_severe", "hypo": "y_hypo"}
FORECASTS = {"fc30": "g_p30", "fc60": "g_p60", "fc90": "g_p90", "fc120": "g_p120"}
COLORS = {"LightGBM": "#2563eb", "GRU": "#c2410c", "Ensemble": "#15803d"}


def img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def main():
    lgb = pd.read_parquet(RES / "lightgbm" / "test_predictions.parquet")
    gru = pd.read_parquet(RES / "gru" / "test_predictions.parquet")
    keys = ["patient_id", "timestamp"]
    pred_cols = [c for c in gru.columns if c.startswith(("p_", "pred_"))]
    df = lgb.merge(gru[keys + pred_cols], on=keys, suffixes=("_lgb", "_gru"))
    assert len(df) == len(lgb) == len(gru), "test rows differ between models"
    t2d = (df.cohort == 2).to_numpy()
    out, rows, charts = {}, [], []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, target) in zip(axes, ALERTS.items()):
        m = df[target].notna().to_numpy()
        y = df.loc[m, target].to_numpy()
        scores = {"LightGBM": df.loc[m, f"p_{name}_lgb"].to_numpy(), "GRU": df.loc[m, f"p_{name}_gru"].to_numpy()}
        scores["Ensemble"] = (scores["LightGBM"] + scores["GRU"]) / 2
        out[name] = {}
        for model, s in scores.items():
            r = {"auroc": roc_auc_score(y, s), "auprc": average_precision_score(y, s),
                 "auroc_t2d": roc_auc_score(y[t2d[m]], s[t2d[m]]), "auprc_t2d": average_precision_score(y[t2d[m]], s[t2d[m]])}
            out[name][model] = r
            rows.append(f"<tr><td>{name}</td><td>{model}</td><td>{r['auroc']:.3f}</td><td>{r['auprc']:.3f}</td>"
                        f"<td>{r['auroc_t2d']:.3f}</td><td>{r['auprc_t2d']:.3f}</td></tr>")
            fpr, tpr, _ = roc_curve(y[t2d[m]], s[t2d[m]])
            ax.plot(fpr, tpr, color=COLORS[model], label=f"{model} ({r['auroc_t2d']:.3f})")
        a, b = scores["LightGBM"] > 0.5, scores["GRU"] > 0.5
        out[name]["agreement_at_0.5"] = float((a == b).mean())
        ax.plot([0, 1], [0, 1], "k:", lw=0.6)
        ax.set(title=f"{name}: ROC, T2D test patients", xlabel="FPR", ylabel="TPR")
        ax.legend(fontsize=8)
    charts.append(img(fig))

    fc_rows = []
    for name, target in FORECASTS.items():
        m = df[target].notna().to_numpy()
        y = df.loc[m, target].to_numpy()
        p = {"LightGBM": df.loc[m, f"pred_{name}_lgb"].to_numpy(), "GRU": df.loc[m, f"pred_{name}_gru"].to_numpy()}
        p["Ensemble"] = (p["LightGBM"] + p["GRU"]) / 2
        out[name] = {k: {"mae": float(np.mean(np.abs(v - y))), "mae_t2d": float(np.mean(np.abs(v - y)[t2d[m]]))}
                     for k, v in p.items()}
        fc_rows.append(f"<tr><td>{name[2:]} min</td>" + "".join(
            f"<td>{out[name][k]['mae']:.1f} / {out[name][k]['mae_t2d']:.1f}</td>" for k in ("LightGBM", "GRU", "Ensemble")) + "</tr>")

    (RES / "comparison.json").write_text(json.dumps(out, indent=2, default=float))
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Model Comparison</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.muted{{color:#64748b}} img{{max-width:100%}}</style></head><body>
<h1>LightGBM vs GRU vs Ensemble</h1>
<p class="muted">{len(df):,} prediction points from {df.patient_id.nunique()} held-out test patients.
Ensemble = plain average of the two models (no tuning on test data).</p>
<h2>Alerts (next 2 h)</h2>
<table><tr><th>Alert</th><th>Model</th><th>AUROC</th><th>AUPRC</th><th>AUROC (T2D)</th><th>AUPRC (T2D)</th></tr>{''.join(rows)}</table>
<h2>Forecast MAE, mg/dL (all / T2D)</h2>
<table><tr><th>Horizon</th><th>LightGBM</th><th>GRU</th><th>Ensemble</th></tr>{''.join(fc_rows)}</table>
{''.join(charts)}</body></html>"""
    (RES / "comparison.html").write_text(html, encoding="utf-8")

    for name in ALERTS:
        print(name.ljust(7), " | ".join(f"{k}: AUROC {v['auroc']:.3f} (T2D {v['auroc_t2d']:.3f}) AUPRC {v['auprc']:.3f}"
                                         for k, v in out[name].items() if isinstance(v, dict)))
    for name in FORECASTS:
        print(name.ljust(7), " | ".join(f"{k}: MAE {v['mae']:.1f} (T2D {v['mae_t2d']:.1f})" for k, v in out[name].items()))
    print(f"Report -> {RES / 'comparison.html'}")


if __name__ == "__main__":
    main()
