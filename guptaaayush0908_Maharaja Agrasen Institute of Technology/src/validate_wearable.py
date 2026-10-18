"""
Validation report for the simulated wearable / CGM stream (Dataset 2).

1. Consistency with the EHR (CGM-derived GMI vs lab HbA1c)
2. Comparison with published CGM / wearable reference ranges
3. Comparison with REAL patient CGM data (ShanghaiT2DM, Zhao et al. 2023, CC BY 4.0),
   matched on mean glucose level
4. Preview of the prediction targets (glucose spike / hypoglycaemia in next 2 h)

Usage:
    python src/validate_wearable.py   # writes data/wearable/validation_report.html
"""
from __future__ import annotations

import base64
import io
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data" / "wearable"
EHR = ROOT / "data" / "ehr"
SH = ROOT / "data" / "external" / "shanghai"
COLORS = {"T2D": "#c2410c", "Prediabetes": "#ca8a04", "Normoglycemic": "#2563eb", "Shanghai": "#0f766e", "Sim": "#c2410c"}
SAMPLE = 600  # T2D patients used for curve-level analyses


def img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def meal_response(cgm: pd.DataFrame, meals: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Glucose change relative to meal time, on a 15-min grid from -30 to +240 min."""
    offsets = np.arange(-30, 241, 15)
    out = []
    g = cgm.set_index([id_col, "timestamp"]).glucose_mgdl.sort_index()
    for pid, grp in meals.groupby(id_col):
        if pid not in g.index.get_level_values(0):
            continue
        s = g.loc[pid]
        s = s[~s.index.duplicated()]
        for t in grp.timestamp:
            vals = s.reindex(t + pd.to_timedelta(offsets, unit="min"), method="nearest", tolerance=pd.Timedelta("8min"))
            if vals.isna().sum() > 2 or np.isnan(vals.iloc[2]):
                continue
            out.append(vals.to_numpy() - vals.iloc[2])
    return pd.DataFrame(out, columns=offsets)


def main():
    s = pd.read_csv(W / "patient_cgm_summary.csv")
    daily = pd.read_csv(W / "daily_summary.csv")
    feats = pd.read_csv(EHR / "patient_features.csv")[["patient_id", "activity_level", "on_sulfonylurea", "on_basal_insulin", "dx_E11"]]
    dx = pd.read_csv(EHR / "diagnoses.csv")
    s = s.merge(feats, on="patient_id")
    s["hypo_risk_drug"] = (s.on_sulfonylurea + s.on_basal_insulin) > 0
    neuro = set(dx[dx.icd10 == "E11.40"].patient_id)
    t2d = s[s.cohort == "T2D"]
    rng = np.random.default_rng(0)
    sample_ids = rng.choice(t2d.patient_id, SAMPLE, replace=False)

    cgm = pd.read_parquet(W / "cgm.parquet", filters=[("patient_id", "in", list(sample_ids))])
    events = pd.read_csv(W / "events.csv", parse_dates=["timestamp"], usecols=["patient_id", "timestamp", "event_type", "meal_slot"])
    vitals_sample = pd.read_parquet(W / "vitals.parquet", filters=[("patient_id", "in", list(sample_ids[:50]))])
    sh = pd.read_csv(SH / "t2dm_summary.csv")
    sh_cgm = pd.read_csv(SH / "t2dm_cgm.csv", parse_dates=["timestamp"])
    sh_meals = pd.read_csv(SH / "t2dm_meals.csv", parse_dates=["timestamp"])

    # ---------------- 1. EHR consistency
    diff = t2d.gmi_pct - t2d.ehr_last_hba1c
    r_t2d = t2d[["gmi_pct", "ehr_last_hba1c"]].corr().iloc[0, 1]

    # ---------------- meal responses (sim resampled to 15-min like the real sensor)
    cgm15 = cgm[cgm.timestamp.dt.minute % 15 == 0]
    sim_meals = events[(events.event_type == "meal") & events.meal_slot.isin(["breakfast", "lunch", "dinner"])
                       & events.patient_id.isin(sample_ids)]
    mr_sim = meal_response(cgm15, sim_meals, "patient_id")
    sh_meals_main = sh_meals.sort_values("timestamp")
    sh_meals_main = sh_meals_main[sh_meals_main.groupby("record_id").timestamp.diff().fillna(pd.Timedelta("1D")) > pd.Timedelta("2h")]
    mr_sh = meal_response(sh_cgm, sh_meals_main, "record_id")
    ttp_sim = mr_sim.loc[:, 0:].idxmax(axis=1).median()
    ttp_sh = mr_sh.loc[:, 0:].idxmax(axis=1).median()
    rise_sim = mr_sim.loc[:, 0:].max(axis=1).median()
    rise_sh = mr_sh.loc[:, 0:].max(axis=1).median()

    # ---------------- 2. Reference ranges
    normo = s[s.cohort == "Normoglycemic"]
    sleep_by = daily.merge(s[["patient_id", "cohort"]], on="patient_id")
    hrv_neuro = daily[daily.patient_id.isin(neuro)].hrv_rmssd_ms.mean()
    hrv_normo = daily[daily.patient_id.isin(normo.patient_id)].hrv_rmssd_ms.mean()
    steps_act = s.groupby("activity_level").mean_daily_steps.mean()
    REF = [
        ("Healthy: time in range 70-180", normo.tir_70_180_pct.median(), 94, 100, "%"),
        ("Healthy: mean glucose", normo.cgm_mean_glucose.mean(), 95, 110, "mg/dL"),
        ("Healthy: glucose CV", normo.cv_pct.median(), 14, 22, "%"),
        ("T2D: time in range 70-180", t2d.tir_70_180_pct.median(), 45, 75, "%"),
        ("T2D: glucose CV", t2d.cv_pct.median(), 22, 36, "%"),
        ("T2D on SU/insulin: time < 70", t2d[t2d.hypo_risk_drug].tbr_70_pct.mean(), 0.5, 4, "%"),
        ("T2D not on SU/insulin: time < 70", t2d[~t2d.hypo_risk_drug].tbr_70_pct.mean(), 0, 1, "%"),
        ("GMI vs lab HbA1c differ by >= 0.5%", (diff.abs() >= 0.5).mean() * 100, 20, 40, "%"),
        ("Post-meal time to peak (T2D)", ttp_sim, 45, 90, "min"),
        ("Sleep duration", daily.sleep_hours.mean(), 5.5, 7.5, "h"),
        ("REM sleep share", daily.rem_sleep_pct.mean(), 18, 28, "%"),
        ("Deep sleep share", daily.deep_sleep_pct.mean(), 10, 25, "%"),
        ("Resting heart rate", daily.resting_hr.mean(), 55, 80, "bpm"),
        ("Steps: sedentary patients", steps_act.get("Sedentary"), 2000, 5000, "/day"),
        ("Steps: active patients", steps_act.get("Active"), 7500, 13000, "/day"),
        ("HRV: neuropathy vs healthy", hrv_neuro / hrv_normo * 100, 30, 80, "% of healthy"),
    ]
    ref_rows, n_ok = [], 0
    for label, v, lo, hi, unit in REF:
        ok = lo <= v <= hi
        n_ok += ok
        ref_rows.append(f"<tr><td>{label}</td><td>{v:.1f} {unit}</td><td>{lo}&ndash;{hi} {unit}</td>"
                        f"<td class='{'ok' if ok else 'warn'}'>{'in range' if ok else 'check'}</td></tr>")

    # ---------------- 3. Real-data comparison, matched on mean glucose
    sh_ok = sh[sh.n_readings >= 200]
    pool = t2d.sort_values("cgm_mean_glucose")
    matched = []
    for mg in sh_ok.cgm_mean_glucose:
        idx = np.searchsorted(pool.cgm_mean_glucose.to_numpy(), mg)
        matched.append(pool.iloc[max(0, idx - 5): idx + 5])
    matched = pd.concat(matched)
    cmp_rows = ""
    for label, col in [("Mean glucose (mg/dL)", "cgm_mean_glucose"), ("CV (%)", "cv_pct"),
                       ("Time in range 70-180 (%)", "tir_70_180_pct"), ("Time > 180 (%)", "tar_180_pct"),
                       ("Time > 250 (%)", "tar_250_pct"), ("Time < 70 (%)", "tbr_70_pct")]:
        a, b = sh_ok[col].median(), matched[col].median()
        cmp_rows += f"<tr><td>{label}</td><td>{a:.1f}</td><td>{b:.1f}</td></tr>"
    cmp_rows += (f"<tr><td>Post-meal time to peak (min)</td><td>{ttp_sh:.0f}</td><td>{ttp_sim:.0f}</td></tr>"
                 f"<tr><td>Post-meal rise (mg/dL, median)</td><td>{rise_sh:.0f}</td><td>{rise_sim:.0f}</td></tr>")

    # ---------------- 4. Prediction target preview (T2D sample)
    tbl = cgm.pivot(index="timestamp", columns="patient_id", values="glucose_mgdl").astype(float)
    fut_max = tbl[::-1].rolling(24, min_periods=12).max()[::-1].shift(-1)
    fut_min = tbl[::-1].rolling(24, min_periods=12).min()[::-1].shift(-1)
    now_ok = tbl < 180
    spike = ((fut_max > 180) & now_ok).sum().sum() / now_ok.sum().sum() * 100
    hypo = ((fut_min < 70) & (tbl >= 70)).sum().sum() / (tbl >= 70).sum().sum() * 100
    low = tbl < 70
    sustained = low & low.shift(1, fill_value=False) & low.shift(2, fill_value=False)  # >= 15 min (consensus)
    episodes = (sustained & ~(sustained.shift(1, fill_value=False))).sum()
    hypo_pw = episodes.mean() / 2
    spike_ep = ((tbl > 180) & (tbl.shift(1) <= 180)).sum().mean() / 14

    # ---------------- integrity
    full_cgm_stats = pd.read_parquet(W / "cgm.parquet", columns=["glucose_mgdl"]).glucose_mgdl
    checks = [
        ("Every EHR patient has a CGM summary", len(s) == len(pd.read_csv(EHR / "patients.csv"))),
        ("CGM values within sensor range 40-400", full_cgm_stats.dropna().between(40, 400).all()),
        ("CGM coverage >= 90% for 95% of patients", (s.cgm_coverage_pct >= 90).mean() >= 0.95),
        ("No duplicate (patient, timestamp) in sample", not cgm.duplicated(["patient_id", "timestamp"]).any()),
        ("Medication doses only for T2D patients",
         events[events.event_type == "medication"].patient_id.isin(s[s.cohort == "T2D"].patient_id).all()),
    ]
    check_rows = "".join(f"<tr><td>{n}</td><td class='{'ok' if v else 'warn'}'>{'pass' if v else 'FAIL'}</td></tr>" for n, v in checks)

    # ---------------- charts
    charts = []
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for c in ["Normoglycemic", "Prediabetes", "T2D"]:
        d = s[s.cohort == c]
        ax[0].scatter(d.ehr_last_hba1c, d.gmi_pct, s=4, alpha=0.4, color=COLORS[c], label=c)
        ax[1].hist(d.tir_70_180_pct, bins=40, alpha=0.6, color=COLORS[c], label=c)
        ax[2].hist(d.cv_pct, bins=40, alpha=0.6, color=COLORS[c], label=c)
    ax[0].plot([4, 13], [4, 13], "k--", lw=0.8)
    ax[0].set(title=f"CGM GMI vs lab HbA1c (T2D r = {r_t2d:.2f})", xlabel="EHR HbA1c %", ylabel="GMI %")
    ax[1].set_title("Time in range 70-180 (%)"); ax[2].set_title("Glucose variability CV (%)"); ax[0].legend()
    charts.append(img(fig))

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    ax[0].scatter(t2d.cgm_mean_glucose, t2d.tir_70_180_pct, s=3, alpha=0.25, color=COLORS["Sim"], label="Simulated T2D")
    ax[0].scatter(sh_ok.cgm_mean_glucose, sh_ok.tir_70_180_pct, s=18, color=COLORS["Shanghai"], label="Real (ShanghaiT2DM)")
    ax[0].set(title="Time in range vs mean glucose", xlabel="mean glucose mg/dL", ylabel="TIR %"); ax[0].legend()
    ax[1].scatter(t2d.cgm_mean_glucose, t2d.cv_pct, s=3, alpha=0.25, color=COLORS["Sim"])
    ax[1].scatter(sh_ok.cgm_mean_glucose, sh_ok.cv_pct, s=18, color=COLORS["Shanghai"])
    ax[1].set(title="Variability (CV) vs mean glucose", xlabel="mean glucose mg/dL", ylabel="CV %")
    for df_, lab, col in [(mr_sim, "Simulated (Indian meals)", COLORS["Sim"]), (mr_sh, "Real (ShanghaiT2DM)", COLORS["Shanghai"])]:
        m, q1, q3 = df_.median(), df_.quantile(0.25), df_.quantile(0.75)
        ax[2].plot(df_.columns, m, color=col, label=lab)
        ax[2].fill_between(df_.columns, q1, q3, color=col, alpha=0.15)
    ax[2].axvline(0, color="k", lw=0.6)
    ax[2].set(title="Glucose change after a main meal (median, IQR)", xlabel="minutes from meal", ylabel="mg/dL"); ax[2].legend()
    charts.append(img(fig))

    # diurnal profile
    fig, ax = plt.subplots(figsize=(15, 3.4))
    for df_, idc, lab, col in [(cgm15, "patient_id", "Simulated T2D (Indian meal times)", COLORS["Sim"]),
                               (sh_cgm, "record_id", "Real ShanghaiT2DM", COLORS["Shanghai"])]:
        d = df_.dropna().copy()
        d["dev"] = d.glucose_mgdl - d.groupby(idc).glucose_mgdl.transform("mean")
        prof = d.groupby(d.timestamp.dt.hour + d.timestamp.dt.minute / 60).dev.mean()
        ax.plot(prof.index, prof.values, color=col, label=lab)
    ax.set(title="Average daily glucose pattern (deviation from each patient's mean)", xlabel="hour of day", ylabel="mg/dL")
    ax.set_xticks(range(0, 25, 2)); ax.legend()
    charts.append(img(fig))

    # one patient, 3 days
    pid = t2d[t2d.patient_id.isin(sample_ids[:50]) & t2d.hypo_risk_drug].patient_id.iloc[0] \
        if (t2d.patient_id.isin(sample_ids[:50]) & t2d.hypo_risk_drug).any() else sample_ids[0]
    c1 = cgm[cgm.patient_id == pid].set_index("timestamp").glucose_mgdl
    v1 = vitals_sample[vitals_sample.patient_id == pid].set_index("timestamp")
    win = slice(c1.index.min() + pd.Timedelta("4D"), c1.index.min() + pd.Timedelta("7D"))
    ev1 = pd.read_csv(W / "events.csv", parse_dates=["timestamp"]).query("patient_id == @pid")
    fig, ax = plt.subplots(3, 1, figsize=(15, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]})
    ax[0].plot(c1[win].index, c1[win].values, color=COLORS["Sim"], lw=1.2)
    ax[0].axhspan(70, 180, color="#16a34a", alpha=0.08)
    for r in ev1[(ev1.event_type == "meal") & ev1.timestamp.between(win.start, win.stop)].itertuples():
        ax[0].annotate(f"{r.detail}\n{int(r.carbs_g)}g", (r.timestamp, 45), fontsize=7, ha="center", color="#475569")
        ax[0].axvline(r.timestamp, color="#94a3b8", lw=0.6, ls=":")
    for r in ev1[(ev1.event_type == "medication") & ev1.timestamp.between(win.start, win.stop)].itertuples():
        ax[0].plot(r.timestamp, 390, marker="v", color="#16a34a" if r.dose_taken else "#dc2626", ms=6)
    ax[0].set(ylabel="CGM mg/dL", ylim=(35, 410), title=f"{pid}: 3 days of the twin (green/red triangles = dose taken/missed)")
    ax[1].bar(v1[win].index, v1[win].steps, width=0.0035, color="#64748b"); ax[1].set_ylabel("steps")
    ax[2].plot(v1[win].index, v1[win].heart_rate_bpm, color="#7c3aed", lw=0.8); ax[2].set_ylabel("HR bpm")
    asleep = v1[win].sleep_stage.notna()
    for a in ax:
        a.fill_between(v1[win].index, 0, 1, where=asleep, color="#1e3a8a", alpha=0.07, transform=a.get_xaxis_transform())
    charts.append(img(fig))

    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Wearable Validation Report</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.ok{{color:#15803d;font-weight:600}} .warn{{color:#b45309;font-weight:600}} .muted{{color:#64748b}} img{{max-width:100%}}
</style></head><body>
<h1>Wearable &amp; CGM Stream &mdash; Validation Report</h1>
<p>{len(s):,} patients &times; 14 days &times; 288 readings/day. CGM (glucose), smartwatch (heart rate, steps, sleep stage),
and an event log (meals, medication doses, exercise). Fully synthetic and driven by each patient's EHR twin parameters.</p>

<h2>1. Agreement with the EHR</h2>
<p>CGM-derived GMI vs latest lab HbA1c in T2D: correlation <b>{r_t2d:.2f}</b>, mean difference {diff.mean():+.2f}%,
{(diff.abs() >= 0.5).mean():.0%} of patients differ by &ge; 0.5% (real-world studies report about 28%).</p>

<h2>2. Published reference ranges ({n_ok}/{len(REF)} in range)</h2>
<p class="muted">Approximate ranges from CGM studies in healthy adults and T2D, international consensus on time in range,
and wearable sleep and activity studies.</p>
<table><tr><th>Metric</th><th>Simulated</th><th>Reference</th><th></th></tr>{''.join(ref_rows)}</table>

<h2>3. Comparison with real patient CGM data</h2>
<p>ShanghaiT2DM (Zhao et al., <i>Scientific Data</i> 2023, CC BY 4.0): {len(sh_ok)} recordings from T2D patients, 15-min CGM.
Simulated T2D patients were matched to each real recording on mean glucose (10 nearest each).
The Shanghai cohort is mostly hospital patients (about half on insulin) eating Chinese meals at earlier times,
so meal timing and the shape of the daily curve are expected to differ.</p>
<table><tr><th>Metric (median)</th><th>Real: ShanghaiT2DM</th><th>Simulated (glucose-matched)</th></tr>{cmp_rows}</table>

<h2>4. Prediction targets preview (T2D, {SAMPLE} patients)</h2>
<table>
<tr><td>Readings &lt;180 followed by a <b>spike &gt;180 within 2 h</b></td><td>{spike:.1f}% of time points</td></tr>
<tr><td>Readings &ge;70 followed by <b>hypoglycaemia &lt;70 within 2 h</b></td><td>{hypo:.2f}% of time points</td></tr>
<tr><td>Hyperglycaemic excursions (&gt;180) per patient per day</td><td>{spike_ep:.1f}</td></tr>
<tr><td>Hypoglycaemic episodes (&lt;70) per patient per week</td><td>{hypo_pw:.2f}</td></tr>
</table>

<h2>5. Data integrity</h2><table>{check_rows}</table>

<h2>6. Charts</h2>{''.join(charts)}
</body></html>"""
    (W / "validation_report.html").write_text(html, encoding="utf-8")

    print(f"T2D GMI-HbA1c r={r_t2d:.2f}, mean diff {diff.mean():+.2f}, |diff|>=0.5: {(diff.abs() >= 0.5).mean():.0%}")
    print(f"Reference ranges in range: {n_ok}/{len(REF)}")
    for label, v, lo, hi, unit in REF:
        print(f"  {'OK ' if lo <= v <= hi else '!! '} {label}: {v:.1f} {unit} (ref {lo}-{hi})")
    print("Real vs simulated (glucose-matched medians):")
    print("  " + cmp_rows.replace("</td><td>", " | ").replace("<tr><td>", "").replace("</td></tr>", "\n  "))
    print(f"Spike-ahead positives {spike:.1f}% | hypo-ahead {hypo:.2f}% | hypo episodes/wk {hypo_pw:.2f}")
    print(f"Integrity: {sum(v for _, v in checks)}/{len(checks)}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
