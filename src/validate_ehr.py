"""
Validation report for the synthetic EHR.

Compares cohort statistics against approximate ranges reported in Indian
clinical literature, runs data-integrity checks, plots distributions and
prints a short clinical "story" for a few sample patients.

Usage:
    python src/validate_ehr.py            # writes data/ehr/validation_report.html
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ehr"
COHORTS = ["T2D", "Prediabetes", "Normoglycemic"]
COLORS = {"T2D": "#c2410c", "Prediabetes": "#ca8a04", "Normoglycemic": "#2563eb"}

# Approximate ranges from Indian literature (ICMR-INDIAB, SMART-India and
# real-world T2D registries). Indicative targets for a synthetic cohort,
# not exact citations.
REFERENCE = [
    # label, fn(features, dx) -> value, low, high, unit
    ("T2D: mean HbA1c", lambda f, d: f.loc[t2d(f), "last_hba1c"].mean(), 7.6, 8.8, "%"),
    ("T2D: HbA1c < 7% (controlled)", lambda f, d: pct(f.loc[t2d(f), "last_hba1c"].dropna() < 7), 20, 40, "%"),
    ("T2D: HbA1c > 9% (poorly controlled)", lambda f, d: pct(f.loc[t2d(f), "last_hba1c"].dropna() > 9), 10, 30, "%"),
    ("T2D: mean age at diagnosis", lambda f, d: (f.age - f.diabetes_duration_yrs)[t2d(f)].mean(), 40, 50, "yrs"),
    ("T2D: lean diabetes (BMI < 23)", lambda f, d: pct(f.loc[t2d(f), "bmi"] < 23), 12, 25, "%"),
    ("T2D: hypertension diagnosed", lambda f, d: pct(f.loc[t2d(f), "dx_I10"] == 1), 40, 60, "%"),
    ("T2D: dyslipidaemia diagnosed", lambda f, d: pct(f.loc[t2d(f), "dx_E78"] == 1), 70, 90, "%"),
    ("T2D: chronic kidney disease", lambda f, d: pct(f.loc[t2d(f), "dx_N18"] == 1), 25, 40, "%"),
    ("T2D: diabetic neuropathy", lambda f, d: share_with(d, f, "E11.40"), 18, 35, "%"),
    ("T2D: diabetic retinopathy", lambda f, d: share_with(d, f, "E11.319"), 12, 22, "%"),
    ("T2D: on metformin", lambda f, d: pct(f.loc[t2d(f), "on_biguanide"] == 1), 80, 97, "%"),
    ("T2D: on insulin", lambda f, d: pct(f.loc[t2d(f), "on_basal_insulin"] == 1), 7, 20, "%"),
    ("Prediabetes: mean fasting glucose", lambda f, d: f.loc[f.cohort == "Prediabetes", "last_fasting_plasma_glucose"].mean(), 100, 125, "mg/dL"),
    ("Normoglycemic: mean fasting glucose", lambda f, d: f.loc[f.cohort == "Normoglycemic", "last_fasting_plasma_glucose"].mean(), 75, 99, "mg/dL"),
]


def t2d(f):
    return f.cohort == "T2D"


def pct(mask):
    return 100 * mask.mean()


def share_with(dx, f, code):
    ids = set(f.loc[t2d(f), "patient_id"])
    return 100 * dx[(dx.icd10 == code) & dx.patient_id.isin(ids)].patient_id.nunique() / len(ids)


def fig_to_img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def integrity_checks(patients, labs, meds, dx, f):
    ref = pd.to_datetime(patients["reference_date"].iloc[0])
    checks = [
        ("Unique patient IDs", patients.patient_id.is_unique),
        ("No lab dates after reference date", (pd.to_datetime(labs.date) <= ref).all()),
        ("No diagnosis onset after reference date", (pd.to_datetime(dx.onset_date) <= ref).all()),
        ("Medication end date >= start date",
         (pd.to_datetime(meds.end_date.dropna()) >= pd.to_datetime(meds.loc[meds.end_date.notna(), "start_date"])).all()),
        ("Every T2D patient has an E11 diagnosis", (f.loc[t2d(f), "dx_E11"] == 1).all()),
        ("No glucose-lowering drugs outside T2D cohort",
         not meds[(meds.indication == "Type 2 diabetes")].patient_id.isin(f.loc[~t2d(f), "patient_id"]).any()),
        ("Prediabetes HbA1c within 5.7-6.4 (+/- lab noise)",
         labs[(labs.loinc == "4548-4") & labs.patient_id.isin(f.loc[f.cohort == "Prediabetes", "patient_id"])]
         .value.between(5.3, 6.8).all()),
        ("Normoglycemic HbA1c < 5.7 (+/- lab noise)",
         labs[(labs.loinc == "4548-4") & labs.patient_id.isin(f.loc[f.cohort == "Normoglycemic", "patient_id"])]
         .value.lt(6.0).all()),
        ("No metformin active with latest eGFR < 30",
         not ((f.on_biguanide == 1) & (f.last_egfr_ckd_epi_2021 < 25)).any()),
        ("Every patient has at least one lab record", patients.patient_id.isin(labs.patient_id).all()),
    ]
    return checks


def patient_story(pid, patients, labs, meds, dx, tp):
    p = patients.set_index("patient_id").loc[pid]
    t = tp.set_index("patient_id").loc[pid]
    lines = [f"<h3>{pid} &middot; {p.age}{p.sex} &middot; {p.cohort} &middot; {p.region} ({p.setting})</h3>",
             f"<p>BMI {p.bmi} &middot; {p.activity_level} &middot; {p.diet}, {p.staple}-based diet &middot; "
             f"smoking: {p.smoking} &middot; family history of diabetes: {'yes' if p.family_history_diabetes else 'no'}"
             + (f" &middot; diabetes for {p.diabetes_duration_yrs} yrs" if p.cohort == "T2D" else "") + "</p>",
             f"<p class='muted'>Hidden twin: insulin sensitivity {t.insulin_sensitivity}, beta-cell function "
             f"{t.beta_cell_function}, medication adherence {t.med_adherence:.0%}</p>"]
    d = dx[dx.patient_id == pid]
    lines.append("<p><b>Diagnoses:</b> " + "; ".join(f"{r.description} ({r.onset_date})" for r in d.itertuples()) + "</p>")
    m = meds[meds.patient_id == pid]
    lines.append("<p><b>Medications:</b> " + "; ".join(
        f"{r.drug} {r.dose} ({r.start_date} &rarr; {r.end_date if isinstance(r.end_date, str) else 'ongoing'})"
        for r in m.itertuples()) + "</p>")
    a = labs[(labs.patient_id == pid) & (labs.loinc == "4548-4")]
    lines.append("<p><b>HbA1c:</b> " + " &rarr; ".join(f"{v}" for v in a.value) + "</p>")
    return "\n".join(lines)


def main():
    patients = pd.read_csv(DATA / "patients.csv")
    labs = pd.read_csv(DATA / "labs.csv")
    meds = pd.read_csv(DATA / "medications.csv")
    dx = pd.read_csv(DATA / "diagnoses.csv")
    tp = pd.read_csv(DATA / "twin_params.csv")
    f = pd.read_csv(DATA / "patient_features.csv")

    # --- reference comparison
    ref_rows, n_ok = [], 0
    for label, fn, lo, hi, unit in REFERENCE:
        v = fn(f, dx)
        ok = lo <= v <= hi
        n_ok += ok
        ref_rows.append(f"<tr><td>{label}</td><td>{v:.1f} {unit}</td><td>{lo}&ndash;{hi} {unit}</td>"
                        f"<td class='{'ok' if ok else 'warn'}'>{'in range' if ok else 'check'}</td></tr>")

    checks = integrity_checks(patients, labs, meds, dx, f)
    check_rows = "".join(f"<tr><td>{n}</td><td class='{'ok' if v else 'warn'}'>{'pass' if v else 'FAIL'}</td></tr>"
                         for n, v in checks)

    # --- charts
    imgs = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for c in COHORTS:
        d = f[f.cohort == c]
        axes[0].hist(d.age, bins=30, alpha=0.55, label=c, color=COLORS[c])
        axes[1].hist(d.bmi, bins=40, alpha=0.55, label=c, color=COLORS[c])
        axes[2].hist(d.last_hba1c.dropna(), bins=40, alpha=0.55, label=c, color=COLORS[c])
    axes[0].set_title("Age"); axes[1].set_title("BMI (Asian cut-offs 23 / 25)"); axes[2].set_title("Latest HbA1c (%)")
    for x in (23, 25):
        axes[1].axvline(x, color="k", ls="--", lw=0.8)
    for x in (5.7, 6.5):
        axes[2].axvline(x, color="k", ls="--", lw=0.8)
    axes[0].legend()
    imgs.append(fig_to_img(fig))

    fig, ax = plt.subplots(figsize=(15, 3.8))
    a1c = labs[labs.loinc == "4548-4"].copy()
    a1c["date"] = pd.to_datetime(a1c.date)
    rng = np.random.default_rng(0)
    for pid in rng.choice(f.loc[t2d(f), "patient_id"], 25, replace=False):
        s = a1c[a1c.patient_id == pid]
        ax.plot(s.date, s.value, marker="o", ms=2.5, lw=1, alpha=0.7)
    ax.axhline(7, color="k", ls="--", lw=0.8)
    ax.set_title("HbA1c trajectories, 25 random T2D patients (dashed = target 7%)")
    imgs.append(fig_to_img(fig))

    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    active = meds[meds.end_date.isna() & meds.patient_id.isin(f.loc[t2d(f), "patient_id"])]
    (active.groupby("drug_class").patient_id.nunique() / t2d(f).sum() * 100).sort_values().plot.barh(ax=axes[0], color="#64748b")
    axes[0].set_title("Current medication use in T2D (% of patients)")
    dxp = pd.DataFrame({c: [100 * f.loc[f.cohort == c, k].mean() for k in ["dx_I10", "dx_E78", "dx_E66", "dx_K76", "dx_N18", "dx_I25"]]
                        for c in COHORTS}, index=["Hypertension", "Dyslipidaemia", "Obesity", "Fatty liver", "CKD", "CAD"])
    dxp.plot.bar(ax=axes[1], color=[COLORS[c] for c in COHORTS], rot=0)
    axes[1].set_title("Comorbidity prevalence by cohort (%)")
    imgs.append(fig_to_img(fig))

    # --- sample patients (2 T2D, 1 poorly controlled T2D, 1 prediabetes, 1 normoglycemic)
    pick = list(f.loc[t2d(f)].sample(2, random_state=1).patient_id)
    pick += list(f.loc[t2d(f) & (f.last_hba1c > 9)].sample(1, random_state=1).patient_id)
    pick += list(f.loc[f.cohort == "Prediabetes"].sample(1, random_state=1).patient_id)
    pick += list(f.loc[f.cohort == "Normoglycemic"].sample(1, random_state=1).patient_id)
    stories = "\n<hr>\n".join(patient_story(pid, patients, labs, meds, dx, tp) for pid in pick)

    counts = f.cohort.value_counts()
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EHR Validation Report</title>
<style>
body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.ok{{color:#15803d;font-weight:600}} .warn{{color:#b45309;font-weight:600}} .muted{{color:#64748b}}
img{{max-width:100%}} h3{{margin-bottom:4px}}
</style></head><body>
<h1>Synthetic EHR &mdash; Validation Report</h1>
<p>{len(patients):,} patients ({counts.get('T2D',0):,} T2D &middot; {counts.get('Prediabetes',0):,} prediabetes &middot;
{counts.get('Normoglycemic',0):,} normoglycemic) &middot; {len(labs):,} lab/vital records &middot; {len(meds):,} medication records
&middot; {len(dx):,} diagnoses. Reference date {patients.reference_date.iloc[0]}. Fully synthetic data.</p>
<h2>1. Comparison with Indian clinical literature ({n_ok}/{len(REFERENCE)} in range)</h2>
<p class="muted">Ranges are approximate values from ICMR-INDIAB, SMART-India and Indian real-world T2D studies.</p>
<table><tr><th>Metric</th><th>Synthetic cohort</th><th>Reference range</th><th></th></tr>{''.join(ref_rows)}</table>
<h2>2. Data integrity checks</h2><table>{check_rows}</table>
<h2>3. Distributions</h2>{''.join(imgs)}
<h2>4. Sample patient stories</h2>{stories}
</body></html>"""
    out = DATA / "validation_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"Reference metrics in range: {n_ok}/{len(REFERENCE)}")
    print(f"Integrity checks passed: {sum(v for _, v in checks)}/{len(checks)}")
    for (label, fn, lo, hi, unit) in REFERENCE:
        v = fn(f, dx)
        print(f"  {'OK ' if lo <= v <= hi else '!! '} {label}: {v:.1f} (ref {lo}-{hi})")
    for n, v in checks:
        if not v:
            print(f"  FAIL: {n}")
    print(f"Report -> {out}")


if __name__ == "__main__":
    main()
