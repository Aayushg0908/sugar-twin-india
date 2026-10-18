"""
Synthetic EHR generator for the Type 2 Diabetes Digital Twin.

Builds a cohort of Indian adults (T2D / prediabetes / normoglycemic) with
linked demographics, diagnoses, labs + vitals, medications and genetic
markers. Every patient also gets hidden physiological "twin parameters"
(insulin sensitivity, beta-cell function, ...). The same parameters will
drive the wearable / CGM simulator, so both data streams describe the
same virtual body.

Clinical logic, not random numbers:
  * BMI, activity, age and genetics -> insulin sensitivity / beta-cell function
  * physiology -> untreated HbA1c; medicines x adherence -> observed HbA1c
  * doctors escalate therapy when HbA1c > 7 (with realistic clinical inertia)
  * diabetes duration + glycaemia + BP -> kidney decline and complications

All data is fully synthetic. No real patient information is used.

Usage:
    python src/generate_ehr.py --n 5000 --seed 42 --out data/ehr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2026-09-30")  # "today" in the simulation
WINDOW_MONTHS = 36                     # recorded lab history (3 years)
DAYS_PER_MONTH = 30.44

COHORT_MIX = {"T2D": 0.60, "Prediabetes": 0.20, "Normoglycemic": 0.20}

# Approximate population shares and dietary patterns by region
REGIONS = {"North": 0.24, "South": 0.22, "East": 0.18, "West": 0.20, "Central": 0.11, "Northeast": 0.05}
STAPLE = {"North": "wheat", "South": "rice", "East": "rice", "West": "mixed", "Central": "wheat", "Northeast": "rice"}
VEG_RATE = {"North": 0.45, "South": 0.25, "East": 0.10, "West": 0.45, "Central": 0.45, "Northeast": 0.05}
MEAL_CARBS_G = {"rice": 85, "wheat": 70, "mixed": 75}

# Glycaemic band each cohort must stay inside (HbA1c %, ADA / ICMR criteria)
A1C_BAND = {"Normoglycemic": (4.5, 5.6), "Prediabetes": (5.7, 6.4)}

LOINC = {
    "hba1c":  ("4548-4",  "HbA1c", "%", "laboratory"),
    "fpg":    ("1558-6",  "Fasting plasma glucose", "mg/dL", "laboratory"),
    "ppg":    ("1521-4",  "Post-prandial glucose (2h)", "mg/dL", "laboratory"),
    "tc":     ("2093-3",  "Total cholesterol", "mg/dL", "laboratory"),
    "ldl":    ("13457-7", "LDL cholesterol", "mg/dL", "laboratory"),
    "hdl":    ("2085-9",  "HDL cholesterol", "mg/dL", "laboratory"),
    "tg":     ("2571-8",  "Triglycerides", "mg/dL", "laboratory"),
    "creat":  ("2160-0",  "Serum creatinine", "mg/dL", "laboratory"),
    "egfr":   ("98979-8", "eGFR (CKD-EPI 2021)", "mL/min/1.73m2", "laboratory"),
    "uacr":   ("9318-7",  "Urine albumin/creatinine ratio", "mg/g", "laboratory"),
    "alt":    ("1742-6",  "ALT", "U/L", "laboratory"),
    "sbp":    ("8480-6",  "Systolic blood pressure", "mmHg", "vital-signs"),
    "dbp":    ("8462-4",  "Diastolic blood pressure", "mmHg", "vital-signs"),
    "hr":     ("8867-4",  "Heart rate", "bpm", "vital-signs"),
    "weight": ("29463-7", "Body weight", "kg", "vital-signs"),
    "bmi":    ("39156-5", "BMI", "kg/m2", "vital-signs"),
    "waist":  ("8280-0",  "Waist circumference", "cm", "vital-signs"),
}

# name, class, dose, HbA1c lowering at full adherence (%-points)
GLUCOSE_DRUGS = {
    "metformin_500":  ("Metformin", "Biguanide", "500 mg BD", 0.8),
    "metformin_1000": ("Metformin", "Biguanide", "1000 mg BD", 1.0),
    "glimepiride":    ("Glimepiride", "Sulfonylurea", "2 mg OD", 0.8),
    "gliclazide":     ("Gliclazide MR", "Sulfonylurea", "60 mg OD", 0.8),
    "sitagliptin":    ("Sitagliptin", "DPP-4 inhibitor", "100 mg OD", 0.5),
    "vildagliptin":   ("Vildagliptin", "DPP-4 inhibitor", "50 mg BD", 0.5),
    "teneligliptin":  ("Teneligliptin", "DPP-4 inhibitor", "20 mg OD", 0.5),
    "dapagliflozin":  ("Dapagliflozin", "SGLT2 inhibitor", "10 mg OD", 0.5),
    "empagliflozin":  ("Empagliflozin", "SGLT2 inhibitor", "10 mg OD", 0.5),
}
SU_CHOICES = ["glimepiride", "gliclazide"]
DPP4_CHOICES = ["sitagliptin", "vildagliptin", "teneligliptin"]
SGLT2_CHOICES = ["dapagliflozin", "empagliflozin"]

# Simplified T2D-associated SNPs: (gene, rsID, risk allele, base risk-allele freq, log-OR)
# Frequencies are illustrative approximations for South Asian populations.
SNPS = [
    ("TCF7L2", "rs7903146", "T", 0.28, 0.35),
    ("FTO", "rs9939609", "A", 0.32, 0.15),
    ("KCNQ1", "rs2237892", "C", 0.70, 0.25),
    ("SLC30A8", "rs13266634", "C", 0.75, 0.12),
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------------------
# 1. Baseline patients (vectorised)
# --------------------------------------------------------------------------
def sample_patients(n: int, rng: np.random.Generator) -> pd.DataFrame:
    counts = {k: int(round(v * n)) for k, v in COHORT_MIX.items()}
    counts["T2D"] += n - sum(counts.values())
    cohort = rng.permutation(np.concatenate([[k] * c for k, c in counts.items()]))
    t2d, pre = cohort == "T2D", cohort == "Prediabetes"

    def by_cohort(t2d_v, pre_v, normo_v):
        return np.where(t2d, t2d_v, np.where(pre, pre_v, normo_v))

    df = pd.DataFrame({"patient_id": [f"P{i:05d}" for i in range(1, n + 1)], "cohort": cohort})

    age = rng.normal(by_cohort(53, 44, 37), by_cohort(11, 12, 12))
    df["age"] = np.clip(age, by_cohort(28, 22, 20), by_cohort(85, 80, 75)).round().astype(int)
    df["sex"] = np.where(rng.random(n) < 0.52, "M", "F")
    male = (df["sex"] == "M").to_numpy()
    df["region"] = rng.choice(list(REGIONS), n, p=list(REGIONS.values()))
    df["setting"] = np.where(rng.random(n) < by_cohort(0.60, 0.50, 0.45), "Urban", "Rural")
    urban = (df["setting"] == "Urban").to_numpy()

    df["height_cm"] = np.where(male, rng.normal(165.5, 6.5, n), rng.normal(152.5, 6.0, n)).round(1)
    bmi = rng.normal(by_cohort(26.3, 25.2, 22.6), 3.8) + 1.0 * urban
    df["bmi"] = np.clip(bmi, 16, 45).round(1)
    df["weight_kg"] = (df["bmi"] * (df["height_cm"] / 100) ** 2).round(1)
    waist = 38 + 1.95 * df["bmi"] + 4 * male + rng.normal(0, 4, n)
    df["waist_cm"] = waist.round(1)

    df["diet"] = np.where(rng.random(n) < df["region"].map(VEG_RATE), "Vegetarian", "Non-vegetarian")
    df["staple"] = df["region"].map(STAPLE)

    act = []
    for c in cohort:
        p = {"T2D": [0.55, 0.35, 0.10], "Prediabetes": [0.45, 0.40, 0.15], "Normoglycemic": [0.30, 0.45, 0.25]}[c]
        act.append(rng.choice(["Sedentary", "Moderate", "Active"], p=p))
    df["activity_level"] = act

    df["smoking"] = [rng.choice(["Current", "Former", "Never"], p=[0.18, 0.10, 0.72] if m else [0.02, 0.01, 0.97])
                     for m in male]
    df["alcohol"] = [rng.choice(["None", "Occasional", "Regular"], p=[0.65, 0.25, 0.10] if m else [0.97, 0.025, 0.005])
                     for m in male]
    df["family_history_diabetes"] = rng.random(n) < by_cohort(0.55, 0.40, 0.22)
    df["family_history_cvd"] = rng.random(n) < 0.25

    # Diabetes duration (years); age at diagnosis >= 25
    dur = np.minimum(rng.gamma(1.6, 4.0, n), df["age"] - 25)
    df["diabetes_duration_yrs"] = np.where(t2d, np.maximum(dur, 0.1), np.nan).round(1)
    return df


def sample_genetics(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    shift = df["cohort"].map({"T2D": 0.04, "Prediabetes": 0.02, "Normoglycemic": 0.0}).to_numpy()
    out = pd.DataFrame({"patient_id": df["patient_id"]})
    score = np.zeros(n)
    for gene, rsid, allele, freq, log_or in SNPS:
        k = rng.binomial(2, np.clip(freq + shift, 0, 1))  # Hardy-Weinberg
        out[f"{gene}_{rsid}_{allele}_count"] = k
        score += k * log_or
    background = rng.normal(shift * 10, 1.0)  # polygenic background
    prs = score / np.std(score) + background
    out["polygenic_risk_score_z"] = ((prs - prs.mean()) / prs.std()).round(2)
    return out


def sample_twin_params(df: pd.DataFrame, gen: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Hidden physiology. Not visible to the prediction model; it drives both data streams."""
    n = len(df)
    c = df["cohort"].to_numpy()
    t2d, pre = c == "T2D", c == "Prediabetes"
    act_f = df["activity_level"].map({"Sedentary": 0.90, "Moderate": 1.0, "Active": 1.12}).to_numpy()
    bmi, age = df["bmi"].to_numpy(), df["age"].to_numpy()
    dur = np.nan_to_num(df["diabetes_duration_yrs"].to_numpy())
    tcf = gen["TCF7L2_rs7903146_T_count"].to_numpy()

    si_base = np.where(t2d, 0.48, np.where(pre, 0.70, 1.0))
    si = si_base * np.exp(-0.045 * (bmi - 23)) * act_f * np.exp(-0.004 * (age - 40)) * rng.lognormal(0, 0.15, n)
    si = np.clip(si, 0.08, 2.0)

    beta = np.where(t2d, rng.normal(0.58, 0.10, n) - 0.02 * dur - 0.03 * tcf - 0.08 * (bmi < 23),
                    np.where(pre, rng.normal(0.70, 0.08, n), rng.normal(0.90, 0.07, n)))
    beta = np.clip(beta, 0.08, 1.0)

    poor = rng.random(n) < np.where(t2d, 0.30, 0.12)
    adherence = np.where(poor, rng.beta(2, 4, n), rng.beta(8, 2, n))

    act = df["activity_level"]
    tp = pd.DataFrame({"patient_id": df["patient_id"]})
    tp["insulin_sensitivity"] = si.round(3)
    tp["beta_cell_function"] = beta.round(3)
    tp["a1c_offset"] = rng.normal(np.where(t2d, 0.6, np.where(pre, -0.25, 0.0)), np.where(t2d, 1.3, 0.2)).round(3)
    tp["carb_sensitivity_mgdl_per_10g"] = (3.0 / np.sqrt(si * beta)).round(2)
    tp["dawn_phenomenon_mgdl"] = np.clip(rng.normal(np.where(t2d, 15, np.where(pre, 8, 3)), np.where(t2d, 6, 3)), 0, None).round(1)
    hr = 72 + 0.4 * (bmi - 23) - act.map({"Sedentary": 0, "Moderate": 3, "Active": 7}).to_numpy() \
        + 3 * t2d - 2 * (df["sex"] == "M") + rng.normal(0, 5, n)
    tp["resting_hr_bpm"] = np.clip(hr, 52, 100).round(1)
    hrv = 42 - 0.45 * (age - 40) - 6 * t2d - 3 * pre - 3 * (act == "Sedentary") + rng.normal(0, 7, n)
    tp["hrv_rmssd_ms"] = np.clip(hrv, 10, 90).round(1)
    tp["sleep_duration_h"] = np.clip(rng.normal(6.8, 0.7, n), 4.5, 9).round(2)
    tp["sleep_efficiency"] = np.clip(rng.normal(0.86, 0.05, n) - 0.03 * (bmi >= 27.5), 0.6, 0.97).round(3)
    tp["stress_reactivity"] = rng.lognormal(0, 0.25, n).round(2)
    steps = act.map({"Sedentary": 3500, "Moderate": 6500, "Active": 9500}).to_numpy() * rng.lognormal(0, 0.25, n)
    tp["daily_steps_baseline"] = steps.round(-1).astype(int)
    tp["meal_carbs_g"] = (df["staple"].map(MEAL_CARBS_G).to_numpy() + rng.normal(0, 12, n)).round(0)
    tp["med_adherence"] = adherence.round(3)
    tp["weight_trend_kg_per_yr"] = rng.normal(0.3, 1.0, n).round(2)
    return tp


# --------------------------------------------------------------------------
# 2. Comorbidity "truth" (blood pressure, lipids, kidney, liver)
# --------------------------------------------------------------------------
def sample_comorbidity_state(df: pd.DataFrame, tp: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    c = df["cohort"].to_numpy()
    t2d, pre = c == "T2D", c == "Prediabetes"
    male = (df["sex"] == "M").to_numpy()
    age, bmi = df["age"].to_numpy(), df["bmi"].to_numpy()
    dur = np.nan_to_num(df["diabetes_duration_yrs"].to_numpy())

    s = pd.DataFrame({"patient_id": df["patient_id"]})
    sbp = 120 + 0.55 * (age - 30) + 0.9 * (bmi - 23) + 4 * t2d + 2 * pre + 3 * male + rng.normal(0, 14, n)
    s["sbp_untreated"] = sbp
    s["dbp_offset"] = rng.normal(0, 5, n)
    s["has_htn"] = sbp >= 140
    s["htn_diagnosed"] = s["has_htn"] & (rng.random(n) < np.where(t2d, 0.85, 0.55))

    s["ldl_base"] = np.clip(rng.normal(118, 32, n) + 0.5 * (bmi - 23), 40, 260)
    s["tg_base"] = 130 * np.where(t2d, 1.3, 1.0) * (1 + 0.03 * (bmi - 23)) * \
        np.where(df["alcohol"] == "Regular", 1.2, 1.0) * rng.lognormal(0, 0.35, n)
    s["hdl_base"] = np.clip(np.where(male, 40, 46) - 0.6 * (bmi - 23) - 3 * t2d + rng.normal(0, 8, n), 20, 90)

    # Untreated glycaemic burden used for complication risk
    a1c_u = np.clip(5.1 + 1.6 * np.log(1 / (tp["insulin_sensitivity"] * tp["beta_cell_function"])) + tp["a1c_offset"], 4.5, 14)
    s["a1c_untreated_now"] = a1c_u
    dm_decline = np.where(t2d, 0.8 + 0.4 * np.maximum(0, a1c_u - 8) + 0.8 * s["has_htn"], 0.0)
    s["egfr_decline_extra"] = dm_decline
    s["egfr_now"] = np.clip(112 - 0.9 * np.maximum(age - 25, 0) - dm_decline * dur + rng.normal(0, 10, n), 8, 130)
    uacr_med = np.where(t2d, 8, 6) * np.exp(0.35 * np.maximum(0, a1c_u - 7) * t2d) * np.where(s["has_htn"], 1.8, 1.0) * (1 + 0.06 * dur)
    s["uacr_now"] = uacr_med * rng.lognormal(0, 0.9, n)

    nafld_p = sigmoid(-2.3 + 0.25 * (bmi - 23) + 1.0 * t2d + 0.5 * pre + 0.004 * (s["tg_base"] - 150))
    s["has_nafld"] = rng.random(n) < nafld_p
    s["alt_base"] = rng.normal(24, 7, n).clip(8) + s["has_nafld"] * rng.lognormal(3.0, 0.4, n)

    smoker = (df["smoking"] == "Current").to_numpy()
    s["has_neuropathy"] = t2d & (rng.random(n) < sigmoid(-2.5 + 0.12 * dur + 0.35 * (a1c_u - 8) + 0.03 * (age - 50)))
    s["has_retinopathy"] = t2d & (rng.random(n) < sigmoid(-3.2 + 0.13 * dur + 0.4 * (a1c_u - 8) + 0.4 * s["has_htn"]))
    cad_logit = -5.2 + 0.07 * (age - 40) + 0.7 * smoker + 0.6 * t2d + 0.5 * s["has_htn"] + 0.4 * df["family_history_cvd"] + 0.03 * dur + 0.4 * male
    s["has_cad"] = rng.random(n) < sigmoid(cad_logit)
    s["has_hypothyroid"] = rng.random(n) < np.where(male, 0.03, 0.10)
    return s


# --------------------------------------------------------------------------
# 3. Per-patient timeline simulation
# --------------------------------------------------------------------------
def egfr_to_creatinine(egfr, age, female):
    """Invert the CKD-EPI 2021 (race-free) equation."""
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    r = egfr / (142 * 0.9938 ** age * (1.012 if female else 1.0))
    ratio = r ** (1 / a) if r >= 1 else r ** (-1 / 1.2)
    return ratio * k


def creatinine_to_egfr(scr, age, female):
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    x = scr / k
    return 142 * min(x, 1) ** a * max(x, 1) ** -1.2 * 0.9938 ** age * (1.012 if female else 1.0)


def fpg_from_a1c(a1c):
    return 18.5 * a1c - 8 + max(0.0, a1c - 6.5) * 12


class PatientSim:
    def __init__(self, p: dict, tp: dict, st: dict, rng: np.random.Generator):
        self.p, self.tp, self.st, self.rng = p, tp, st, rng
        self.pid = p["patient_id"]
        self.female = p["sex"] == "F"
        self.labs, self.meds, self.dx = [], [], []
        self.glucose_regimen = {}   # slot -> open med record (dict)
        self.static_meds = []       # (start_day, kind, effect) for BP / statin
        # Clinical inertia and access/cost limits on how far therapy is intensified
        self.escalation_prob = 0.16 if p["setting"] == "Urban" else 0.10
        self.max_glucose_drugs = int(rng.choice([2, 3, 4, 5], p=[0.30, 0.35, 0.27, 0.08]))
        self._metformin_stopped = rng.random() < 0.08  # GI intolerance

    # ---------------- records ----------------
    def add_dx(self, code, desc, day):
        self.dx.append({"patient_id": self.pid, "icd10": code, "description": desc, "onset_day": day})

    def start_med(self, slot, name, cls, dose, indication, day, effect=0.0):
        rec = {"patient_id": self.pid, "drug": name, "drug_class": cls, "dose": dose,
               "indication": indication, "start_day": day, "end_day": np.nan, "_effect": effect}
        self.meds.append(rec)
        if slot:
            self.glucose_regimen[slot] = rec
        return rec

    def stop_med(self, slot, day):
        rec = self.glucose_regimen.pop(slot, None)
        if rec is not None:
            rec["end_day"] = day

    def lab(self, key, day, value, decimals=1):
        if value is None or not np.isfinite(value):
            return
        loinc, name, unit, cat = LOINC[key]
        self.labs.append({"patient_id": self.pid, "day": day, "category": cat, "loinc": loinc,
                          "test": name, "value": round(float(value), decimals), "unit": unit})

    # ---------------- physiology over time ----------------
    def years_back(self, m):
        return -m / 12.0

    def weight_at(self, m):
        return self.p["weight_kg"] - self.tp["weight_trend_kg_per_yr"] * self.years_back(m)

    def bmi_at(self, m):
        return self.weight_at(m) / (self.p["height_cm"] / 100) ** 2

    def untreated_a1c(self, m):
        yb = self.years_back(m)
        cohort = self.p["cohort"]
        beta_drift = {"T2D": 0.02, "Prediabetes": 0.01, "Normoglycemic": 0.0}[cohort]
        beta = min(1.0, self.tp["beta_cell_function"] + beta_drift * yb)
        si = self.tp["insulin_sensitivity"] * np.exp(-0.045 * (self.bmi_at(m) - self.p["bmi"]))
        a1c = 5.1 + 1.6 * np.log(1 / (si * beta)) + self.tp["a1c_offset"]
        if cohort in A1C_BAND:
            return float(np.clip(a1c, *A1C_BAND[cohort]))
        return float(np.clip(a1c, 6.6, 14.0))

    def egfr_at(self, m):
        yb = self.years_back(m)
        return min(125.0, self.st["egfr_now"] + (0.9 + self.st["egfr_decline_extra"]) * yb)

    def static_effect(self, kind, day):
        return sum(e for d, k, e in self.static_meds if k == kind and d <= day)

    def adherence_now(self):
        return float(np.clip(self.tp["med_adherence"] + self.rng.normal(0, 0.05), 0, 1))

    # ---------------- glucose therapy ----------------
    def glucose_effect(self):
        return sum(r["_effect"] for r in self.glucose_regimen.values())

    def escalate(self, a1c, egfr, day):
        reg, rng = self.glucose_regimen, self.rng
        if "biguanide" not in reg and egfr >= 30 and not getattr(self, "_metformin_stopped", False):
            self.start_med("biguanide", *GLUCOSE_DRUGS["metformin_500"][:3], "Type 2 diabetes", day, GLUCOSE_DRUGS["metformin_500"][3])
        elif "biguanide" in reg and reg["biguanide"]["dose"] == "500 mg BD":
            self.stop_med("biguanide", day)
            self.start_med("biguanide", *GLUCOSE_DRUGS["metformin_1000"][:3], "Type 2 diabetes", day, GLUCOSE_DRUGS["metformin_1000"][3])
        elif "su" not in reg and "dpp4" not in reg:
            if rng.random() < 0.55:
                key = rng.choice(SU_CHOICES); slot = "su"
            else:
                key = rng.choice(DPP4_CHOICES); slot = "dpp4"
            self.start_med(slot, *GLUCOSE_DRUGS[key][:3], "Type 2 diabetes", day, GLUCOSE_DRUGS[key][3])
        elif "sglt2" not in reg and egfr >= 25:
            key = rng.choice(SGLT2_CHOICES)
            self.start_med("sglt2", *GLUCOSE_DRUGS[key][:3], "Type 2 diabetes", day, GLUCOSE_DRUGS[key][3])
        elif "su" not in reg or "dpp4" not in reg:
            slot = "dpp4" if "su" in reg else "su"
            key = rng.choice(DPP4_CHOICES if slot == "dpp4" else SU_CHOICES)
            self.start_med(slot, *GLUCOSE_DRUGS[key][:3], "Type 2 diabetes", day, GLUCOSE_DRUGS[key][3])
        elif "insulin" not in reg and a1c > 8.5:
            self.start_med("insulin", "Insulin glargine", "Basal insulin", "10 units HS", "Type 2 diabetes", day, 1.2)
        elif "insulin" in reg:
            units = int(reg["insulin"]["dose"].split()[0])
            if units >= 40:
                return False
            units += 4
            self.stop_med("insulin", day)
            self.start_med("insulin", "Insulin glargine", "Basal insulin", f"{units} units HS", "Type 2 diabetes",
                           day, min(2.2, 1.2 + 0.03 * (units - 10)))
        else:
            return False
        return True

    def safety_review(self, a1c, egfr, day):
        reg = self.glucose_regimen
        if "biguanide" in reg and egfr < 30:
            self.stop_med("biguanide", day)
            self._metformin_stopped = True
        if "sglt2" in reg and egfr < 20:
            self.stop_med("sglt2", day)
        if "su" in reg and a1c < 6.2 and self.rng.random() < 0.5:
            self.stop_med("su", day)  # hypoglycaemia risk

    # ---------------- static diagnoses & meds ----------------
    def setup_static(self):
        p, st, rng = self.p, self.st, self.rng
        age = p["age"]

        def past_day(max_years):
            return -rng.uniform(0, max(0.2, max_years)) * 365.25

        if p["cohort"] == "T2D":
            self.dx_day = -p["diabetes_duration_yrs"] * 365.25
            self.add_dx("E11.9", "Type 2 diabetes mellitus", self.dx_day)
        elif p["cohort"] == "Prediabetes":
            self.add_dx("R73.03", "Prediabetes", past_day(4))

        if st["htn_diagnosed"]:
            d = past_day(min(12, age - 28))
            self.add_dx("I10", "Essential hypertension", d)
            self.start_med(None, "Telmisartan", "ARB", "40 mg OD", "Hypertension", d + 30)
            self.static_meds.append((d + 30, "bp", 10.0))
            if st["sbp_untreated"] - 10 >= 145 and rng.random() < 0.6:
                d2 = d + rng.uniform(90, max(120, -d))
                d2 = min(d2, -15)
                self.start_med(None, "Amlodipine", "Calcium channel blocker", "5 mg OD", "Hypertension", d2)
                self.static_meds.append((d2, "bp", 8.0))

        dyslip = st["ldl_base"] >= 130 or st["tg_base"] >= 150 or st["hdl_base"] < (50 if self.female else 40)
        if dyslip and rng.random() < (0.85 if p["cohort"] == "T2D" else 0.5):
            d = past_day(6)
            self.add_dx("E78.5", "Dyslipidaemia", d)
        on_statin = (p["cohort"] == "T2D" and age >= 40 and rng.random() < 0.55) or \
                    (st["ldl_base"] >= 160 and rng.random() < 0.4) or st["has_cad"]
        if on_statin:
            d = past_day(p["diabetes_duration_yrs"] if p["cohort"] == "T2D" else 5)
            drug = rng.choice(["Atorvastatin|20 mg OD", "Atorvastatin|10 mg OD", "Rosuvastatin|10 mg OD"])
            name, dose = drug.split("|")
            self.start_med(None, name, "Statin", dose, "Dyslipidaemia / CV risk", d)
            self.static_meds.append((d, "statin", 0.35))

        if p["bmi"] >= 25 and rng.random() < 0.6:
            self.add_dx("E66.9", "Obesity (Asian BMI cut-off)", past_day(5))
        if st["has_nafld"] and rng.random() < 0.6:
            self.add_dx("K76.0", "Non-alcoholic fatty liver disease", past_day(5))
        if st["has_cad"]:
            d = past_day(min(10, age - 35))
            self.add_dx("I25.10", "Coronary artery disease", d)
            self.start_med(None, "Aspirin", "Antiplatelet", "75 mg OD", "Coronary artery disease", d)
        if st["has_hypothyroid"]:
            d = past_day(8)
            self.add_dx("E03.9", "Hypothyroidism", d)
            self.start_med(None, "Levothyroxine", "Thyroid hormone", rng.choice(["25 mcg OD", "50 mcg OD", "75 mcg OD"]), "Hypothyroidism", d)
        if p["cohort"] == "T2D":
            dur_days = p["diabetes_duration_yrs"] * 365.25
            if st["has_neuropathy"]:
                self.add_dx("E11.40", "T2DM with diabetic neuropathy", self.dx_day + rng.uniform(0.5, 1) * dur_days)
            if st["has_retinopathy"]:
                self.add_dx("E11.319", "T2DM with diabetic retinopathy", self.dx_day + rng.uniform(0.5, 1) * dur_days)

    # ---------------- visit record ----------------
    def record_visit(self, m, day, a1c, annual, avail):
        p, st, tp, rng = self.p, self.st, self.tp, self.rng
        adh = self.adherence_now()
        has = lambda prob: rng.random() < prob

        w = self.weight_at(m) + rng.normal(0, 0.6)
        bmi = w / (p["height_cm"] / 100) ** 2
        self.lab("weight", day, w)
        self.lab("bmi", day, bmi)
        if annual and has(0.6):
            self.lab("waist", day, p["waist_cm"] + (bmi - p["bmi"]) * 2.2 + rng.normal(0, 1.5))

        if has(0.95):
            sbp = st["sbp_untreated"] + 0.55 * (-self.years_back(m)) - self.static_effect("bp", day) * adh + rng.normal(0, 8)
            dbp = 0.5 * sbp + 12 + st["dbp_offset"] + rng.normal(0, 4)
            self.lab("sbp", day, sbp, 0)
            self.lab("dbp", day, dbp, 0)
            self.lab("hr", day, tp["resting_hr_bpm"] + rng.normal(0, 4), 0)

        if has(avail["hba1c"]):
            reported = a1c + rng.normal(0, 0.1)
            if p["cohort"] in A1C_BAND:
                reported = float(np.clip(reported, *A1C_BAND[p["cohort"]]))
            self.lab("hba1c", day, reported)
        if has(avail["fpg"]):
            self.lab("fpg", day, fpg_from_a1c(a1c) * rng.lognormal(0, 0.07), 0)
        if has(avail["ppg"]):
            self.lab("ppg", day, (fpg_from_a1c(a1c) * 1.35 + max(0, a1c - 6) * 10) * rng.lognormal(0, 0.09), 0)

        if annual:
            statin = self.static_effect("statin", day) * adh
            if has(0.9):
                ldl = st["ldl_base"] * (1 - statin) + rng.normal(0, 8)
                tg = st["tg_base"] * (1 + 0.06 * max(0, a1c - 7)) * (1 - 0.3 * statin) * rng.lognormal(0, 0.15)
                hdl = st["hdl_base"] + rng.normal(0, 3)
                self.lab("ldl", day, max(ldl, 25), 0)
                self.lab("tg", day, tg, 0)
                self.lab("hdl", day, hdl, 0)
                self.lab("tc", day, max(ldl, 25) + hdl + tg / 5, 0)
            if has(0.9):
                age_then = p["age"] - self.years_back(m)
                scr = egfr_to_creatinine(self.egfr_at(m), age_then, self.female) * rng.lognormal(0, 0.05)
                self.lab("creat", day, scr, 2)
                self.lab("egfr", day, creatinine_to_egfr(scr, age_then, self.female), 0)
            if has(avail["uacr"]):
                uacr = st["uacr_now"] * np.exp(-0.08 * self.years_back(m)) * rng.lognormal(0, 0.3)
                self.lab("uacr", day, uacr, 1)
            if has(0.8):
                self.lab("alt", day, st["alt_base"] * rng.lognormal(0, 0.2), 0)

    # ---------------- main loops ----------------
    def run(self):
        self.setup_static()
        rng = self.rng
        window_start = -(WINDOW_MONTHS - 1)
        cohort = self.p["cohort"]

        if cohort == "T2D":
            dx_m = int(round(-self.p["diabetes_duration_yrs"] * 12))
            a1c = self.untreated_a1c(dx_m)
            avail = {"hba1c": 0.85, "fpg": 0.9, "ppg": 0.7, "uacr": 0.6}
            last_change = dx_m
            # Treatment at diagnosis
            dx_day = dx_m * DAYS_PER_MONTH
            if self.egfr_at(dx_m) >= 30:
                self.escalate(a1c, self.egfr_at(dx_m), dx_day)
            if a1c >= 9:
                self.escalate(a1c, self.egfr_at(dx_m), dx_day)  # metformin dose up
                self.escalate(a1c, self.egfr_at(dx_m), dx_day)  # add 2nd agent
            if a1c >= 11:
                self.start_med("insulin", "Insulin glargine", "Basal insulin", "10 units HS", "Type 2 diabetes", dx_day, 1.2)

            visit_months = set()
            missed = 0.15
            first_window_visit_year = {}
            for m in range(dx_m, 1):
                if m > dx_m:
                    target = max(5.5, self.untreated_a1c(m) - self.glucose_effect() * self.adherence_now())
                    a1c += (target - a1c) * 0.35 + rng.normal(0, 0.06)
                if (m - dx_m) % 3 == 0:
                    day = m * DAYS_PER_MONTH + rng.uniform(-20, 20)
                    day = min(day, 0.0)
                    if m >= window_start and rng.random() > missed:
                        year_block = (m - window_start) // 12
                        annual = year_block not in first_window_visit_year
                        first_window_visit_year.setdefault(year_block, m)
                        self.record_visit(m, day, a1c, annual, avail)
                        visit_months.add(m)
                    if m > dx_m:
                        egfr = self.egfr_at(m)
                        self.safety_review(a1c, egfr, day)
                        if (a1c > 9.5 and "insulin" not in self.glucose_regimen and len(self.glucose_regimen) >= 2
                                and m - last_change >= 3 and rng.random() < 0.35):
                            self.start_med("insulin", "Insulin glargine", "Basal insulin", "10 units HS",
                                           "Type 2 diabetes", day, 1.0)
                            last_change = m
                        elif (a1c > 7.0 and m - last_change >= 3 and rng.random() < self.escalation_prob
                                and len(self.glucose_regimen) < self.max_glucose_drugs):
                            if self.escalate(a1c, egfr, day):
                                last_change = m
        else:
            interval, missed = (6, 0.25) if cohort == "Prediabetes" else (12, 0.30)
            avail = ({"hba1c": 0.7, "fpg": 0.95, "ppg": 0.5, "uacr": 0.2} if cohort == "Prediabetes"
                     else {"hba1c": 0.5, "fpg": 0.95, "ppg": 0.2, "uacr": 0.1})
            offset = rng.integers(0, interval)
            for m in range(window_start + offset, 1, interval):
                if rng.random() < missed:
                    continue
                day = min(m * DAYS_PER_MONTH + rng.uniform(-20, 20), 0.0)
                annual = interval == 12 or (m - window_start - offset) % 12 == 0
                a1c = self.untreated_a1c(m) + rng.normal(0, 0.05)
                self.record_visit(m, day, a1c, annual, avail)

        if not self.labs:
            m = -int(rng.integers(0, 6))
            a1c_now = a1c if cohort == "T2D" else self.untreated_a1c(m)
            self.record_visit(m, m * DAYS_PER_MONTH, a1c_now, True, avail)

        # Kidney diagnosis from the latest available labs
        egfr_vals = [r["value"] for r in self.labs if r["loinc"] == "98979-8"]
        uacr_vals = [r["value"] for r in self.labs if r["loinc"] == "9318-7"]
        last_egfr = egfr_vals[-1] if egfr_vals else None
        last_uacr = uacr_vals[-1] if uacr_vals else None
        persistent_alb = len(uacr_vals) >= 2 and min(uacr_vals[-2:]) >= 30
        if (last_egfr is not None and last_egfr < 60) or persistent_alb:
            stage = ("N18.1" if last_egfr is None or last_egfr >= 90 else "N18.2" if last_egfr >= 60 else
                     "N18.3" if last_egfr >= 30 else "N18.4" if last_egfr >= 15 else "N18.5")
            desc = {"N18.1": "CKD stage 1", "N18.2": "CKD stage 2", "N18.3": "CKD stage 3",
                    "N18.4": "CKD stage 4", "N18.5": "CKD stage 5"}[stage]
            onset = next((r["day"] for r in self.labs if (r["loinc"] == "98979-8" and r["value"] < 60) or
                          (r["loinc"] == "9318-7" and r["value"] >= 30)), -30)
            self.add_dx(stage, desc, onset)
        return self


# --------------------------------------------------------------------------
# 4. Assembly
# --------------------------------------------------------------------------
def day_to_date(days: pd.Series) -> pd.Series:
    return (REF_DATE + pd.to_timedelta(days.round(), unit="D")).dt.date


def build_features(patients, labs, meds, dx, gen) -> pd.DataFrame:
    """One wide row per patient with the latest values: convenient input for ML."""
    latest = (labs.sort_values("date").groupby(["patient_id", "test"])["value"].last().unstack())
    latest.columns = ["last_" + c.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
                      .replace("-", "_").replace(",", "") for c in latest.columns]
    a1c = labs[labs["loinc"] == "4548-4"].sort_values("date").groupby("patient_id")["value"]
    trend = a1c.agg(["first", "last", "count", "mean"]).rename(columns={
        "first": "hba1c_first", "last": "hba1c_last", "count": "hba1c_n", "mean": "hba1c_mean"})
    trend["hba1c_change"] = trend["hba1c_last"] - trend["hba1c_first"]

    active = meds[meds["end_date"].isna()]
    med_flags = pd.crosstab(active["patient_id"], active["drug_class"]).clip(upper=1)
    med_flags.columns = ["on_" + c.lower().replace(" ", "_").replace("-", "").replace("/", "_") for c in med_flags.columns]
    med_flags["n_glucose_drugs"] = active[active["indication"] == "Type 2 diabetes"].groupby("patient_id").size()

    dx_flags = pd.crosstab(dx["patient_id"], dx["icd10"].str[:3]).clip(upper=1)
    dx_flags.columns = ["dx_" + c for c in dx_flags.columns]

    out = patients.set_index("patient_id")
    for t in (latest, trend[["hba1c_mean", "hba1c_change", "hba1c_n"]], med_flags, dx_flags,
              gen.set_index("patient_id")):
        out = out.join(t)
    flag_cols = [c for c in out.columns if c.startswith(("on_", "dx_"))] + ["n_glucose_drugs"]
    out[flag_cols] = out[flag_cols].fillna(0).astype(int)
    return out.reset_index()


def generate(n: int, seed: int, out_dir: Path):
    rng = np.random.default_rng(seed)
    patients = sample_patients(n, rng)
    genetics = sample_genetics(patients, rng)
    twin = sample_twin_params(patients, genetics, rng)
    state = sample_comorbidity_state(patients, twin, rng)

    # Hidden physiology adjustments that depend on complications
    twin.loc[state["has_neuropathy"], "hrv_rmssd_ms"] = (twin.loc[state["has_neuropathy"], "hrv_rmssd_ms"] - 5).clip(lower=8)

    labs, meds, dx = [], [], []
    tp_rows = twin.to_dict("records")
    st_rows = state.to_dict("records")
    for i, p in enumerate(patients.to_dict("records")):
        sim = PatientSim(p, tp_rows[i], st_rows[i], rng).run()
        labs += sim.labs
        meds += sim.meds
        dx += sim.dx

    labs = pd.DataFrame(labs)
    labs["date"] = day_to_date(labs.pop("day"))
    labs = labs[["patient_id", "date", "category", "loinc", "test", "value", "unit"]].sort_values(["patient_id", "date"])

    meds = pd.DataFrame(meds).drop(columns="_effect")
    meds["start_date"] = day_to_date(meds.pop("start_day"))
    end = meds.pop("end_day")
    meds["end_date"] = day_to_date(end.fillna(0)).where(end.notna(), None)
    meds = meds.sort_values(["patient_id", "start_date"])

    dx = pd.DataFrame(dx)
    dx["onset_date"] = day_to_date(dx.pop("onset_day"))
    dx = dx.sort_values(["patient_id", "onset_date"])

    patients["reference_date"] = REF_DATE.date()
    features = build_features(patients, labs, meds, dx, genetics)

    out_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(out_dir / "patients.csv", index=False)
    dx.to_csv(out_dir / "diagnoses.csv", index=False)
    labs.to_csv(out_dir / "labs.csv", index=False)
    meds.to_csv(out_dir / "medications.csv", index=False)
    genetics.to_csv(out_dir / "genetics.csv", index=False)
    twin.to_csv(out_dir / "twin_params.csv", index=False)
    features.to_csv(out_dir / "patient_features.csv", index=False)

    print(f"Generated {n} patients -> {out_dir}")
    for name, d in [("patients", patients), ("diagnoses", dx), ("labs", labs), ("medications", meds),
                    ("genetics", genetics), ("twin_params", twin), ("patient_features", features)]:
        print(f"  {name:<17} {len(d):>8,} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "ehr")
    args = ap.parse_args()
    generate(args.n, args.seed, args.out)
