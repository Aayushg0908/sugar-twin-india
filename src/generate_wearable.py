"""
Wearable + CGM simulator for the Type 2 Diabetes Digital Twin (Dataset 2).

For every patient in the synthetic EHR (Dataset 1) this simulates 14 days of
5-minute data from a continuous glucose monitor and a smartwatch:
glucose, heart rate, steps and sleep stages, plus an event log of meals,
medication doses and exercise.

The simulation is driven by each patient's hidden twin parameters
(insulin sensitivity, beta-cell function, carb sensitivity, adherence, ...)
and calibrated so the mean CGM glucose matches their latest lab HbA1c, so
both data streams describe the same virtual body.

Glucose model (per patient, 5-minute steps)
    gut:      carbs -> 2-compartment absorption (fast / medium / slow GI)
    excursion X:  dX/dt = -X/tau*(1+exercise) + c*Ra(t) - exercise uptake
    glucose  = baseline(day) + dawn rise + X + stress - hypo pressure + slow drift
    CGM      = glucose with sensor noise, compression lows, gaps, warm-up
    tau, c come from insulin sensitivity x beta-cell function.

Usage:
    python src/generate_wearable.py              # all patients, 14 days
    python src/generate_wearable.py --limit 200  # quick test
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
EHR = ROOT / "data" / "ehr"
OUT = ROOT / "data" / "wearable"

REF_DATE = pd.Timestamp("2026-09-30")
N_DAYS = 14
STEP_MIN = 5
BPD = 24 * 60 // STEP_MIN             # bins per day (288)
T = N_DAYS * BPD                       # total bins (4032)
START = REF_DATE - pd.Timedelta(days=N_DAYS - 1)
CHUNK = 1000

FAST, MED, SLOW = 0, 1, 2
GI_KABS = np.array([1 / 30, 1 / 40, 1 / 55])  # absorption rate per minute
GI_NAME = ["high", "medium", "low"]

# (dish, carb multiplier, GI class). Menus follow the patient's regional staple.
MENUS = {
    "rice": {
        "breakfast": [("Idli-sambar", 1.0, FAST), ("Dosa-chutney", 1.0, FAST), ("Upma", 0.9, MED), ("Pongal", 1.0, FAST)],
        "lunch": [("Rice-sambar-poriyal", 1.1, FAST), ("Rice-fish curry", 1.0, FAST), ("Curd rice", 0.9, MED), ("Rice-dal-sabzi", 1.0, MED)],
        "dinner": [("Rice-rasam", 1.0, FAST), ("Chapati-kurma", 0.9, MED), ("Dosa", 0.9, FAST), ("Rice-dal", 1.0, MED)],
    },
    "wheat": {
        "breakfast": [("Aloo paratha", 1.0, MED), ("Poha", 0.9, FAST), ("Bread-omelette", 0.7, MED), ("Besan chilla", 0.6, SLOW)],
        "lunch": [("Roti-dal-sabzi", 1.0, SLOW), ("Rajma-chawal", 1.1, MED), ("Chole-roti", 1.0, SLOW), ("Kadhi-chawal", 1.0, MED)],
        "dinner": [("Roti-sabzi", 0.9, SLOW), ("Dal-roti", 0.9, SLOW), ("Khichdi", 1.0, MED), ("Paneer-roti", 0.8, SLOW)],
    },
    "mixed": {
        "breakfast": [("Poha", 0.9, FAST), ("Thepla-curd", 0.9, MED), ("Upma", 0.9, MED), ("Idli-sambar", 1.0, FAST)],
        "lunch": [("Dal-rice-roti", 1.1, MED), ("Roti-sabzi-dal", 1.0, SLOW), ("Pav bhaji", 1.1, FAST), ("Rice-curry", 1.0, FAST)],
        "dinner": [("Khichdi-kadhi", 1.0, MED), ("Roti-sabzi", 0.9, SLOW), ("Bhakri-sabzi", 0.9, SLOW), ("Dal-rice", 1.0, MED)],
    },
}
SLOT_SCALE = {"breakfast": 0.75, "lunch": 1.0, "dinner": 0.95}
SNACKS = [("Chai-biscuits", 25, FAST), ("Samosa-chai", 35, FAST), ("Fruit", 20, MED), ("Roasted chana", 15, SLOW)]
FEASTS = [("Wedding feast", 1.9, FAST), ("Festival thali", 1.8, FAST), ("Biryani party", 1.8, FAST)]
FASTING_FOODS = [("Sabudana khichdi", 45, FAST), ("Fruit and milk", 30, MED)]
SWEETS = ("Sweets (mithai)", 40, FAST)

# Exercise codes -> (name, glucose intensity, mean steps per 5-min, extra HR)
EXERCISE = {1: ("Yoga", 0.6, 20, 15), 2: ("Walk", 1.0, 480, 5), 3: ("Brisk walk / gym", 1.8, 600, 22)}
EX_INTENSITY = np.array([0.0, 0.6, 1.0, 1.8], dtype=np.float32)
EX_STEPS = np.array([0, 20, 480, 600], dtype=np.float32)
EX_HR = np.array([0, 15, 5, 22], dtype=np.float32)

# Glucose-lowering effect per drug class (HbA1c %-points) and hypo pressure (mg/dL)
DRUG_EFFECT = {"Biguanide": 0.9, "Sulfonylurea": 0.8, "DPP-4 inhibitor": 0.5, "SGLT2 inhibitor": 0.5, "Basal insulin": 1.2}
HYPO_AMP = {"Sulfonylurea": 35.0, "Basal insulin": 30.0}
STAGES = np.array([None, "awake", "light", "deep", "rem"], dtype=object)  # index = code + 1


def hour_to_bin(day, hour):
    return int(day * BPD + round(hour * 60 / STEP_MIN))


# --------------------------------------------------------------------------
# Inputs from the EHR
# --------------------------------------------------------------------------
def load_patients(limit: int | None):
    pats = pd.read_csv(EHR / "patients.csv")
    tp = pd.read_csv(EHR / "twin_params.csv")
    feats = pd.read_csv(EHR / "patient_features.csv")[["patient_id", "last_hba1c", "hba1c_mean"]]
    meds = pd.read_csv(EHR / "medications.csv")
    df = pats.merge(tp, on="patient_id").merge(feats, on="patient_id")
    if limit:
        df = df.head(limit)

    active = meds[meds.end_date.isna() & (meds.indication == "Type 2 diabetes")]
    regimen = {pid: list(g[["drug", "drug_class", "dose"]].itertuples(index=False, name=None))
               for pid, g in active.groupby("patient_id")}
    df["regimen"] = df.patient_id.map(regimen).apply(lambda r: r if isinstance(r, list) else [])
    return df.reset_index(drop=True)


def target_a1c(df, rng):
    """HbA1c the CGM is calibrated to: latest lab value plus realistic CGM-lab discordance."""
    est = 5.1 + 1.6 * np.log(1 / (df.insulin_sensitivity * df.beta_cell_function)) + df.a1c_offset
    a1c = df.last_hba1c.fillna(df.hba1c_mean).fillna(est - np.where(df.cohort == "T2D", 1.0, 0.0))
    t2d = (df.cohort == "T2D").to_numpy()
    a1c = a1c + rng.normal(0, np.where(t2d, 0.45, 0.15))  # CGM-lab discordance
    a1c = np.where(df.cohort == "Normoglycemic", np.clip(a1c, 4.6, 5.6),
                   np.where(df.cohort == "Prediabetes", np.clip(a1c, 5.7, 6.4), np.clip(a1c, 5.4, 13.0)))
    return a1c


# --------------------------------------------------------------------------
# Behaviour: meals, medication, exercise, sleep, stress, special days
# --------------------------------------------------------------------------
def simulate_behaviour(df, rng):
    meals, doses, exercise, stress, nights, days = [], [], [], [], [], []
    weekday = [(START + pd.Timedelta(days=d)).dayofweek for d in range(N_DAYS)]  # 0=Mon

    for i, p in enumerate(df.itertuples(index=False)):
        menu = MENUS[p.staple]
        act = {"Sedentary": 0, "Moderate": 1, "Active": 2}[p.activity_level]
        urban = p.setting == "Urban"
        day_type = ["weekend" if weekday[d] >= 5 else "weekday" for d in range(N_DAYS)]
        if rng.random() < 0.35:
            day_type[rng.integers(N_DAYS)] = "feast"
        if rng.random() < 0.20:
            d = rng.integers(N_DAYS)
            if day_type[d] != "feast":
                day_type[d] = "fasting"
        ill = np.zeros(N_DAYS, bool)
        if rng.random() < 0.06:
            s = rng.integers(N_DAYS)
            ill[s:s + rng.integers(2, 5)] = True

        # sleep the night before day 0
        prev_bed = rng.normal(23.1, 0.6)  # hours from midnight of day -1
        prev_dur = np.clip(p.sleep_duration_h * rng.normal(1, 0.12), 3.5, 10)
        nights.append((i, -1, prev_bed, prev_bed + prev_dur, p.sleep_efficiency))
        wake = prev_bed + prev_dur - 24

        for d in range(N_DAYS):
            wk = day_type[d] == "weekend"
            fasting, feast = day_type[d] == "fasting", day_type[d] == "feast"
            poor_sleep_prev = prev_dur < 5.5
            n_meals_logged = 0

            def add_meal(slot, hour, name, carbs, gi):
                logged = rng.random() < 0.6
                est = carbs * rng.lognormal(0, 0.3) if logged else np.nan
                meals.append((i, hour_to_bin(d, hour), slot, name, carbs, gi, logged, est))

            # Breakfast
            bf = max(rng.normal(8.5, 0.5) + 0.6 * wk, wake + 0.3)
            if not fasting and rng.random() > 0.08:
                name, mult, gi = menu["breakfast"][rng.integers(4)]
                add_meal("breakfast", bf, name, p.meal_carbs_g * SLOT_SCALE["breakfast"] * mult * rng.lognormal(0, 0.15), gi)
            # Lunch
            lunch = rng.normal(13.5, 0.5) + 0.4 * wk
            if fasting:
                name, carbs, gi = FASTING_FOODS[rng.integers(2)]
                add_meal("lunch", lunch, name, carbs * rng.lognormal(0, 0.15), gi)
            elif rng.random() > 0.03:
                name, mult, gi = menu["lunch"][rng.integers(4)]
                add_meal("lunch", lunch, name, p.meal_carbs_g * mult * (1.1 if wk else 1.0) * rng.lognormal(0, 0.15), gi)
            # Evening snack (chai time)
            if rng.random() < (0.8 if urban else 0.65):
                name, carbs, gi = SNACKS[rng.integers(4)]
                add_meal("snack", rng.normal(17.3, 0.4), name, carbs * rng.lognormal(0, 0.2), gi)
            # Dinner (late, Indian style)
            dinner = rng.normal(21.0, 0.6) + 0.4 * wk - 1.0 * fasting
            if feast:
                name, mult, gi = FEASTS[rng.integers(3)]
                add_meal("dinner", dinner, name, p.meal_carbs_g * mult * rng.lognormal(0, 0.1), gi)
                add_meal("snack", dinner + 0.8, *SWEETS)
            else:
                name, mult, gi = menu["dinner"][rng.integers(4)]
                add_meal("dinner", dinner, name, p.meal_carbs_g * SLOT_SCALE["dinner"] * mult * (1.3 if fasting else 1.0)
                         * rng.lognormal(0, 0.15), gi)

            # Medication doses
            taken = total = 0
            hypo_taken = hypo_total = 0.0
            for drug, cls, dose in p.regimen:
                times = [8.5, 20.5] if "BD" in dose else [22.0] if "HS" in dose else [8.5]
                for h in times:
                    ok = rng.random() < p.med_adherence * (0.85 if ill[d] else 1.0)
                    doses.append((i, hour_to_bin(d, h + rng.normal(0, 0.3)), drug, cls, dose, ok,
                                  bool(ok and rng.random() < 0.8)))
                    taken += ok
                    total += 1
                    if cls in HYPO_AMP:
                        hypo_taken += HYPO_AMP[cls] * ok / len(times)
                        hypo_total += HYPO_AMP[cls] / len(times)

            # Exercise
            if not ill[d]:
                if rng.random() < [0.05, 0.25, 0.6][act]:
                    code = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3] if act < 2 else [0.2, 0.3, 0.5])
                    exercise.append((i, hour_to_bin(d, max(rng.normal(6.6, 0.5), wake + 0.1)), int(rng.integers(6, 13)), code))
                if rng.random() < [0.03, 0.10, 0.15][act]:
                    exercise.append((i, hour_to_bin(d, lunch + rng.uniform(0.25, 0.6)), int(rng.integers(3, 6)), 2))
                if rng.random() < [0.12, 0.35, 0.50][act]:
                    exercise.append((i, hour_to_bin(d, dinner + rng.uniform(0.3, 0.7)), int(rng.integers(3, 8)), 2))

            # Stress episodes
            n_stress = 0
            if rng.random() < ((0.45 if urban else 0.3) if day_type[d] == "weekday" else 0.15):
                stress.append((i, hour_to_bin(d, rng.uniform(10, 19)), rng.uniform(0.6, 1.2)))
                n_stress = 1

            # Tonight's sleep
            bed = max(rng.normal(23.1, 0.6) + 0.5 * (weekday[(d + 1) % 7] >= 5 if d + 1 < N_DAYS else 0), dinner + 0.75)
            dur = p.sleep_duration_h * rng.normal(1, 0.12) - 0.5 * n_stress * rng.random() + 0.5 * ill[d] + 0.4 * wk
            dur = float(np.clip(dur, 3.5, 10.5))
            nights.append((i, d, bed, bed + dur, p.sleep_efficiency - 0.05 * ill[d]))

            days.append({"i": i, "day": d, "day_type": "illness" if ill[d] else day_type[d], "ill": bool(ill[d]),
                         "poor_sleep_prev": bool(poor_sleep_prev),
                         "missed_dose_frac": (1 - taken / total) if total else 0.0,
                         "hypo_amp": hypo_taken * (2.5 if fasting else 1.0), "stress_events": n_stress})
            prev_dur, wake = dur, bed + dur - 24

    meals = pd.DataFrame(meals, columns=["i", "t", "slot", "food", "carbs_g", "gi", "logged", "logged_carbs_g"])
    doses = pd.DataFrame(doses, columns=["i", "t", "drug", "drug_class", "dose", "taken", "logged"])
    exercise = pd.DataFrame(exercise, columns=["i", "t", "n_bins", "code"])
    stress = pd.DataFrame(stress, columns=["i", "t", "magnitude"])
    nights = pd.DataFrame(nights, columns=["i", "night", "bed_h", "wake_h", "efficiency"])
    days = pd.DataFrame(days)
    for t in (meals, doses, exercise, stress):
        t.drop(t.index[(t.t < 0) | (t.t >= T)], inplace=True)
    return meals, doses, exercise, stress, nights, days


# --------------------------------------------------------------------------
# Physiology for one chunk of patients
# --------------------------------------------------------------------------
def unit_peak(tau):
    """Peak excursion produced by a 10 g medium-GI meal with c = 1."""
    k = GI_KABS[MED]
    q1, q2, x = np.full_like(tau, 10.0), np.zeros_like(tau), np.zeros_like(tau)
    peak = np.zeros_like(tau)
    for _ in range(80):
        ra = k * q2
        q1, q2 = q1 - STEP_MIN * k * q1, q2 + STEP_MIN * k * (q1 - q2)
        x = x + STEP_MIN * (-x / tau + ra)
        peak = np.maximum(peak, x)
    return peak


def ar1(rng, shape, phi, sd):
    w = rng.normal(0, sd * np.sqrt(1 - phi ** 2), shape)
    return lfilter([1.0], [1.0, -phi], w, axis=0).astype(np.float32)


def sleep_stages(nights, n, rng):
    """Stage code per bin: -1 not in bed, 0 awake, 1 light, 2 deep, 3 rem."""
    stage = np.full((T, n), -1, np.int8)
    b = np.round(nights.night * BPD + nights.bed_h * 12).astype(int).to_numpy()
    w = np.round(nights.night * BPD + nights.wake_h * 12).astype(int).to_numpy()
    b_c, w_c = np.clip(b, 0, T), np.clip(w, 0, T)
    lengths = np.maximum(w_c - b_c, 0)
    owner = np.repeat(np.arange(len(nights)), lengths)
    offset = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths) + np.repeat(b_c - b, lengths)
    total = np.repeat(w - b, lengths)
    x = offset / total                                # fraction of the night elapsed
    phase = (offset * STEP_MIN % 90) / 90             # position in 90-min sleep cycle
    eff = nights.efficiency.to_numpy()[owner]
    p_awake = (1 - eff) * 0.6 + 0.6 * (offset < 3)
    p_deep = 0.75 * (1 - x) ** 2 * (phase < 0.45)
    p_rem = (0.35 + 0.8 * x) * (phase > 0.55)
    p_light = np.full_like(x, 0.55)
    probs = np.stack([p_awake, p_light, p_deep, p_rem], 1)
    probs /= probs.sum(1, keepdims=True)
    code = (rng.random(len(x))[:, None] > np.cumsum(probs, 1)).sum(1).astype(np.int8)
    stage[np.repeat(b_c, lengths) + offset - np.repeat(b_c - b, lengths), nights.i.to_numpy()[owner]] = code
    return stage


def simulate_chunk(df, meals, doses, exercise, stress, nights, days, a1c_target, rng):
    n = len(df)
    si, beta = df.insulin_sensitivity.to_numpy(), df.beta_cell_function.to_numpy()
    tau = (35 / (si * beta) ** 0.35).astype(np.float64)          # minutes
    peak10 = np.minimum(df["carb_sensitivity_mgdl_per_10g"].to_numpy() * 1.5, 15.0)  # peak rise per 10 g carbs
    c = peak10 / unit_peak(tau)

    d_ill = days.pivot(index="day", columns="i", values="ill").to_numpy(bool)
    d_poor = days.pivot(index="day", columns="i", values="poor_sleep_prev").to_numpy(bool)
    d_miss = days.pivot(index="day", columns="i", values="missed_dose_frac").to_numpy()
    d_hypo = days.pivot(index="day", columns="i", values="hypo_amp").to_numpy()
    meal_scale = 1 + 0.15 * d_poor + 0.2 * d_ill

    # Sparse inputs bucketed by time
    carb_in = [[] for _ in range(T)]
    for r in meals.itertuples(index=False):
        carb_in[r.t].append((r.i, r.gi, r.carbs_g))
    stress_in = [[] for _ in range(T)]
    for r in stress.itertuples(index=False):
        stress_in[r.t].append((r.i, r.magnitude))
    ex = np.zeros((T, n), np.int8)
    for r in exercise.itertuples(index=False):
        ex[r.t:r.t + r.n_bins, r.i] = r.code
    ex_int = EX_INTENSITY[ex]

    q1, q2 = np.zeros((3, n)), np.zeros((3, n))
    x, s = np.zeros(n), np.zeros(n)
    last_meal = np.full(n, -60)
    X = np.empty((T, n), np.float32)
    S = np.empty((T, n), np.float32)
    P = np.empty((T, n), np.float32)
    k = GI_KABS[:, None]
    for t in range(T):
        d = t // BPD
        for i, gi, carbs in carb_in[t]:
            q1[gi, i] += carbs
            if carbs >= 15:
                last_meal[i] = t
        for i, mag in stress_in[t]:
            s[i] += mag
        ra = (k * q2).sum(0)
        q1, q2 = q1 - STEP_MIN * k * q1, q2 + STEP_MIN * k * (q1 - q2)
        e = ex_int[t]
        x = x + STEP_MIN * (-x / tau * (1 + e) + c * meal_scale[d] * ra - 0.4 * e)
        x = np.maximum(x, -40)
        s = s - STEP_MIN * s / 90
        hrs = (t - last_meal) * STEP_MIN / 60
        X[t], S[t] = x, s
        P[t] = d_hypo[d] * np.clip((hrs - 3) / 4, 0, 1)

    hour = (np.arange(T) % BPD) * STEP_MIN / 60
    dawn = df.dawn_phenomenon_mgdl.to_numpy()[None, :] * np.exp(-0.5 * ((hour[:, None] - 6.5) / 1.2) ** 2)
    drug_mgdl = np.array([sum(DRUG_EFFECT.get(cls, 0) for _, cls, _ in r) for r in df.regimen]) * 28.7 * 0.6
    target_mean = 28.7 * a1c_target - 46.7                              # ADAG estimated average glucose
    day_var = rng.normal(0, 0.07, (N_DAYS, n)) * target_mean            # day-to-day baseline variation
    day_off = d_miss * drug_mgdl + d_ill * np.where(df.cohort == "T2D", 35, 12) + d_poor * 4 + day_var
    drift = ar1(rng, (T, n), np.exp(-STEP_MIN / 240), 1.0) * (0.09 * target_mean)  # slow between-meal drift
    comp = (np.repeat(day_off, BPD, axis=0) + dawn + X + 14 * df.stress_reactivity.to_numpy() * S - P + drift)
    gb = np.maximum(target_mean - comp.mean(0), 62)
    glucose = np.clip(gb + comp, 40, 500).astype(np.float32)

    stage = sleep_stages(nights, n, rng)
    asleep = stage >= 0

    # ---- CGM sensor
    cgm = glucose * (1 + ar1(rng, (T, n), 0.7, 0.045)) + rng.normal(0, 3, (T, n))
    starts = (rng.random((T, n)) < 0.05 / 90) & asleep                  # ~5% of nights: compression low
    cgm -= lfilter([15, 30, 30, 30, 25, 15], [1.0], starts.astype(np.float32), axis=0)
    cgm = np.clip(np.round(cgm), 40, 400)
    cgm[:12] = np.nan                                                    # sensor warm-up (1 h)
    for d in range(N_DAYS):
        for i in np.where(rng.random(n) < 0.08)[0]:                      # sync gaps 1-4 h
            g0 = d * BPD + rng.integers(0, BPD)
            cgm[g0:g0 + rng.integers(12, 49), i] = np.nan

    # ---- Steps
    walk_steps = rng.poisson(EX_STEPS[ex]).astype(np.float32)
    day_hours = hour[:BPD]
    w_hour = np.select([day_hours < 7, day_hours < 10, day_hours < 17, day_hours < 21], [0.5, 1.3, 1.0, 1.3], 0.6)
    steps = np.zeros((T, n), np.float32)
    wk_mult = np.array([0.85 if (START + pd.Timedelta(days=d)).dayofweek >= 5 else 1.0 for d in range(N_DAYS)])
    for d in range(N_DAYS):
        sl = slice(d * BPD, (d + 1) * BPD)
        awake_w = w_hour[:, None] * (~asleep[sl])
        target = df.daily_steps_baseline.to_numpy() * wk_mult[d] * rng.lognormal(0, 0.2, n) * np.where(d_ill[d], 0.4, 1)
        bg = np.maximum(target - walk_steps[sl].sum(0), 600)
        lam = bg * awake_w / np.maximum(awake_w.sum(0), 1)
        steps[sl] = rng.poisson(lam) + walk_steps[sl]

    # ---- Heart rate
    stage_adj = np.array([7, 3, -4, -7, -1], np.float32)[stage + 1]
    ill_t = np.repeat(d_ill, BPD, axis=0)
    hr = (df.resting_hr_bpm.to_numpy()[None, :] + stage_adj + 0.05 * steps + EX_HR[ex] + 8 * S + 8 * ill_t
          + 0.02 * np.maximum(glucose - 200, 0) + 12 * (glucose < 65) + rng.normal(0, 2.5, (T, n)))
    hr = np.round(np.clip(hr, 40, 190))

    # ---- Watch off-wrist / charging gaps (daytime)
    for d in range(N_DAYS):
        for i in np.where(rng.random(n) < 0.35)[0]:
            g0 = d * BPD + rng.integers(7 * 12, 21 * 12)
            g1 = g0 + rng.integers(12, 19)
            hr[g0:g1, i] = np.nan
            steps[g0:g1, i] = np.nan

    return dict(glucose=glucose, cgm=cgm, hr=hr, steps=steps, stage=stage, ex=ex, S=S, gb=gb)


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------
def daily_summary(df, sim, nights, days, rng):
    rows = []
    cgm, hr, stage, steps, glucose = sim["cgm"], sim["hr"], sim["stage"], sim["steps"], sim["glucose"]
    hrv_base = df.hrv_rmssd_ms.to_numpy()
    for d in range(N_DAYS):
        sl = slice(d * BPD, (d + 1) * BPD)
        g = cgm[sl]
        valid = ~np.isnan(g)
        nv = valid.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.nanmean(g, 0)
            sd = np.nanstd(g, 0)
            frac = lambda m: np.where(nv > 0, (m & valid).sum(0) / np.maximum(nv, 1) * 100, np.nan)
            # last night = sleep that ended this morning (bins before ~noon in today's window or late yesterday)
            night_sl = slice(max(0, d * BPD - 6 * 12), d * BPD + 12 * 12)
            st = stage[night_sl]
            slept = (st >= 1).sum(0) * STEP_MIN / 60
            in_bed = (st >= 0).sum(0)
            sleeping_hr = np.where(st >= 1, hr[night_sl], np.nan)
            night_g = np.where(st >= 0, glucose[night_sl], np.nan)
            resting = np.nanpercentile(sleeping_hr, 10, axis=0)
            dd = days[days.day == d].set_index("i")
            hrv = (hrv_base * (1 - 0.15 * (slept < 5.5)) * (1 - 0.2 * dd.ill.to_numpy())
                   * (1 - 0.001 * np.maximum(np.nanmean(night_g, 0) - 150, 0)) * rng.lognormal(0, 0.12, len(df)))
            rows.append(pd.DataFrame({
                "patient_id": df.patient_id, "date": (START + pd.Timedelta(days=d)).date(),
                "cgm_coverage_pct": nv / BPD * 100, "mean_glucose": mean, "sd_glucose": sd, "cv_pct": sd / mean * 100,
                "tir_70_180_pct": frac((g >= 70) & (g <= 180)), "tar_180_pct": frac(g > 180), "tar_250_pct": frac(g > 250),
                "tbr_70_pct": frac(g < 70), "tbr_54_pct": frac(g < 54), "max_glucose": np.nanmax(np.where(valid, g, -1), 0),
                "min_glucose": np.nanmin(np.where(valid, g, 999), 0),
                "steps": np.nansum(steps[sl], 0), "active_minutes": (sim["ex"][sl] > 0).sum(0) * STEP_MIN,
                "sleep_hours": slept, "sleep_efficiency_pct": np.where(in_bed > 0, (st >= 1).sum(0) / np.maximum(in_bed, 1) * 100, np.nan),
                "deep_sleep_pct": np.where(slept > 0, (st == 2).sum(0) / np.maximum((st >= 1).sum(0), 1) * 100, np.nan),
                "rem_sleep_pct": np.where(slept > 0, (st == 3).sum(0) / np.maximum((st >= 1).sum(0), 1) * 100, np.nan),
                "resting_hr": resting, "hrv_rmssd_ms": hrv,
            }))
    out = pd.concat(rows)
    out.loc[out.cgm_coverage_pct == 0, ["max_glucose", "min_glucose"]] = np.nan
    return out.round(1)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(limit, seed):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    df = load_patients(limit)
    a1c = target_a1c(df, rng)
    meals, doses, exercise, stress, nights, days = simulate_behaviour(df, rng)
    print(f"Behaviour simulated for {len(df)} patients ({time.time() - t0:.0f}s)")

    OUT.mkdir(parents=True, exist_ok=True)
    times = START + pd.to_timedelta(np.arange(T) * STEP_MIN, unit="min")
    cgm_writer = vit_writer = None
    daily, patient_rows = [], []
    for c0 in range(0, len(df), CHUNK):
        idx = np.arange(c0, min(c0 + CHUNK, len(df)))
        sub = df.iloc[idx].reset_index(drop=True)

        def local(t):
            t = t[t.i.isin(idx)].copy()
            t["i"] -= c0
            return t

        sim = simulate_chunk(sub, local(meals), local(doses), local(exercise), local(stress), local(nights),
                             local(days), a1c[idx], rng)
        n = len(sub)
        pid = np.repeat(sub.patient_id.to_numpy(), T)
        ts = np.tile(times.to_numpy(), n)
        cgm_tab = pa.table({"patient_id": pid, "timestamp": ts,
                            "glucose_mgdl": pa.array(sim["cgm"].T.ravel(), mask=np.isnan(sim["cgm"].T.ravel())).cast(pa.int16())})
        vit_tab = pa.table({"patient_id": pid, "timestamp": ts,
                            "heart_rate_bpm": pa.array(sim["hr"].T.ravel(), mask=np.isnan(sim["hr"].T.ravel())).cast(pa.int16()),
                            "steps": pa.array(sim["steps"].T.ravel(), mask=np.isnan(sim["steps"].T.ravel())).cast(pa.int16()),
                            "sleep_stage": pa.array(STAGES[sim["stage"].T.ravel() + 1], type=pa.string()).dictionary_encode()})
        if cgm_writer is None:
            cgm_writer = pq.ParquetWriter(OUT / "cgm.parquet", cgm_tab.schema, compression="zstd")
            vit_writer = pq.ParquetWriter(OUT / "vitals.parquet", vit_tab.schema, compression="zstd")
        cgm_writer.write_table(cgm_tab)
        vit_writer.write_table(vit_tab)

        ds = daily_summary(sub, sim, local(nights), local(days), rng)
        daily.append(ds)
        g = sim["cgm"]
        mean_g = np.nanmean(g, 0)
        patient_rows.append(pd.DataFrame({
            "patient_id": sub.patient_id, "cohort": sub.cohort, "ehr_last_hba1c": sub.last_hba1c,
            "cgm_mean_glucose": mean_g.round(1), "gmi_pct": (3.31 + 0.02392 * mean_g).round(2),
            "cv_pct": (np.nanstd(g, 0) / mean_g * 100).round(1),
            "tir_70_180_pct": (np.nanmean((g >= 70) & (g <= 180), 0) / np.mean(~np.isnan(g), 0) * 100).round(1),
            "tar_180_pct": (np.nansum(g > 180, 0) / (~np.isnan(g)).sum(0) * 100).round(1),
            "tar_250_pct": (np.nansum(g > 250, 0) / (~np.isnan(g)).sum(0) * 100).round(1),
            "tbr_70_pct": (np.nansum(g < 70, 0) / (~np.isnan(g)).sum(0) * 100).round(2),
            "tbr_54_pct": (np.nansum(g < 54, 0) / (~np.isnan(g)).sum(0) * 100).round(2),
            "cgm_coverage_pct": ((~np.isnan(g)).mean(0) * 100).round(1),
            "mean_daily_steps": ds.groupby("patient_id", sort=False).steps.mean().to_numpy().round(0),
            "mean_sleep_hours": ds.groupby("patient_id", sort=False).sleep_hours.mean().to_numpy().round(2),
            "mean_resting_hr": ds.groupby("patient_id", sort=False).resting_hr.mean().to_numpy().round(1),
        }))
        print(f"  patients {c0 + 1}-{c0 + n} done ({time.time() - t0:.0f}s)")
    cgm_writer.close()
    vit_writer.close()

    pid_of = df.patient_id.to_numpy()
    to_ts = lambda t: START + pd.to_timedelta(t * STEP_MIN, unit="min")
    ev_meals = pd.DataFrame({"patient_id": pid_of[meals.i], "timestamp": to_ts(meals.t), "event_type": "meal",
                             "meal_slot": meals.slot, "detail": meals.food, "carbs_g": meals.carbs_g.round(0),
                             "glycaemic_index": [GI_NAME[g] for g in meals.gi], "logged": meals.logged,
                             "logged_carbs_g": meals.logged_carbs_g.round(0)})
    ev_doses = pd.DataFrame({"patient_id": pid_of[doses.i], "timestamp": to_ts(doses.t), "event_type": "medication",
                             "detail": doses.drug + " " + doses.dose, "drug_class": doses.drug_class,
                             "dose_taken": doses.taken, "logged": doses.logged})
    ev_ex = pd.DataFrame({"patient_id": pid_of[exercise.i], "timestamp": to_ts(exercise.t), "event_type": "exercise",
                          "detail": [EXERCISE[c][0] for c in exercise.code], "duration_min": exercise.n_bins * STEP_MIN,
                          "logged": True})
    events = pd.concat([ev_meals, ev_doses, ev_ex]).sort_values(["patient_id", "timestamp"])
    events.to_csv(OUT / "events.csv", index=False)

    days_out = days.assign(patient_id=pid_of[days.i], date=[(START + pd.Timedelta(days=int(d))).date() for d in days.day])
    days_out = days_out[["patient_id", "date", "day_type", "ill", "poor_sleep_prev", "missed_dose_frac", "stress_events"]]
    days_out.round(3).to_csv(OUT / "day_context.csv", index=False)
    pd.concat(daily).to_csv(OUT / "daily_summary.csv", index=False)
    pd.concat(patient_rows).to_csv(OUT / "patient_cgm_summary.csv", index=False)
    print(f"Done in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only simulate the first N patients")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    main(args.limit, args.seed)
