# Data Dictionary: Wearable & CGM Stream (Dataset 2)

Fully **synthetic** 5-minute data for every EHR patient (5,000 patients × 14 days × 288 readings/day ≈ 20 million rows per stream),
simulating a **continuous glucose monitor** (FreeStyle Libre / Dexcom style) plus a **smartwatch** (Apple Watch / Fitbit / Noise style).

- **Period:** 2026-09-17 00:00 to 2026-09-30 23:55 (the 14 days before the EHR reference date)
- **Regenerate:** `python src/generate_wearable.py`. Seed 7 is deterministic, and the run takes about 40 s.
- **Validate:** `python src/validate_wearable.py`, which writes `data/wearable/validation_report.html`
- **Read Parquet in Python:** `pd.read_parquet("data/wearable/cgm.parquet", filters=[("patient_id", "==", "P00001")])`

## Files

| File | Grain | What it is | Model may use? |
|---|---|---|---|
| `cgm.parquet` | patient × 5 min | CGM glucose | ✅ |
| `vitals.parquet` | patient × 5 min | Heart rate, steps, sleep stage | ✅ |
| `events.csv` | 1 per event | Meals, medication doses, exercise | ✅ only the `logged*` columns |
| `daily_summary.csv` | patient × day | What a wearable app shows: TIR, sleep, steps, HRV | ✅ |
| `patient_cgm_summary.csv` | 1 per patient | 14-day CGM metrics + EHR HbA1c (dashboard / validation) | ✅ |
| `day_context.csv` | patient × day | **Hidden truth**: day type, illness, poor sleep, missed doses, stress | ❌ evaluation only |

### cgm.parquet
| Column | Type | Notes |
|---|---|---|
| patient_id | string | Links to `data/ehr/patients.csv` |
| timestamp | datetime | Local time (IST), every 5 min |
| glucose_mgdl | int16, nullable | Sensor reading, 40–400 mg/dL. **Null** = sensor warm-up (first hour) or sync gap |

### vitals.parquet
| Column | Type | Notes |
|---|---|---|
| heart_rate_bpm | int16, nullable | Null while the watch is off-wrist or charging (~35% of days, 1–1.5 h) |
| steps | int16, nullable | Steps in the 5-minute bin |
| sleep_stage | string | `awake`, `light`, `deep`, `rem` while in bed; null otherwise |

### events.csv
| Column | Applies to | Notes |
|---|---|---|
| event_type | all | `meal`, `medication`, `exercise` |
| meal_slot, detail | meal | breakfast / lunch / snack / dinner; Indian dish name (Idli-sambar, Roti-dal-sabzi, Biryani, ...) |
| carbs_g, glycaemic_index | meal | **True** carbohydrate amount and GI class (hidden truth) |
| logged, logged_carbs_g | meal | What the patient entered in the app: only ~60% of meals are logged, with ±30% estimation error |
| drug_class, dose_taken, logged | medication | `dose_taken` = truth; `logged` = patient confirmed the reminder (~80% of taken doses) |
| duration_min | exercise | Yoga / Walk / Brisk walk-gym (auto-detected by the watch, so always logged) |

### daily_summary.csv
`cgm_coverage_pct, mean_glucose, sd_glucose, cv_pct, tir_70_180_pct, tar_180_pct, tar_250_pct, tbr_70_pct, tbr_54_pct,
max_glucose, min_glucose, steps, active_minutes, sleep_hours, sleep_efficiency_pct, deep_sleep_pct, rem_sleep_pct,
resting_hr, hrv_rmssd_ms`. Sleep and HRV refer to the night that ended that morning.

## How it is generated

**Each patient's behaviour** is drawn from their EHR profile:
- **Meals** follow Indian timings: breakfast about 8:30, lunch about 13:30, chai snack about 17:15 and a **late dinner about 21:00**. Weekends run later and larger.
  Dishes and carbs follow the regional staple (rice-heavy South/East, wheat North/Central). About 8% of breakfasts are skipped.
- **Special days:** festival or wedding feasts (≈1.8–2× dinner carbs plus sweets), fasting days (sabudana or fruit, then a big dinner),
  and illness spells of 2–4 days (higher glucose, fewer steps, higher HR, missed doses).
- **Medication:** each active prescription from `medications.csv` at its real schedule (OD / BD / HS). Each dose is taken with
  probability = the patient's adherence.
- **Exercise:** morning yoga, walks or gym sessions and post-meal walks, more frequent for active patients.
- **Sleep:** bedtime about 23:00 (always after dinner), with duration and efficiency from the twin. Stages follow 90-minute cycles,
  with more deep sleep early in the night and more REM late.
- **Stress:** weekday work-stress episodes (more common in urban patients) raise glucose and HR.

**Glucose physiology** (per patient, 5-minute steps):
```
gut:        carbs -> 2-compartment absorption, speed set by GI (high 30 min, medium 40, low 55)
excursion:  dX/dt = -X / tau * (1 + exercise) + c * Ra(t) - exercise uptake
glucose  =  daily baseline + dawn rise + X + stress - hypo pressure (SU / insulin) + slow drift
  tau (clearance time) = 35 / (insulin sensitivity * beta-cell function)^0.35
  c   scaled so 10 g carbs raises glucose by min(1.5 * carb sensitivity, 15) mg/dL
  daily baseline shifts with missed doses, illness, poor sleep the night before and day-to-day variation
  baseline calibrated so the 14-day mean glucose = estimated average glucose of the patient's HbA1c (+/- realistic discordance)
```
**Sensor realism:** multiplicative sensor noise (MARD-like), compression lows during sleep on ~5% of nights, a one-hour warm-up,
1–4 h sync gaps on ~8% of days, and readings capped to the sensor range of 40–400.

**Heart rate** = resting HR + sleep-stage effect + steps + exercise + stress + illness + hyperglycaemia + a hypoglycaemia surge.
This adrenergic response gives the model an early warning signal for lows.
**Nightly HRV** falls with poor sleep, illness, high night-time glucose and neuropathy.

## Prediction targets (computed from `cgm.parquet` at modelling time)
| Target | Definition |
|---|---|
| Glucose spike (primary) | Current reading < 180 and **any reading > 180 in the next 2 h** |
| Severe spike | Any reading > 250 in the next 2 h |
| Hypoglycaemia | Current reading ≥ 70 and at least two consecutive readings < 70 (≥ 10 min) in the next 2 h |

## Validation summary (`validate_wearable.py`)
- **Agreement with the EHR:** CGM-derived GMI vs lab HbA1c, r = 0.93 in T2D; 35% differ by ≥ 0.5% (real-world studies report about 28%).
- **Reference ranges:** 16/16 metrics in published ranges (healthy TIR 99%, T2D TIR 58%, CV, hypoglycaemia by drug class,
  sleep, steps, resting HR, HRV).
- **Comparison with real patients:** ShanghaiT2DM CGM data, matched on mean glucose (simulated vs real medians):

| Metric | Real (ShanghaiT2DM) | Simulated |
|---|---|---|
| Time in range 70–180 | 83.5% | 82.3% |
| Time > 180 | 14.6% | 14.9% |
| Time < 70 | 0.3% | 0.3% |
| CV | 27.5% | 24.6% |
| Post-meal time to peak | 90 min | 90 min |
| Post-meal rise | 52 mg/dL | 88 mg/dL* |

\*Expected: Indian rice/roti meals (65–90 g carbs) are much larger than the small, low-carb hospital meals in the Shanghai cohort.

## External data attribution
ShanghaiT2DM: Zhao Q. et al., "Chinese diabetes datasets for data-driven machine learning", *Scientific Data* 10, 35 (2023).
figshare: [doi:10.6084/m9.figshare.20444397](https://doi.org/10.6084/m9.figshare.20444397), licensed CC BY 4.0.
It is used only for validation. Raw files are in `data/external/shanghai/raw`, and tidy versions come from `src/process_shanghai.py`.
