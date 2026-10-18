# Data Dictionary: Synthetic EHR (Dataset 1)

All records are **fully synthetic**. No real patient data was used (DPDP Act / HIPAA compliant by design).
Patients have IDs only, with no names, phone numbers or addresses.

- **Cohort:** 5,000 Indian adults: 3,000 Type 2 Diabetes, 1,000 prediabetes, 1,000 normoglycemic
- **Reference date ("today"):** 2026-09-30
- **Recorded history:** the last 3 years of labs and vitals. Diagnoses and medications can start earlier.
- **Regenerate:** `python src/generate_ehr.py --n 5000 --seed 42`. The same seed always gives the same data.
- **Validate:** `python src/validate_ehr.py`, which writes `data/ehr/validation_report.html`

## Files

| File | Grain | Rows (approx.) | Purpose |
|---|---|---|---|
| `patients.csv` | 1 per patient | 5,000 | Demographics, lifestyle, family history |
| `diagnoses.csv` | 1 per condition | 14,600 | ICD-10 coded problem list with onset dates |
| `labs.csv` | 1 per measurement | 336,000 | Labs and vitals over time (LOINC coded) |
| `medications.csv` | 1 per prescription period | 12,600 | Drug, dose, start/end dates |
| `genetics.csv` | 1 per patient | 5,000 | T2D risk SNPs and polygenic risk score |
| `twin_params.csv` | 1 per patient | 5,000 | **Hidden** physiology that drives the wearable simulator |
| `patient_features.csv` | 1 per patient | 5,000 | Wide ML-ready table: latest values, flags, trends |

## patients.csv
| Column | Description |
|---|---|
| patient_id | `P00001` to `P05000` |
| cohort | `T2D`, `Prediabetes`, `Normoglycemic` |
| age, sex | Years; `M` / `F` |
| region, setting | North/South/East/West/Central/Northeast; Urban/Rural |
| height_cm, weight_kg, bmi, waist_cm | Body measures at the reference date |
| diet, staple | Vegetarian / Non-vegetarian; rice / wheat / mixed (by region) |
| activity_level | Sedentary / Moderate / Active |
| smoking, alcohol | Current/Former/Never; None/Occasional/Regular |
| family_history_diabetes, family_history_cvd | Boolean |
| diabetes_duration_yrs | Years since T2D diagnosis (T2D only) |

## diagnoses.csv
`patient_id, icd10, description, onset_date`

Codes used: E11.9 T2D, R73.03 prediabetes, I10 hypertension, E78.5 dyslipidaemia, E66.9 obesity (Asian BMI cut-off ≥ 25),
K76.0 fatty liver, I25.10 coronary artery disease, E03.9 hypothyroidism, E11.40 diabetic neuropathy,
E11.319 diabetic retinopathy, N18.1–N18.5 CKD stages (eGFR < 60, or 2 consecutive UACR ≥ 30).

## labs.csv
`patient_id, date, category (laboratory | vital-signs), loinc, test, value, unit`

| Test | LOINC | Unit | Frequency |
|---|---|---|---|
| HbA1c | 4548-4 | % | Every visit |
| Fasting plasma glucose | 1558-6 | mg/dL | Every visit |
| Post-prandial glucose (2h) | 1521-4 | mg/dL | Most visits |
| Total / LDL / HDL cholesterol, triglycerides | 2093-3 / 13457-7 / 2085-9 / 2571-8 | mg/dL | Yearly |
| Serum creatinine, eGFR (CKD-EPI 2021) | 2160-0 / 98979-8 | mg/dL, mL/min/1.73m² | Yearly |
| Urine albumin/creatinine ratio | 9318-7 | mg/g | Yearly (not always ordered) |
| ALT | 1742-6 | U/L | Yearly |
| Systolic / diastolic BP, heart rate | 8480-6 / 8462-4 / 8867-4 | mmHg, bpm | Every visit |
| Weight, BMI, waist | 29463-7 / 39156-5 / 8280-0 | kg, kg/m², cm | Every visit (waist yearly) |

Visits: T2D about every 3 months, prediabetes about every 6 months, normoglycemic yearly.
Visits are sometimes missed and some tests are not ordered, so **missing values are intentional** to mimic real EHRs.

## medications.csv
`patient_id, drug, drug_class, dose, indication, start_date, end_date` (empty `end_date` = ongoing)

Glucose-lowering drugs follow a treatment ladder: Metformin → dose increase → Sulfonylurea or DPP-4 inhibitor →
SGLT2 inhibitor → basal insulin. Therapy is intensified when HbA1c > 7%, with realistic clinical inertia
(intensification is not guaranteed at every visit) and access limits (rural patients are intensified more slowly).
Safety rules: metformin stopped if eGFR < 30, SGLT2i stopped if eGFR < 20, sulfonylurea often stopped if HbA1c < 6.2
(hypoglycaemia risk). About 8% of patients cannot tolerate metformin.
Other drugs: Telmisartan, Amlodipine, Atorvastatin/Rosuvastatin, Aspirin, Levothyroxine.

## genetics.csv
Risk-allele counts (0/1/2) for TCF7L2 rs7903146, FTO rs9939609, KCNQ1 rs2237892 and SLC30A8 rs13266634,
plus `polygenic_risk_score_z`. Allele frequencies are illustrative approximations, not exact population values.

## twin_params.csv (hidden: do NOT use as model input)
The patient's "true physiology". It determines their EHR values and will drive the simulated CGM and wearable
stream (Dataset 2), so both streams describe the same body.

| Column | Meaning |
|---|---|
| insulin_sensitivity | 1.0 = healthy; lower = more insulin resistant (worse with BMI, inactivity, age) |
| beta_cell_function | 0–1 pancreatic insulin output; declines with diabetes duration and TCF7L2 risk alleles |
| a1c_offset | Individual glycaemic offset |
| carb_sensitivity_mgdl_per_10g | Glucose rise per 10 g carbohydrate |
| dawn_phenomenon_mgdl | Early-morning glucose rise |
| resting_hr_bpm, hrv_rmssd_ms | Cardiac baseline (HRV lower with age, diabetes, neuropathy) |
| sleep_duration_h, sleep_efficiency | Sleep baseline |
| stress_reactivity | Glucose/HR response to stress |
| daily_steps_baseline | Typical step count |
| meal_carbs_g | Typical main-meal carbohydrate load (rice-based diets highest) |
| med_adherence | 0–1 fraction of doses taken (about 30% of T2D patients are poor adherers) |
| weight_trend_kg_per_yr | Weight drift |

## Clinical logic (summary)
- **Insulin sensitivity** = f(cohort, BMI, activity, age) and **beta-cell function** = f(cohort, duration, genetics, lean phenotype)
- **Untreated HbA1c** = 5.1 + 1.6·ln(1 / (sensitivity × beta-cell)) + individual offset
- **Observed HbA1c** moves toward (untreated − drug effects × adherence) with a time constant of about 3 months
- **Fasting and post-meal glucose** are derived from HbA1c with measurement noise
- **eGFR** declines with age, plus extra decline from diabetes, high HbA1c and hypertension. Creatinine is back-calculated with CKD-EPI 2021.
- **Complications** (neuropathy, retinopathy, CAD) are drawn from risk models using duration, HbA1c, BP, smoking and age
- Uses **Asian BMI cut-offs** (overweight ≥ 23, obese ≥ 25) and captures **lean diabetes** (about 16% of T2D with BMI < 23)

## Validation against Indian literature (seed 42)
| Metric (T2D cohort) | Synthetic | Approx. reference |
|---|---|---|
| Mean HbA1c | 7.8% | 7.6–8.8% |
| HbA1c < 7% | 25% | 20–40% |
| HbA1c > 9% | 15% | 10–30% |
| Mean age at diagnosis | 46.7 yrs | 40–50 |
| Hypertension | 45% | 40–60% |
| Dyslipidaemia | 83% | 70–90% |
| CKD | 33% | 25–40% |
| Neuropathy / Retinopathy | 24% / 18% | 18–35% / 12–22% |
| Metformin / Insulin use | 89% / 11% | 80–97% / 7–20% |

Reference ranges are approximations from ICMR-INDIAB, SMART-India and Indian real-world T2D studies.
Verify specific figures before quoting them in the final presentation.
