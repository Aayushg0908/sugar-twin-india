# Sugar Twin India: a Digital Twin that Predicts Glucose Spikes Before They Happen

> Happiest Health **Digital Twin Challenge 2026**. *Work in progress.*
> Team **MITIANS**, Maharaja Agrasen Institute of Technology

A virtual replica of each patient that fuses **electronic health records** with **wearable / CGM data**
to predict an adverse glycaemic event about 2 hours in advance and show it to a doctor on a dashboard.

## Status
| Component | Status |
|---|---|
| 1. Synthetic EHR dataset (5,000 Indian patients) | ✅ Done |
| 2. Wearable / CGM time-series dataset (14 days, 5-min) | ✅ Done |
| 3a. Fusion training table (EHR + wearable features) | ✅ Done |
| 3b. LightGBM models (alerts + forecasts) | 🏋️ Training (see [TRAINING_LIGHTGBM.md](TRAINING_LIGHTGBM.md)) |
| 3c. GRU deep-learning model | ⏳ Next |
| 4. Doctor dashboard | ⏳ |

## Quick start
```bash
pip install -r requirements.txt
python src/generate_ehr.py --n 5000 --seed 42   # Dataset 1 -> data/ehr/*.csv
python src/generate_wearable.py                 # Dataset 2 -> data/wearable/ (about 40 s)
python src/check_data.py                        # confirms data matches the reference
python src/build_features.py                    # fused training table -> data/model/features.parquet
python src/train_lightgbm.py                    # models -> models/lightgbm, report -> results/lightgbm
```
Validation reports (optional):
```bash
python src/validate_ehr.py                      # -> data/ehr/validation_report.html
python src/process_shanghai.py                  # needs the ShanghaiT2DM download (see docs/wearable_dictionary.md)
python src/validate_wearable.py                 # -> data/wearable/validation_report.html
```

## Data
| Dataset | What | Docs |
|---|---|---|
| 1. EHR | Demographics, diagnoses (ICD-10), labs & vitals (LOINC), medications, genetics, hidden twin physiology | [docs/data_dictionary.md](docs/data_dictionary.md) |
| 2. Wearable / CGM | Glucose, heart rate, steps and sleep stage every 5 min; meals, doses and exercise log | [docs/wearable_dictionary.md](docs/wearable_dictionary.md) |

Both datasets are driven by the same hidden twin parameters (insulin sensitivity, beta-cell function, adherence, ...),
so a patient's lab HbA1c, medications and CGM curve are consistent with each other.
The simulator is validated against Indian clinical literature and against **real patient CGM data (ShanghaiT2DM)**.

## Models
| Model | Role |
|---|---|
| **LightGBM** | Main model: spike / severe / hypo risk in the next 2 h + 30–120 min glucose forecast, with SHAP explanations |
| **GRU** (PyTorch, GPU) | Deep-learning comparison: reads raw 5-min sequences + EHR |

All results are reported on **held-out test patients** (patient-level split in `data/splits.csv`).

## Data privacy
All patient data is synthetic, generated from statistical and clinical rules. No real patient data is used for training.
The public, anonymised ShanghaiT2DM dataset (Zhao et al., *Scientific Data* 2023, CC BY 4.0) is used only to validate realism.

<!-- TODO before submission: full team details, problem statement, tech stack, video link,
     architecture diagram, presentation, license -->
