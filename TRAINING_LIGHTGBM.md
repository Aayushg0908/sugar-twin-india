# Training the LightGBM models (helper laptop guide)

This guide is for training the LightGBM part of **Sugar Twin India** on a separate laptop (no GPU needed).
Total time: about **15 minutes of setup + 30–90 minutes of training**, mostly unattended.

**You need:** Windows, macOS or Linux · **16 GB RAM** recommended · about **5 GB free disk** ·
**Python 3.11–3.13** (3.13 tested) · **Git** · an internet connection for the setup only.

> ⚡ Keep the laptop **plugged in** and stop it from **sleeping** during training.
> On Windows go to Settings → System → Power → Screen and sleep → *Never*. On macOS run `caffeinate -i` in a second terminal.

---

## 1. Get the code
```bash
git clone https://github.com/Aayushg0908/sugar-twin-india.git
cd sugar-twin-india
```

## 2. Create a clean Python environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks the activate script, run this once, then try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
macOS only: LightGBM needs OpenMP, so run `brew install libomp` first.

> Use the **exact** versions in `requirements.txt`. The data is regenerated from random seeds,
> and different numpy/pandas versions can produce slightly different data.

## 3. Regenerate the datasets (about 2 minutes)
The data is not stored in git (it's about 100 MB). It is rebuilt identically from fixed seeds:
```bash
python src/generate_ehr.py --n 5000 --seed 42
python src/generate_wearable.py
```

## 4. Check the data matches the reference
```bash
python src/check_data.py
```
✅ You should see **"All data matches the reference. Safe to train."**
❌ If you see `MISMATCH`, check you're inside the `.venv` and installed `requirements.txt`, then repeat step 3.
If it still mismatches, you can continue, but tell Aayush.

## 5. Build the training table (about 2–5 minutes)
This fuses the EHR with the wearable data into one row per patient per prediction time.
```bash
python src/build_features.py
```
Output: `data/model/features.parquet`, about 3.2 million rows × 107 columns. It uses the patient split in `data/splits.csv`
(70% train / 15% validation / 15% test), which is committed to git so both laptops use the same patients.

## 6. Smoke test first (about 3–5 minutes)
Catches problems before the long run:
```bash
python src/train_lightgbm.py --quick
```
It should finish with `Done in X min. Results -> .../results/lightgbm`.
Open `results/lightgbm/report.html` in a browser and check the tables have numbers.

## 7. Full training (about 30–90 minutes)
```bash
python src/train_lightgbm.py
```
It trains 7 models one after another and prints progress every 200 rounds:

| Model | Predicts |
|---|---|
| `spike` | Glucose goes above 180 in the next 2 h (**main alert**) |
| `severe` | Glucose goes above 250 in the next 2 h |
| `hypo` | Glucose goes below 70 in the next 2 h |
| `fc30`, `fc60`, `fc90`, `fc120` | Glucose value 30 / 60 / 90 / 120 min ahead |

Each model stops by itself when the validation score stops improving (early stopping).

### Optional: denser training (only after step 7 works)
Prediction points every 15 min instead of 30 doubles the data (about 6.4M rows, needs the full 16 GB):
```bash
python src/build_features.py --every 15
python src/train_lightgbm.py
```
Close Chrome and other heavy apps first. This overwrites the step 7 results, so **send those back first**.

## 8. Send the results back
These are the files Aayush needs:
```
models/lightgbm/*.txt                       (7 trained models)
results/lightgbm/report.html                (charts + tables)
results/lightgbm/metrics.json               (all numbers)
results/lightgbm/feature_importance.csv
results/lightgbm/test_predictions.parquet   (for combining with the GRU model)
```
**Option A, if you've been added as a collaborator on GitHub:**
```bash
git checkout -b lightgbm-results
git add models/lightgbm results/lightgbm
git commit -m "LightGBM training results"
git push -u origin lightgbm-results
```
**Option B:** zip the `models/lightgbm` and `results/lightgbm` folders and send the zip (Google Drive / WhatsApp / email).

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `MemoryError` or the laptop freezes | Close other apps; make sure you used the default `build_features.py` (every 30 min) |
| `ModuleNotFoundError` | The venv isn't active. Re-run the activate command from step 2 |
| `lightgbm` import error on macOS | `brew install libomp`, then `pip install --force-reinstall lightgbm==4.7.0` |
| `FileNotFoundError: data/...` | Run the steps in order: 3 → 5 → 7 |
| Training looks stuck | It's probably not. Each model prints every 200 rounds and the big ones take several minutes |
| `check_data.py` mismatch | Wrong package versions. Recreate the venv and reinstall `requirements.txt` |

## What the training does (short version)
- **Inputs (about 100 features):** current glucose and its trend, last 6 h glucose statistics, this patient's usual glucose at this time of day,
  logged meals and carbs, logged medicine doses, steps, heart rate, last night's sleep and HRV, time of day,
  plus EHR facts (HbA1c, diabetes duration, medicines, kidney function, BMI, genetic risk...).
- **No cheating:** the model only sees what a real app would know at that moment. Hidden simulator truth (true carbs,
  illness days, twin physiology) is never used.
- **Honest testing:** all reported numbers are on **test patients the model never saw**.
