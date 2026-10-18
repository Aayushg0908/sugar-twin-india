# Training the LightGBM models (helper laptop guide)

This guide is for training the LightGBM part of **Sugar Twin India** on a separate laptop (no GPU needed).
Total time: about **10 minutes of setup + 30–90 minutes of training**, mostly unattended.

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

> Use the **exact** versions in `requirements.txt` so results match.

## 3. Check the data arrived intact
The datasets (about 120 MB) come with the repo in `data/ehr` and `data/wearable`, so nothing needs generating.
```bash
python src/check_data.py
```
✅ You should see **"All data matches the reference. Safe to train."**
❌ If you see `MISMATCH`, the clone is probably incomplete. Run `git pull` and try again, or tell Aayush.

## 4. (Skip) Regenerating the data
Not needed. For reference only, the same data can be rebuilt from fixed seeds with
`python src/generate_ehr.py --n 5000 --seed 42` and then `python src/generate_wearable.py`.

## 5. Build the training table (about 2–5 minutes)
This fuses the EHR with the wearable data into one row per patient per prediction time.
It's the only file you build yourself. It's derived from the data in git and too big to store (several hundred MB).
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
| `severe` | Glucose goes above 250 in the next 2 h (from below 250) |
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

## 8. After training: test the models before pushing
Do all four checks below. Push only if every one passes.

> 📦 **About the model files:** LightGBM saves trained models as **`.txt` files** (its native format), not `.pkl`.
> This is intentional: `.txt` models load on any OS and any Python version, while `.pkl` files can break across
> versions. So there is **no `.pkl` to push**. The `.txt` files in `models/lightgbm/` *are* the trained models.

### Check 1: all files exist
```bash
git status
```
You should see these new, untracked files:
```
models/lightgbm/spike.txt, severe.txt, hypo.txt, fc30.txt, fc60.txt, fc90.txt, fc120.txt   (7 models)
results/lightgbm/report.html, metrics.json, feature_importance.csv, test_predictions.parquet
```
`data/model/features.parquet` must **not** appear (it's ignored on purpose).

### Check 2: every model loads and predicts
Run this from the repo folder with the venv active. It's one line and works in PowerShell and in macOS/Linux terminals:
```bash
python -c "import lightgbm as lgb, pandas as pd; df=pd.read_parquet('data/model/features.parquet'); te=df[df.split=='test'].head(1000); X=te.drop(columns=['patient_id','timestamp','split','y_spike','y_severe','y_hypo','g_p30','g_p60','g_p90','g_p120']); [print(n, lgb.Booster(model_file=f'models/lightgbm/{n}.txt').predict(X)[:5].round(3)) for n in ['spike','severe','hypo','fc30','fc60','fc90','fc120']]"
```
✅ Pass if it prints 7 lines with no error:
- `spike`, `severe`, `hypo` → numbers **between 0 and 1** (risk probabilities)
- `fc30` … `fc120` → numbers roughly **between -100 and +150** (predicted *change* in glucose, mg/dL)

### Check 3: the results make sense
```bash
python -c "import json; m=json.load(open('results/lightgbm/metrics.json')); print('quick mode:', m['run']['quick']); [print(k, 'AUROC', round(m[k]['auroc'],3), '| trend rule', round(m[k]['baseline_trend_auroc'],3)) for k in ['spike','severe','hypo']]; [print(k, 'MAE', round(m[k]['mae'],1), '| persistence', round(m[k]['baseline_persistence_mae'],1), '| Clarke A+B', round(m[k]['clarke_zones_pct']['A']+m[k]['clarke_zones_pct']['B'],1)) for k in ['fc30','fc60','fc90','fc120']]"
```
✅ Pass if **all** of these are true:
| Check | Expected |
|---|---|
| `quick mode` | `False`. If `True`, these are smoke-test results: rerun step 7 |
| Every AUROC | **Higher** than its trend-rule AUROC, and between about **0.70 and 0.98** |
| Every MAE | **Lower** than its persistence MAE |
| Clarke A+B | Above about **95%** |

⚠️ If any AUROC is **above 0.99**, something is probably leaking future information. Don't push; send Aayush the output.

### Check 4: read the report
Open `results/lightgbm/report.html` in a browser. The tables should have numbers (no blanks or `nan`), and the charts should show
ROC curves above the diagonal and forecast points clustered along the diagonal line of the Clarke grid.

## 9. Push the results
Only after all four checks pass.

**Option A, if you've been added as a collaborator on GitHub:**
```bash
git pull
git checkout -b lightgbm-results
git add models/lightgbm results/lightgbm
git status
git commit -m "LightGBM training results"
git push -u origin lightgbm-results
```
`git status` should list only files under `models/lightgbm/` and `results/lightgbm/`. If anything else appears, unstage it with
`git restore --staged <file>` before committing. Then tell Aayush the branch `lightgbm-results` is pushed.

> If the push fails with "file exceeds GitHub's file size limit" (100 MB), it's almost certainly
> `test_predictions.parquet`. Remove it with `git rm --cached results/lightgbm/test_predictions.parquet`,
> commit and push again, and send that one file separately (Option B).

**Option B, if you can't push:** zip the `models/lightgbm` and `results/lightgbm` folders and send the zip (Google Drive / WhatsApp / email).

If you also run the optional 15-min training afterwards, repeat steps 8 and 9 on a new branch, e.g. `lightgbm-results-15min`.

---

## Troubleshooting
| Problem | Fix |
|---|---|
| `MemoryError` or the laptop freezes | Close other apps; make sure you used the default `build_features.py` (every 30 min) |
| `ModuleNotFoundError` | The venv isn't active. Re-run the activate command from step 2 |
| `lightgbm` import error on macOS | `brew install libomp`, then `pip install --force-reinstall lightgbm==4.7.0` |
| `FileNotFoundError: data/model/features.parquet` | Run step 5 before step 6/7 |
| Training looks stuck | It's probably not. Each model prints every 200 rounds and the big ones take several minutes |
| `check_data.py` mismatch | Incomplete clone. Run `git pull`, or re-clone the repo |

## What the training does (short version)
- **Inputs (about 100 features):** current glucose and its trend, last 6 h glucose statistics, this patient's usual glucose at this time of day,
  logged meals and carbs, logged medicine doses, steps, heart rate, last night's sleep and HRV, time of day,
  plus EHR facts (HbA1c, diabetes duration, medicines, kidney function, BMI, genetic risk...).
- **No cheating:** the model only sees what a real app would know at that moment. Hidden simulator truth (true carbs,
  illness days, twin physiology) is never used.
- **Honest testing:** all reported numbers are on **test patients the model never saw**.
