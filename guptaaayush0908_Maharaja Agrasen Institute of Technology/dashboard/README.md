# Sugar Twin: Doctor Dashboard

A single-page React dashboard on top of the trained models and the personalized twin.

| Area | What it shows | Powered by |
|---|---|---|
| Patient list (left) | Everyone monitored, sorted by current risk, with current glucose and trend | ML ensemble |
| Glucose chart | Last 12 h of CGM, next 2 h forecast with likely range, meals / doses / exercise | ML ensemble (LightGBM + GRU) |
| Right now | Current glucose, spike / low / severe-high risk with plain-language reasons | ML ensemble + LightGBM SHAP |
| Suggestions | Actions with their simulated effect for this patient ("Try it" loads them below) | Personalized twin |
| What if…? | Meal (carbs, food type, timing), walk, skipped dose, compared with doing nothing or the usual meal | ML forecast + twin |
| Twin profile | Carb sensitivity, recovery time, walking effect vs population | Twin fitted on week 1 |
| 14-day overview | Time in range vs targets, typical day (AGP), daily time in range, GMI / CV | CGM data |
| Health record | Latest labs, HbA1c history, medicines, diagnoses | EHR |
| Accuracy | Headline test-set results | results/ |

Patients shown are the 750 **test** patients (never used for training) plus the demo patient **Sunita (NEW001)**,
who was created separately (`src/new_patient_demo.py`). The top bar replays the 14 days: drag the slider or press **Replay**.
**Reveal what really happened** overlays the real glucose after "now" to show how good the forecast was.

## Run it

Prerequisites: the project's Python packages (`pip install -r requirements.txt` in the repo root),
`pip install -r dashboard/backend/requirements.txt`, Node 18+, and the training table
(`python src/build_features.py`, which creates `data/model/features.parquet`).

**Terminal 1, API** (from the repo root; takes ~20 s to load):
```bash
python -m uvicorn app:app --app-dir dashboard/backend --port 8000
```

**Terminal 2, web app:**
```bash
cd dashboard/frontend
npm install
npm run dev
```
Open http://localhost:5173

## Stack
FastAPI · React 19 + TypeScript · Vite · Tailwind CSS 4 · Recharts · lucide-react icons.
