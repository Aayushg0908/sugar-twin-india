# Sugar Twin India: demo video script (about 22 minutes)

For: the Happiest Health Digital Twin Challenge jury. Format: screen recording with voice-over; faces on camera for the intro and close.
Split it between the three team members; each part suggests a presenter (swap freely). Speak slowly, and show more than you tell.

**Before recording**
- Start the API and the dashboard (README section 7). Open http://localhost:5173 in a fresh / incognito window so the "Demo time machine" strip shows.
- Light theme, browser zoom 100%, window about 1440 px wide. Close other tabs and notifications.
- Have these ready in tabs: `docs/architecture.png`, `results/fusion_ablation/report.html`, `results/twin_simulator/report.html`, `results/scenario_tests/report.html`, the GitHub repo.
- Pick the demo moments in advance (see part 5) so nothing is searched for live.

| # | Part | Time | Who |
|---|---|---|---|
| 1 | Hook and team | 0:00 – 1:30 | Aayush Gupta (team leader) |
| 2 | The problem | 1:30 – 4:00 | Dhruv Gandhi |
| 3 | Our digital twin in one picture | 4:00 – 6:30 | Aayush Gupta |
| 4 | The data: two streams, one body | 6:30 – 10:00 | Divyansh Singh |
| 5 | Live demo: the doctor dashboard | 10:00 – 17:00 | Aayush Gupta |
| 6 | Models and results | 17:00 – 20:30 | Dhruv Gandhi |
| 7 | Limitations, next steps, close | 20:30 – 22:00 | Divyansh Singh |

---

## 1. Hook and team (0:00 – 1:30)

**Screen:** team on camera, then the dashboard's patient list.

> "A person with diabetes sees their doctor maybe four times a year. In between, their glucose rises and falls hundreds of times, and nobody is watching.
> What if every patient had a virtual copy of their body that warned the doctor about a dangerous spike two hours before it happened,
> and let the doctor test advice on that copy before giving it?
> We are team guptaaayush0908 from Maharaja Agrasen Institute of Technology: Aayush Gupta, Dhruv Gandhi and Divyansh Singh, and this is **Sugar Twin India**."

Introduce Aayush Gupta, Dhruv Gandhi and Divyansh Singh in one line each (name and what they built).

## 2. The problem (1:30 – 4:00)

**Screen:** a simple slide or the README problem section.

Points to make:
- India has about **101 million people with diabetes** (ICMR-INDIAB 2023). Type 2 diabetes is the condition we chose.
- Harm happens **between visits**: post-meal spikes from rice-heavy meals and late dinners, festivals and weddings, missed doses, illness, and lows on sulfonylureas or insulin.
- CGMs and smartwatches now collect the data, but: a doctor cannot read 288 readings a day for hundreds of patients, and a CGM only shows what **already** happened.
- The EHR knows the patient's history (HbA1c, kidneys, medicines, genetics) but is not connected to the live stream.

> "So the challenge is: fuse the record with the live data, predict the event before it happens, and turn it into something a doctor can act on in seconds."

## 3. Our digital twin in one picture (4:00 – 6:30)

**Screen:** `docs/architecture.png`, zoom into each column as you talk.

Walk left to right:
1. **Data:** a synthetic EHR for 5,000 Indian patients and 14 days of 5-minute CGM and smartwatch data for each of them.
2. **Fusion:** every 30 minutes we combine both streams into 97 features, using only what an app would actually know at that moment.
3. **Models:** LightGBM and a GRU deep-learning model predict spikes above 180, above 250 and lows below 70 in the next 2 hours, plus the glucose curve. Next to them, a **personal physiology twin** learns how each patient's own body responds to food and walking.
4. **Serving and dashboard:** a FastAPI service and a React dashboard for the doctor.

> "Two kinds of twin work together: the ML models are good at *predicting* what will happen; the physiology twin is good at answering *what if we change something*."

## 4. The data: two streams, one body (6:30 – 10:00)

**Screen:** `docs/data_dictionary.md` and `docs/wearable_dictionary.md` on GitHub, then `data/wearable/validation_report.html`.

- **Why synthetic:** the DPDP Act and HIPAA restrict real patient data, and the challenge requires synthetic or open data.
- **EHR (static):** demographics, ICD-10 diagnoses, 3 years of LOINC labs, medicines that follow a realistic treatment ladder, and genetic markers (TCF7L2, FTO, KCNQ1, SLC30A8, polygenic score). Validated against Indian ranges: mean HbA1c 7.8%, 45% hypertension, 89% on metformin.
- **Wearables (dynamic):** CGM, heart rate, HRV, steps and sleep stages every 5 minutes; meals with Indian timings and dishes, feasts, fasting days, illness, missed doses; realistic sensor noise and gaps; only about 60% of meals are logged, like in real life.
- **One body, two streams:** both are generated from the same hidden physiology, so HbA1c, medicines and the CGM curve agree (CGM-estimated vs lab HbA1c r = 0.93). The hidden values are never given to the models.
- **Real-data check:** compared with real CGM data from the ShanghaiT2DM study, time in range 82% vs 84%, time above 180 15% vs 15%, peak 90 min after meals in both.

## 5. Live demo: the doctor dashboard (10:00 – 17:00)

**Screen:** the dashboard. Move the mouse slowly; pause on each thing you name.

1. **Time machine (30 s).** Point at the blue strip: "We replay 14 recorded days. In a clinic this is the live clock." Click *Got it*.
2. **Triage (1 min).** Patient list: "55 urgent, 58 watch." Click the *Urgent* tile, then a red-dot patient: "Low sugar likely." Point out that the list is sorted by risk.
3. **Clinical summary (1 min).** On Sunita (demo patient): read the left column (now and next 2 hours) and one right-column finding: "Low sugar 10% of the time, mostly in the morning: consider reviewing Glimepiride."
4. **Forecast chart (1 min).** Explain: black line is measured CGM, blue dashed line is the forecast, shaded band is the likely range, circles are meals with carbs.
   Click **Reveal what really happened**: "The grey line is what actually happened afterwards; the forecast followed it."
5. **Play the replay (1 min).** Press *Play replay* and let it run for 15–20 seconds while a spike builds: "Watch the High sugar card turn from Unlikely to Possible to Likely, before glucose crosses 180." Pause.
6. **Reasons (30 s).** In *Right now*, read the reasons: "usually peaks near 220 after this time of day", "high for hours". "These come from SHAP explanations of the model, rewritten in plain language."
7. **Twin suggestions and what-if (1.5 min).** Read a suggestion and its simulated effect (e.g. −37 mg/dL peak). Click **Try it**: the what-if panel opens. Show the twin profile (sugar rise per roti, recovery time, walking effect).
   Change the meal from rice to roti-dal, add a 20-minute walk, then skip a dose, and show how the curve and "time above 180" change.
8. **Health record and genetics (1 min).** Open *Health record*: labs, medicines, diagnoses and the HbA1c trend with target bands. Open *Genetic markers*: risk copies per gene and the polygenic score.
9. **Themes (15 s).** Toggle dark mode: "for night shifts."

## 6. Models and results (17:00 – 20:30)

**Screen:** the README results section, then the HTML reports.

- **Honest testing:** 750 test patients never seen in training; the split is by patient, not by time.
- **Spike alert:** AUROC 0.958 (0.924 for T2D only) against 0.71 for a simple trend rule; 95% of spikes flagged in advance; about 1.5 alerts per patient-day.
- **Lows:** AUROC 0.978, 94.5% of lows flagged in advance.
- **Forecast:** ±13.8 mg/dL at 60 minutes vs ±21.3 if you assume glucose stays the same.
- **Fusion helps:** open `results/fusion_ablation/report.html`: CGM only 0.952 → full fusion 0.956; 60-minute error 14.8 → 13.9 mg/dL.
- **Twin:** open `results/twin_simulator/report.html`: predicting a new meal's peak, 25 mg/dL error vs 31 for one-size-fits-all.
- **Why two kinds of twin:** open `results/scenario_tests/report.html`: the ML models pass the clinical scenarios (feasts, missed doses, illness, night lows), but fail some what-if checks, because they learn correlation, not cause and effect. That is exactly why what-if questions go to the physiology twin.
- **Robustness:** still works on day 1, without meal logging, without a smartwatch, and with a noisier sensor.
- **New patient:** Sunita was created outside the dataset; all 48 of her spikes and 18 lows were flagged early.

## 7. Limitations, next steps, close (20:30 – 22:00)

**Screen:** team on camera, or the README limitations section.

Limitations (say them; juries trust teams that do):
- The data is synthetic; real-world accuracy will be lower.
- Low-sugar alerts are sensitive but produce false alarms (about 3 in 4).

Next steps:
- Validate on real, consented CGM data with a partner hospital.
- Re-fit the twin continuously as new meals are logged.
- A patient-side app with an "I'm about to eat" button (our experiment shows it lifts spike AUROC to 0.975 with a meal plan).
- Extend the twin to hypertension and cardiovascular risk using the same EHR + wearable fusion.

> "Sugar Twin turns a stream of numbers into one question a doctor can answer in seconds: *who needs me now, and what should I tell them?* Thank you."

Show the GitHub link on screen for the last 5 seconds.
