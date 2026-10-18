"""
Robustness tests: how do the models hold up under real-world deployment problems?
All on held-out TEST patients; labels always come from the original CGM.

  1. Cold start      - a brand-new user's first days (no personal history yet)
  2. No meal logging - the patient never logs meals in the app
  3. No smartwatch   - CGM only: no heart rate, steps, sleep or auto-detected exercise
  4. Noisier sensor  - CGM readings with ~10% extra error (cheaper / older sensor)

Usage:
    python src/test_robustness.py   # -> results/robustness/{report.html, results.json}
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_features as bf  # noqa: E402
import train_gru as tg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "robustness"
ID_COLS = ["patient_id", "timestamp", "split"]
TARGETS = ["y_spike", "y_severe", "y_hypo", "g_p30", "g_p60", "g_p90", "g_p120"]

MEAL_FEATS = {"carbs_1h": 0.0, "carbs_3h": 0.0, "carbs_today": 0.0, "last_meal_carbs": np.nan, "last_meal_gi": np.nan,
              "min_since_meal": 720.0}
WATCH_FEATS = ["hr_15m", "hr_1h", "hr_vs_resting", "steps_30m", "steps_1h", "steps_3h", "asleep_now", "min_since_sleep",
               "sleep_hours_last_night", "hrv_last_night", "sleep_eff_last_night", "resting_hr_last_night", "steps_yesterday",
               "min_since_exercise", "exercise_min_3h"]
# GRU channels: 0 glucose, 1 glucose missing, 2 HR, 3 HR missing, 4 steps, 5 asleep, 6 carbs, 7 dose, 8 exercise, 9-10 time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Models:
    def __init__(self, device):
        self.device = device
        self.lgb = {k: lgb.Booster(model_file=str(ROOT / "models" / "lightgbm" / f"{k}.txt")) for k in ("spike", "hypo", "fc60")}
        ck = torch.load(ROOT / "models" / "gru" / "gru.pt", map_location=device, weights_only=False)
        tg.SEQ_LEN = ck["seq_len"]
        self.ck = ck
        self.mean, self.std = np.array(ck["static_mean"], np.float32), np.array(ck["static_std"], np.float32)
        self.feat_cols = self.lgb["spike"].feature_name()

    def gru_model(self, n_seq):
        m = tg.TwinGRU(n_seq, len(self.ck["static_features"]), self.ck["config"]).to(self.device)
        m.load_state_dict(self.ck["state_dict"])
        return m.eval()

    def predict(self, rows, seq_all, pid_index):
        X = rows[self.feat_cols]
        out = {"LightGBM": {"spike": self.lgb["spike"].predict(X), "hypo": self.lgb["hypo"].predict(X),
                            "fc60": rows.g_now.to_numpy() + self.lgb["fc60"].predict(X)}}
        d = rows.copy()
        d["p_idx"] = d.patient_id.map(pid_index).astype(np.int64)
        d["t_idx"] = ((d.timestamp - bf.START) / pd.Timedelta(minutes=bf.STEP)).astype(np.int64)
        tens = tg.to_tensors(d, self.mean, self.std, self.device)
        c, r = tg.predict(self.gru_model(seq_all.shape[2]), seq_all, tens)
        out["GRU"] = {"spike": c[:, 0], "hypo": c[:, 2], "fc60": rows.g_now.to_numpy() + r[:, 1]}
        out["Ensemble"] = {k: (out["LightGBM"][k] + out["GRU"][k]) / 2 for k in ("spike", "hypo", "fc60")}
        return out


def score(rows, preds, mask=None):
    mask = np.ones(len(rows), bool) if mask is None else mask
    res = {}
    for model, p in preds.items():
        r = {}
        for alert, col in (("spike", "y_spike"), ("hypo", "y_hypo")):
            m = mask & rows[col].notna().to_numpy()
            y = rows[col].to_numpy()[m]
            r[f"{alert}_auroc"] = float(roc_auc_score(y, p[alert][m])) if 0 < y.sum() < len(y) else None
        m = mask & rows.g_p60.notna().to_numpy()
        r["mae60"] = float(np.abs(p["fc60"][m] - rows.g_p60.to_numpy()[m]).mean())
        res[model] = r
    return res


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = Models(device)
    rows = pd.read_parquet(bf.OUT / "features.parquet", filters=[("split", "==", "test")]).reset_index(drop=True)
    test_ids = sorted(rows.patient_id.unique())
    log(f"{len(rows):,} test rows from {len(test_ids)} patients")
    pids, seq_all = tg.build_sequences(device)
    pid_index = {p: i for i, p in enumerate(pids)}
    test_idx = torch.tensor([pid_index[p] for p in test_ids], device=device)

    results = {}
    base = models.predict(rows, seq_all, pid_index)
    results["baseline"] = score(rows, base)
    log(f"baseline: {results['baseline']['Ensemble']}")

    # 1. Cold start: performance by day since the sensor was started
    day = ((rows.timestamp - bf.START).dt.days).to_numpy()
    results["cold_start"] = {name: score(rows, base, (day >= lo) & (day <= hi))
                             for name, (lo, hi) in {"day 1-2": (0, 1), "day 3-7": (2, 6), "day 8-14": (7, 13)}.items()}
    log("cold start done")

    # 2. No meal logging
    r2 = rows.copy()
    for c, v in MEAL_FEATS.items():
        r2[c] = np.float32(v)
    s2 = seq_all.clone()
    s2[test_idx, :, 6] = 0
    results["no_meal_logging"] = score(rows, models.predict(r2, s2, pid_index))
    del s2, r2
    log("no meal logging done")

    # 3. No smartwatch (CGM only)
    r3 = rows.copy()
    for c in WATCH_FEATS:
        r3[c] = np.nan
    s3 = seq_all.clone()
    s3[test_idx, :, 2] = 0
    s3[test_idx, :, 3] = 1
    s3[test_idx, :, 4] = 0
    s3[test_idx, :, 5] = 0
    s3[test_idx, :, 8] = 0
    results["no_smartwatch"] = score(rows, models.predict(r3, s3, pid_index))
    del s3, r3
    log("no smartwatch done")

    # 4. Noisier sensor: add ~10% correlated error to the CGM, rebuild features / sequences from it
    rng = np.random.default_rng(5)
    _, cgm = bf.load_stream(bf.W / "cgm.parquet", ["glucose_mgdl"])
    _, vit = bf.load_stream(bf.W / "vitals.parquet", ["heart_rate_bpm", "steps", "sleep_stage"])
    sel = np.array([pid_index[p] for p in test_ids])
    G = cgm["glucose_mgdl"][sel]
    e = lfilter([np.sqrt(1 - 0.7 ** 2) * 0.10], [1, -0.7], rng.normal(0, 1, G.shape), axis=1)
    Gn = np.where(np.isnan(G), np.nan, np.clip(np.round(G * (1 + e)), 40, 400)).astype(np.float32)
    ev = pd.read_csv(bf.W / "events.csv", parse_dates=["timestamp"],
                     usecols=["patient_id", "timestamp", "event_type", "logged", "logged_carbs_g", "glycaemic_index", "duration_min"])
    ev["bin"] = ((ev.timestamp - bf.START) / pd.Timedelta(minutes=bf.STEP)).astype(int)
    ev = ev[(ev.bin >= 0) & (ev.bin < bf.T)]
    ev["logged"] = ev.logged.astype(bool)
    daily = pd.read_csv(bf.W / "daily_summary.csv")
    ehr = bf.prepare_ehr(pd.read_csv(bf.EHR / "patient_features.csv"), pd.read_csv(bf.EHR / "medications.csv"), bf.make_splits())
    parts = []
    for c0 in range(0, len(test_ids), bf.CHUNK):
        sl = slice(c0, c0 + bf.CHUNK)
        p = test_ids[sl]
        parts.append(bf.build_chunk(p, Gn[sl], vit["heart_rate_bpm"][sel][sl], vit["steps"][sel][sl],
                                    vit["sleep_stage"][sel][sl], ev, daily[daily.patient_id.isin(p)], ehr, 30))
    noisy = pd.concat(parts).drop(columns=TARGETS)
    noisy = rows[ID_COLS + TARGETS].merge(noisy, on=ID_COLS, how="inner")     # keep the TRUE labels
    s4 = seq_all.clone()
    gch = torch.tensor(np.nan_to_num((Gn - 140) / 50), device=device, dtype=seq_all.dtype)
    s4[test_idx, :, 0] = gch
    results["noisy_sensor"] = score(noisy, models.predict(noisy, s4, pid_index))
    mard_extra = float(np.nanmean(np.abs(Gn - G) / G) * 100)
    results["noisy_sensor"]["_extra_mard_pct"] = mard_extra
    del s4
    log(f"noisy sensor done (extra error {mard_extra:.1f}%)")

    # ---- pass / fail
    b = results["baseline"]["Ensemble"]
    checks = []
    cs = results["cold_start"]["day 1-2"]["Ensemble"]
    checks.append(("1. Cold start: spike AUROC on day 1-2 >= 0.90", cs["spike_auroc"] >= 0.90,
                   f"day 1-2 {cs['spike_auroc']:.3f} vs day 8-14 {results['cold_start']['day 8-14']['Ensemble']['spike_auroc']:.3f}"))
    for key, label, max_drop in [("no_meal_logging", "2. No meal logging", 0.03), ("no_smartwatch", "3. No smartwatch", 0.03),
                                 ("noisy_sensor", "4. Noisier sensor (+10% error)", 0.03)]:
        r = results[key]["Ensemble"]
        drop = b["spike_auroc"] - r["spike_auroc"]
        checks.append((f"{label}: spike AUROC drops <= {max_drop}", drop <= max_drop,
                       f"{b['spike_auroc']:.3f} -> {r['spike_auroc']:.3f} (drop {drop:.3f}); hypo {b['hypo_auroc']:.3f} -> "
                       f"{r['hypo_auroc']:.3f}; MAE60 {b['mae60']:.1f} -> {r['mae60']:.1f}"))
    results["checks"] = [{"test": n, "pass": bool(o), "detail": d} for n, o, d in checks]
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_report(results, checks)
    for n, o, d in checks:
        print(f"  {'PASS' if o else 'FAIL'}  {n}: {d}")
    log(f"Done in {(time.time() - t0) / 60:.1f} min -> {OUT / 'report.html'}")


def write_report(r, checks):
    def fmt(v, d=3):
        return "&ndash;" if v is None else f"{v:.{d}f}"
    rows = ""
    scen = [("Baseline (everything available)", r["baseline"]), ("Cold start: day 1-2", r["cold_start"]["day 1-2"]),
            ("Cold start: day 3-7", r["cold_start"]["day 3-7"]), ("Cold start: day 8-14", r["cold_start"]["day 8-14"]),
            ("No meal logging", r["no_meal_logging"]), ("No smartwatch (CGM only)", r["no_smartwatch"]),
            (f"Noisier sensor (+{r['noisy_sensor']['_extra_mard_pct']:.0f}% error)", r["noisy_sensor"])]
    for name, res in scen:
        rows += f"<tr><td>{name}</td>" + "".join(
            f"<td>{fmt(res[m]['spike_auroc'])}</td><td>{fmt(res[m]['hypo_auroc'])}</td><td>{fmt(res[m]['mae60'], 1)}</td>"
            for m in ("LightGBM", "GRU", "Ensemble")) + "</tr>"
    chk = "".join(f"<tr><td>{n}</td><td class='{'ok' if o else 'bad'}'>{'PASS' if o else 'FAIL'}</td><td>{d}</td></tr>" for n, o, d in checks)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Robustness Tests</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1150px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}} td,th{{border-bottom:1px solid #e2e8f0;padding:5px 7px;text-align:left}}
.ok{{color:#15803d;font-weight:700}} .bad{{color:#b91c1c;font-weight:700}} .muted{{color:#64748b}}</style></head><body>
<h1>Robustness Tests &mdash; {sum(o for _, o, _ in checks)}/{len(checks)} passed</h1>
<p class="muted">Held-out test patients. Labels always come from the original CGM. Models are NOT retrained for these
conditions: this measures how the deployed models cope.</p>
<table><tr><th rowspan=2>Condition</th><th colspan=3>LightGBM</th><th colspan=3>GRU</th><th colspan=3>Ensemble</th></tr>
<tr>{'<th>Spike AUROC</th><th>Hypo AUROC</th><th>MAE 60 min</th>' * 3}</tr>{rows}</table>
<table><tr><th>Check</th><th>Result</th><th>Detail</th></tr>{chk}</table></body></html>"""
    (OUT / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
