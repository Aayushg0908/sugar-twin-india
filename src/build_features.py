"""
Build the model training table by fusing the two data streams:
  * static / historical: EHR (labs, diagnoses, medications, genetics)
  * dynamic / real-time: CGM + smartwatch + logged events

One row = one patient at one prediction time. Prediction times are every
--every minutes (default 30) once 6 h of history is available.

Only information a real system would have at prediction time is used:
past CGM / watch data, what the patient LOGGED in the app, and the EHR.
Hidden simulator truth (twin parameters, true carbs, day context) is never used.

Targets (next 2 hours):
  y_spike   glucose now < 180 and any reading > 180       (primary)
  y_severe  any reading > 250
  y_hypo    glucose now >= 70 and >= 10 min below 70
  g_p30 .. g_p120   glucose 30 / 60 / 90 / 120 min ahead  (forecast)

Usage:
    python src/build_features.py               # every 30 min -> data/model/features.parquet
    python src/build_features.py --every 15    # denser (needs ~16 GB RAM for training)
    python src/build_features.py --limit 300   # quick test
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view as swv

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "data" / "wearable"
EHR = ROOT / "data" / "ehr"
OUT = ROOT / "data" / "model"
SPLITS = ROOT / "data" / "splits.csv"

STEP = 5
BPD = 288
N_DAYS = 14
T = N_DAYS * BPD
HORIZON = 24          # 2 h ahead
HIST = 72             # 6 h history
START = pd.Timestamp("2026-09-17")
CHUNK = 500
GI_CODE = {"low": 0, "medium": 1, "high": 2}

EHR_NUMERIC = ["age", "bmi", "waist_cm", "diabetes_duration_yrs", "last_hba1c", "hba1c_mean", "hba1c_change",
               "last_fasting_plasma_glucose", "last_post_prandial_glucose_2h", "last_egfr_ckd_epi_2021",
               "last_urine_albumin_creatinine_ratio", "last_ldl_cholesterol", "last_hdl_cholesterol",
               "last_triglycerides", "last_systolic_blood_pressure", "last_alt", "n_glucose_drugs",
               "polygenic_risk_score_z", "TCF7L2_rs7903146_T_count"]
EHR_FLAGS = ["on_biguanide", "on_sulfonylurea", "on_dpp4_inhibitor", "on_sglt2_inhibitor", "on_basal_insulin",
             "on_statin", "on_arb", "dx_I10", "dx_N18", "dx_E66", "dx_K76", "dx_I25", "dx_E78"]
EHR_CATEG = {"sex": {"F": 0, "M": 1}, "cohort": {"Normoglycemic": 0, "Prediabetes": 1, "T2D": 2},
             "setting": {"Rural": 0, "Urban": 1}, "activity_level": {"Sedentary": 0, "Moderate": 1, "Active": 2},
             "smoking": {"Never": 0, "Former": 1, "Current": 2}, "staple": {"wheat": 0, "mixed": 1, "rice": 2},
             "diet": {"Vegetarian": 0, "Non-vegetarian": 1}}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_splits():
    """Patient-level 70/15/15 split, stratified by cohort. Committed to git so every machine uses the same one."""
    if SPLITS.exists():
        return pd.read_csv(SPLITS)
    pats = pd.read_csv(EHR / "patients.csv")[["patient_id", "cohort"]]
    rng = np.random.default_rng(2026)
    parts = []
    for _, g in pats.groupby("cohort"):
        ids = rng.permutation(g.patient_id.to_numpy())
        n = len(ids)
        parts.append(pd.DataFrame({"patient_id": ids, "split": ["train"] * int(0.7 * n) + ["val"] * int(0.15 * n)
                                   + ["test"] * (n - int(0.7 * n) - int(0.15 * n))}))
    s = pd.concat(parts).sort_values("patient_id")
    s.to_csv(SPLITS, index=False)
    return s


def load_stream(path: Path, cols: list[str]):
    """Read a 5-min parquet stream into (patients x T) matrices (file is patient-major, T rows each)."""
    tab = pq.read_table(path, columns=["patient_id"] + cols, read_dictionary=["patient_id"])
    n = tab.num_rows // T
    assert n * T == tab.num_rows, "unexpected stream length"
    first = tab.column("patient_id").take(pa.array(np.arange(0, n * T, T))).to_pylist()
    last = tab.column("patient_id").take(pa.array(np.arange(T - 1, n * T, T))).to_pylist()
    assert first == last, "stream is not patient-major"
    out = {}
    for c in cols:
        col = tab.column(c)
        if pa.types.is_dictionary(col.type) or pa.types.is_string(col.type):
            out[c] = pc.is_valid(col).to_numpy(zero_copy_only=False).reshape(n, T)
        else:
            out[c] = col.to_numpy(zero_copy_only=False).astype(np.float32).reshape(n, T)
    return [str(p) for p in first], out


def ffill(a, limit):
    """Forward-fill NaNs along axis 1 for at most `limit` steps."""
    idx = np.where(~np.isnan(a), np.arange(a.shape[1]), -10 ** 6)
    idx = np.maximum.accumulate(idx, axis=1)
    ok = (np.arange(a.shape[1]) - idx) <= limit
    out = np.take_along_axis(a, np.clip(idx, 0, None), axis=1)
    out[~ok] = np.nan
    return out


def roll_sum(a, w, ts):
    """Sum of non-NaN values in the window (t-w, t] at indices ts, via cumulative sums."""
    cs = np.concatenate([np.zeros((a.shape[0], 1), np.float64), np.nancumsum(a, axis=1, dtype=np.float64)], axis=1)
    return (cs[:, ts + 1] - cs[:, np.maximum(ts + 1 - w, 0)]).astype(np.float32)


def roll_count(a, w, ts):
    return roll_sum((~np.isnan(a)).astype(np.float32), w, ts)


def since_last(mask, ts, cap):
    """Minutes since mask was last True (at or before t), capped."""
    idx = np.where(mask, np.arange(mask.shape[1]), -10 ** 6)
    idx = np.maximum.accumulate(idx, axis=1)[:, ts]
    return np.minimum((ts - idx) * STEP, cap).astype(np.float32), idx


def window(a, w, ts):
    """(n, len(ts), w) windows ending at each t (inclusive)."""
    return swv(a, w, axis=1)[:, ts - w + 1]


def day_profile(a):
    """Mean of `a` at the same time of day over PREVIOUS days only (no look-ahead)."""
    n = a.shape[0]
    d = a.reshape(n, N_DAYS, BPD)
    v = np.nan_to_num(d)
    c = (~np.isnan(d)).astype(np.float32)
    cs_v = np.cumsum(v, axis=1) - v
    cs_c = np.cumsum(c, axis=1) - c
    with np.errstate(invalid="ignore", divide="ignore"):
        prof = cs_v / cs_c
    return prof.reshape(n, T).astype(np.float32)


def smooth(a, w=7):
    k = np.ones(w, np.float32)
    v = np.apply_along_axis(lambda r: np.convolve(np.nan_to_num(r), k, "same"), 1, a)
    c = np.apply_along_axis(lambda r: np.convolve((~np.isnan(r)).astype(np.float32), k, "same"), 1, a)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, v / c, np.nan).astype(np.float32)


def event_matrix(ev, rows, value, n):
    m = np.zeros((n, T), np.float32)
    np.add.at(m, (rows, ev.bin.to_numpy()), value)
    return m


# --------------------------------------------------------------------------
# main feature builder for one chunk of patients
# --------------------------------------------------------------------------
def build_chunk(pids, G, HR, ST, AS, ev, daily, ehr, every):
    n = len(pids)
    ts = np.arange(HIST + 12, T - HORIZON, every // STEP)
    nt = len(ts)
    row = {p: i for i, p in enumerate(pids)}
    f = {}

    # ---------------- CGM
    Gf = ffill(G, 3)
    g_now = Gf[:, ts]
    f["g_now"] = g_now
    for lag, name in [(3, "15"), (6, "30"), (12, "60"), (24, "120")]:
        f[f"g_delta_{name}"] = g_now - Gf[:, ts - lag]
    f["g_accel_15"] = f["g_delta_15"] - (Gf[:, ts - 3] - Gf[:, ts - 6])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for w, name in [(12, "1h"), (36, "3h"), (72, "6h")]:
            win = window(G, w, ts)
            f[f"g_mean_{name}"] = np.nanmean(win, 2)
            f[f"g_std_{name}"] = np.nanstd(win, 2)
            f[f"g_max_{name}"] = np.nanmax(win, 2)
            f[f"g_min_{name}"] = np.nanmin(win, 2)
        win6 = window(G, 72, ts)
        valid6 = (~np.isnan(win6)).sum(2)
        f["g_tir_6h"] = ((win6 >= 70) & (win6 <= 180)).sum(2) / np.maximum(valid6, 1)
        f["g_tar_6h"] = (win6 > 180).sum(2) / np.maximum(valid6, 1)
        f["g_coverage_6h"] = valid6 / 72
        s24, c24 = roll_sum(G, BPD, ts), roll_count(G, BPD, ts)
        sq24 = roll_sum(G ** 2, BPD, ts)
        f["g_mean_24h"] = s24 / np.maximum(c24, 1)
        f["g_cv_24h"] = np.sqrt(np.maximum(sq24 / np.maximum(c24, 1) - f["g_mean_24h"] ** 2, 0)) / np.maximum(f["g_mean_24h"], 1)
        f["min_since_above_180"], _ = since_last(G > 180, ts, 1440)
        f["min_since_below_70"], _ = since_last(G < 70, ts, 1440)

        # personal pattern: what usually happens to THIS patient at this time of day (previous days only)
        fut_all = np.full((n, T), np.nan, np.float32)
        fut_all[:, :T - HORIZON] = np.nanmax(swv(G, HORIZON, axis=1)[:, 1:T - HORIZON + 1], 2)
        prof_g = day_profile(smooth(G))
        prof_fut = day_profile(fut_all)
        prof_spike = day_profile(np.where(np.isnan(fut_all), np.nan, (fut_all > 180).astype(np.float32)))
        f["prof_glucose_tod"] = prof_g[:, ts]
        f["prof_future_max_tod"] = prof_fut[:, ts]
        f["prof_spike_rate_tod"] = prof_spike[:, ts]
        f["g_vs_profile"] = g_now - prof_g[:, ts]
        f["g_same_time_yesterday"] = Gf[:, np.maximum(ts - BPD, 0)]

    # ---------------- smartwatch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f["hr_15m"] = np.nanmean(window(HR, 3, ts), 2)
        f["hr_1h"] = np.nanmean(window(HR, 12, ts), 2)
    f["steps_30m"] = roll_sum(ST, 6, ts)
    f["steps_1h"] = roll_sum(ST, 12, ts)
    f["steps_3h"] = roll_sum(ST, 36, ts)
    f["asleep_now"] = AS[:, ts].astype(np.float32)
    f["min_since_sleep"], _ = since_last(AS, ts, 1440)

    day = ts // BPD
    hour = (ts % BPD) * STEP / 60
    dly = daily.copy()
    dly["r"] = dly.patient_id.map(row)
    dly["d"] = (pd.to_datetime(dly.date) - START).dt.days
    def daily_mat(col):
        m = np.full((n, N_DAYS), np.nan, np.float32)
        m[dly.r.to_numpy(), dly.d.to_numpy()] = dly[col].to_numpy()
        return m
    morning_done = np.where(hour >= 11, day, day - 1)                # last night's sleep is final by ~11:00
    for col, name in [("sleep_hours", "sleep_hours_last_night"), ("hrv_rmssd_ms", "hrv_last_night"),
                      ("sleep_efficiency_pct", "sleep_eff_last_night"), ("resting_hr", "resting_hr_last_night")]:
        m = daily_mat(col)
        f[name] = np.where(morning_done >= 0, m[:, np.maximum(morning_done, 0)], np.nan)
    m = daily_mat("steps")
    f["steps_yesterday"] = np.where(day >= 1, m[:, np.maximum(day - 1, 0)], np.nan)
    rhr = daily_mat("resting_hr")                                    # personal baseline from PAST days only
    v, c = np.nan_to_num(rhr), (~np.isnan(rhr)).astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        rest_prev = (np.cumsum(v, 1) - v) / (np.cumsum(c, 1) - c)
    f["hr_vs_resting"] = f["hr_15m"] - rest_prev[:, day]

    # ---------------- logged events (what the app knows)
    ev = ev[ev.patient_id.isin(row)]
    meals = ev[(ev.event_type == "meal") & ev.logged]
    r = meals.patient_id.map(row).to_numpy()
    carbs = event_matrix(meals, r, meals.logged_carbs_g.fillna(0).to_numpy(), n)
    gi = np.full((n, T), np.nan, np.float32)
    gi[r, meals.bin.to_numpy()] = meals.glycaemic_index.map(GI_CODE).to_numpy()
    f["min_since_meal"], idx = since_last(carbs > 0, ts, 720)
    has = idx >= 0
    f["last_meal_carbs"] = np.where(has, carbs[np.arange(n)[:, None], np.maximum(idx, 0)], np.nan)
    f["last_meal_gi"] = np.where(has, gi[np.arange(n)[:, None], np.maximum(idx, 0)], np.nan)
    f["carbs_1h"] = roll_sum(carbs, 12, ts)
    f["carbs_3h"] = roll_sum(carbs, 36, ts)
    since_midnight = (ts % BPD) + 1
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(carbs, 1)], 1)
    f["carbs_today"] = (cs[:, ts + 1] - cs[:, ts + 1 - since_midnight]).astype(np.float32)

    doses = ev[(ev.event_type == "medication") & ev.logged]
    dm = event_matrix(doses, doses.patient_id.map(row).to_numpy(), 1.0, n)
    f["min_since_dose_logged"], _ = since_last(dm > 0, ts, 1440)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(dm, 1)], 1)
    f["doses_logged_today"] = (cs[:, ts + 1] - cs[:, ts + 1 - since_midnight]).astype(np.float32)

    ex = ev[ev.event_type == "exercise"]
    exm = np.zeros((n, T), bool)
    for rr, b, dur in zip(ex.patient_id.map(row), ex.bin, ex.duration_min // STEP):
        exm[rr, b:b + int(dur)] = True
    f["min_since_exercise"], _ = since_last(exm, ts, 1440)
    f["exercise_min_3h"] = roll_sum(exm.astype(np.float32), 36, ts) * STEP

    # ---------------- time
    f["hour"] = np.broadcast_to(hour, (n, nt))
    f["hour_sin"] = np.broadcast_to(np.sin(2 * np.pi * hour / 24), (n, nt))
    f["hour_cos"] = np.broadcast_to(np.cos(2 * np.pi * hour / 24), (n, nt))
    dow = np.array([(START + pd.Timedelta(days=int(d))).dayofweek for d in day])
    f["weekend"] = np.broadcast_to((dow >= 5).astype(np.float32), (n, nt))

    # ---------------- targets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fut = swv(G, HORIZON, axis=1)[:, ts + 1]
        nvalid = (~np.isnan(fut)).sum(2)
        fmax, fmin = np.nanmax(fut, 2), np.nanmin(fut, 2)
    low = fut < 70
    sustained_low = (low[..., 1:] & low[..., :-1]).any(2)
    enough = nvalid >= HORIZON // 2
    y = {}
    y["y_spike"] = np.where(enough & (g_now < 180), (fmax > 180).astype(np.float32), np.nan)
    y["y_severe"] = np.where(enough, (fmax > 250).astype(np.float32), np.nan)
    y["y_hypo"] = np.where(enough & (g_now >= 70), sustained_low.astype(np.float32), np.nan)
    for lag in (6, 12, 18, 24):
        y[f"g_p{lag * STEP}"] = G[:, ts + lag]

    # ---------------- assemble
    df = pd.DataFrame({k: np.ascontiguousarray(v, dtype=np.float32).ravel() for k, v in {**f, **y}.items()})
    df.insert(0, "patient_id", np.repeat(np.array(pids), nt))
    df.insert(1, "timestamp", np.tile((START + pd.to_timedelta(ts * STEP, unit="min")).to_numpy(), n))
    df = df.merge(ehr, on="patient_id", how="left")
    return df[df.g_now.notna()]


def main(every, limit):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    splits = make_splits()

    feats = pd.read_csv(EHR / "patient_features.csv")
    ehr = feats[["patient_id"] + EHR_NUMERIC + EHR_FLAGS].copy()
    for col, mapping in EHR_CATEG.items():
        ehr[col] = feats[col].map(mapping)
    ehr[EHR_NUMERIC + EHR_FLAGS + list(EHR_CATEG)] = ehr[EHR_NUMERIC + EHR_FLAGS + list(EHR_CATEG)].astype(np.float32)
    ehr["prescribed_doses_per_day"] = 0.0
    meds = pd.read_csv(EHR / "medications.csv")
    act = meds[meds.end_date.isna() & (meds.indication == "Type 2 diabetes")]
    per_day = act.dose.str.contains("BD").map({True: 2, False: 1}).groupby(act.patient_id).sum()
    ehr["prescribed_doses_per_day"] = ehr.patient_id.map(per_day).fillna(0).astype(np.float32)
    ehr = ehr.merge(splits[["patient_id", "split"]], on="patient_id")

    pids, cgm = load_stream(W / "cgm.parquet", ["glucose_mgdl"])
    _, vit = load_stream(W / "vitals.parquet", ["heart_rate_bpm", "steps", "sleep_stage"])
    print(f"Loaded streams for {len(pids)} patients ({time.time() - t0:.0f}s)")

    ev = pd.read_csv(W / "events.csv", parse_dates=["timestamp"],
                     usecols=["patient_id", "timestamp", "event_type", "logged", "logged_carbs_g", "glycaemic_index",
                              "duration_min"])
    ev["bin"] = ((ev.timestamp - START) / pd.Timedelta(minutes=STEP)).astype(int)
    ev = ev[(ev.bin >= 0) & (ev.bin < T)]
    ev["logged"] = ev.logged.astype(bool)
    daily = pd.read_csv(W / "daily_summary.csv")

    n_total = len(pids) if not limit else min(limit, len(pids))
    writer, n_rows = None, 0
    for c0 in range(0, n_total, CHUNK):
        sl = slice(c0, min(c0 + CHUNK, n_total))
        p = pids[sl]
        df = build_chunk(p, cgm["glucose_mgdl"][sl], vit["heart_rate_bpm"][sl], vit["steps"][sl],
                         vit["sleep_stage"][sl], ev, daily[daily.patient_id.isin(p)], ehr, every)
        tab = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT / "features.parquet", tab.schema, compression="zstd")
        writer.write_table(tab)
        n_rows += len(df)
        print(f"  patients {c0 + 1}-{sl.stop}: {n_rows:,} rows ({time.time() - t0:.0f}s)")
    writer.close()
    print(f"Done: {n_rows:,} rows x {df.shape[1]} columns -> {OUT / 'features.parquet'} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--every", type=int, default=30, choices=[5, 10, 15, 30, 60], help="minutes between prediction points")
    ap.add_argument("--limit", type=int, default=None, help="only the first N patients (quick test)")
    args = ap.parse_args()
    main(args.every, args.limit)
