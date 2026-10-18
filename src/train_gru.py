r"""
Train and evaluate the GRU deep-learning model of the digital twin (PyTorch, GPU).

Same prediction points, labels and patient split as the LightGBM models (rows of
data/model/features.parquet), so the two can be compared and ensembled fairly.

Architecture (fusion of the two data streams):
    raw 6-hour sequence, 72 x 5-min steps, 11 channels      EHR + daily context vector
    (glucose, HR, steps, sleep, logged carbs/doses,          (HbA1c, meds, kidney, BMI, genetics,
     exercise, missing masks, time of day)                    last night's sleep, personal profile...)
                     |                                                    |
               2-layer GRU (128)                                   MLP (64)
                     \______________________  ______________________/
                                            concat
                                          MLP (128)
               heads: spike / severe / hypo risk (sigmoid) + glucose change at 30/60/90/120 min

Multi-task loss: masked binary cross-entropy for the three alerts + Huber loss for the forecasts.

Outputs:
  models/gru/gru.pt                     weights + normalisation stats + config
  results/gru/metrics.json, report.html, test_predictions.parquet

Usage:
    python src/train_gru.py            # full training (GPU recommended)
    python src/train_gru.py --quick    # smoke test: 10% of train patients, 1 epoch

Experiment options (results go to results/gru_<tag>, models/gru_<tag>):
    --seq-len 144        12 h of history instead of 6 h
    --t2d-weight 2       count T2D patients' rows twice as much in the loss
    --hypo-weight 3      count hypoglycaemia positives 3x (rare event)
    --oracle-meals       DIAGNOSTIC ONLY: give the model every meal with its true carbs
                         (not available in real life) to measure how much unlogged meals cost
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (EHR_CATEG, EHR_FLAGS, EHR_NUMERIC, START, STEP, T, W,  # noqa: E402
                            load_stream)
from train_lightgbm import (binary_report, calibration, clarke_zones, event_detection,  # noqa: E402
                            threshold_for_sensitivity)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "model" / "features.parquet"
MODELS = ROOT / "models" / "gru"
RESULTS = ROOT / "results" / "gru"

SEQ_LEN = 72
CONTEXT = ["prescribed_doses_per_day", "prof_glucose_tod", "prof_future_max_tod", "prof_spike_rate_tod",
           "sleep_hours_last_night", "hrv_last_night", "sleep_eff_last_night", "resting_hr_last_night",
           "steps_yesterday", "g_mean_24h", "g_cv_24h", "min_since_meal", "min_since_dose_logged",
           "doses_logged_today", "carbs_today", "weekend", "hour_sin", "hour_cos"]
STATIC = EHR_NUMERIC + EHR_FLAGS + list(EHR_CATEG) + CONTEXT
CLS = ["y_spike", "y_severe", "y_hypo"]
REG = ["g_p30", "g_p60", "g_p90", "g_p120"]
REG_SCALE = 50.0  # forecast targets are glucose change / 50
CFG = dict(hidden=128, layers=2, static_hidden=64, dropout=0.2, batch=2048, lr=2e-3, weight_decay=1e-4,
           epochs=12, patience=3, reg_weight=1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def img(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def build_sequences(device, oracle_meals=False):
    """(patients, T, 11) float16 tensor on the device + patient order."""
    pids, cgm = load_stream(W / "cgm.parquet", ["glucose_mgdl"])
    _, vit = load_stream(W / "vitals.parquet", ["heart_rate_bpm", "steps", "sleep_stage"])
    n = len(pids)
    row = {p: i for i, p in enumerate(pids)}
    g, hr, st, asleep = cgm["glucose_mgdl"], vit["heart_rate_bpm"], vit["steps"], vit["sleep_stage"]

    ev = pd.read_csv(W / "events.csv", parse_dates=["timestamp"],
                     usecols=["patient_id", "timestamp", "event_type", "logged", "carbs_g", "logged_carbs_g", "duration_min"])
    ev["logged"] = ev.logged.astype(bool)
    ev["bin"] = ((ev.timestamp - START) / pd.Timedelta(minutes=STEP)).astype(int)
    ev = ev[(ev.bin >= 0) & (ev.bin < T)]
    ev["r"] = ev.patient_id.map(row)
    carbs = np.zeros((n, T), np.float32)
    if oracle_meals:   # every meal, true carbs (hidden simulator truth)
        meals = ev[ev.event_type == "meal"]
        np.add.at(carbs, (meals.r.to_numpy(), meals.bin.to_numpy()), meals.carbs_g.fillna(0).to_numpy())
    else:              # only what the patient logged, with their estimate
        meals = ev[(ev.event_type == "meal") & ev.logged]
        np.add.at(carbs, (meals.r.to_numpy(), meals.bin.to_numpy()), meals.logged_carbs_g.fillna(0).to_numpy())
    ev = ev[ev.logged]
    dose = np.zeros((n, T), np.float32)
    meds = ev[ev.event_type == "medication"]
    np.add.at(dose, (meds.r.to_numpy(), meds.bin.to_numpy()), 1.0)
    exm = np.zeros((n, T), np.float32)
    for r, b, d in ev[ev.event_type == "exercise"][["r", "bin", "duration_min"]].itertuples(index=False):
        exm[r, b:b + int(d // STEP)] = 1.0

    hour = (np.arange(T) % 288) * STEP / 60
    chans = [
        np.nan_to_num((g - 140) / 50), np.isnan(g).astype(np.float32),
        np.nan_to_num((hr - 75) / 12), np.isnan(hr).astype(np.float32),
        np.log1p(np.nan_to_num(st)) / 6, asleep.astype(np.float32),
        carbs / 100, dose, exm,
        np.broadcast_to(np.sin(2 * np.pi * hour / 24), (n, T)), np.broadcast_to(np.cos(2 * np.pi * hour / 24), (n, T)),
    ]
    seq = np.stack(chans, axis=2).astype(np.float16)
    del chans, g, hr, st, carbs, dose, exm
    return pids, torch.from_numpy(seq).to(device)


def load_rows(pids, quick):
    cols = ["patient_id", "timestamp", "split", "g_now", "g_delta_30", "cohort", "setting", "on_sulfonylurea",
            "on_basal_insulin"] + CLS + REG + [c for c in STATIC if c not in ("cohort", "setting", "on_sulfonylurea", "on_basal_insulin")]
    df = pd.read_parquet(FEATURES, columns=list(dict.fromkeys(cols)))
    if quick:
        keep = df.loc[df.split == "train", "patient_id"].drop_duplicates().sample(frac=0.1, random_state=0)
        df = df[(df.split != "train") | df.patient_id.isin(keep)]
    row = {p: i for i, p in enumerate(pids)}
    df["p_idx"] = df.patient_id.map(row).astype(np.int64)
    df["t_idx"] = ((df.timestamp - START) / pd.Timedelta(minutes=STEP)).astype(np.int64)
    return df.reset_index(drop=True)


def to_tensors(df, mean, std, device):
    x = ((df[STATIC].to_numpy(np.float32) - mean) / std)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y_cls = df[CLS].to_numpy(np.float32)
    y_reg = ((df[REG].to_numpy(np.float32) - df[["g_now"]].to_numpy(np.float32)) / REG_SCALE)
    return {"p": torch.from_numpy(df.p_idx.to_numpy()).to(device), "t": torch.from_numpy(df.t_idx.to_numpy()).to(device),
            "x": torch.from_numpy(x).to(device, torch.float16),
            "yc": torch.from_numpy(y_cls).to(device), "yr": torch.from_numpy(y_reg).to(device),
            "t2d": torch.from_numpy((df.cohort.to_numpy() == 2).astype(np.float32)).to(device)}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class TwinGRU(nn.Module):
    def __init__(self, n_seq, n_static, cfg):
        super().__init__()
        self.gru = nn.GRU(n_seq, cfg["hidden"], num_layers=cfg["layers"], batch_first=True, dropout=cfg["dropout"])
        self.static = nn.Sequential(nn.Linear(n_static, cfg["static_hidden"]), nn.GELU(), nn.Dropout(cfg["dropout"]),
                                    nn.Linear(cfg["static_hidden"], cfg["static_hidden"]), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(cfg["hidden"] + cfg["static_hidden"], 128), nn.GELU(), nn.Dropout(cfg["dropout"]))
        self.cls_head = nn.Linear(128, len(CLS))
        self.reg_head = nn.Linear(128, len(REG))

    def forward(self, seq, static):
        if not torch.is_autocast_enabled():
            seq, static = seq.float(), static.float()
        _, h = self.gru(seq)
        z = self.trunk(torch.cat([h[-1], self.static(static)], dim=1))
        return self.cls_head(z), self.reg_head(z)


def batch_seq(seq_all, p, t):
    offsets = torch.arange(-SEQ_LEN + 1, 1, device=t.device)
    return seq_all[p[:, None], (t[:, None] + offsets).clamp(min=0)]   # (B, SEQ_LEN, C)


def loss_fn(logits, reg, yc, yr, t2d, cfg):
    """Masked multi-task loss with optional T2D-row and hypo-positive weighting."""
    w_row = 1 + (cfg["t2d_weight"] - 1) * t2d                        # (B,)
    m = ~torch.isnan(yc)
    y = torch.nan_to_num(yc)
    w = w_row[:, None].expand_as(y).clone()
    w[:, 2] = w[:, 2] * torch.where(y[:, 2] > 0.5, cfg["hypo_weight"], 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * w * m).sum() / (w * m).sum().clamp(min=1)
    mr = ~torch.isnan(yr)
    hub = F.huber_loss(reg, torch.nan_to_num(yr), delta=0.6, reduction="none")
    wr = w_row[:, None].expand_as(hub)
    hub = (hub * wr * mr).sum() / (wr * mr).sum().clamp(min=1)
    return bce + cfg["reg_weight"] * hub


@torch.no_grad()
def predict(model, seq_all, data, batch=None):
    model.eval()
    batch = batch or max(1024, 8192 * 72 // SEQ_LEN)   # keep GPU memory flat for long histories
    outs_c, outs_r = [], []
    for i in range(0, len(data["p"]), batch):
        sl = slice(i, i + batch)
        with torch.autocast("cuda", dtype=torch.float16, enabled=seq_all.is_cuda):
            c, r = model(batch_seq(seq_all, data["p"][sl], data["t"][sl]), data["x"][sl])
        outs_c.append(torch.sigmoid(c.float()).cpu())
        outs_r.append(r.float().cpu())
    return torch.cat(outs_c).numpy(), torch.cat(outs_r).numpy() * REG_SCALE


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(args):
    global SEQ_LEN, MODELS, RESULTS
    quick = args.quick
    SEQ_LEN = args.seq_len
    if args.tag:
        MODELS, RESULTS = MODELS.parent / f"gru_{args.tag}", RESULTS.parent / f"gru_{args.tag}"
    t_start = time.time()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else " - training will be slow"))
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    pids, seq_all = build_sequences(device, args.oracle_meals)
    log(f"Sequences on {device}: {tuple(seq_all.shape)} ({seq_all.element_size() * seq_all.nelement() / 1e9:.2f} GB)")
    df = load_rows(pids, quick)
    tr_df, va_df, te_df = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    del df
    mean = np.nanmean(tr_df[STATIC].to_numpy(np.float32), 0)
    std = np.nanstd(tr_df[STATIC].to_numpy(np.float32), 0) + 1e-6
    mean, std = np.nan_to_num(mean), np.where(np.isnan(std), 1.0, std)
    tr, va, te = (to_tensors(d, mean, std, device) for d in (tr_df, va_df, te_df))
    log(f"Rows: train {len(tr_df):,} / val {len(va_df):,} / test {len(te_df):,}")

    cfg = dict(CFG, seq_len=SEQ_LEN, t2d_weight=args.t2d_weight, hypo_weight=args.hypo_weight,
               oracle_meals=args.oracle_meals, tag=args.tag)
    if quick:
        cfg.update(epochs=1, patience=1)
    model = TwinGRU(seq_all.shape[2], len(STATIC), cfg).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = int(np.ceil(len(tr_df) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps_per_epoch * cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_state, bad, history = -1.0, None, 0, []
    m_va = ~np.isnan(va_df.y_spike.to_numpy())
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(tr_df), device=device)
        total = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg["batch"]:(b + 1) * cfg["batch"]]
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                c, r = model(batch_seq(seq_all, tr["p"][idx], tr["t"][idx]), tr["x"][idx])
            loss = loss_fn(c.float(), r.float(), tr["yc"][idx], tr["yr"][idx], tr["t2d"][idx], cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
            if b % 200 == 0 and b:
                log(f"  epoch {epoch} batch {b}/{steps_per_epoch} loss {total / (b + 1):.4f}")
        pc, pr = predict(model, seq_all, va)
        auc = roc_auc_score(va_df.y_spike.to_numpy()[m_va], pc[m_va, 0])
        mae60 = np.nanmean(np.abs(va_df.g_now.to_numpy() + pr[:, 1] - va_df.g_p60.to_numpy()))
        history.append({"epoch": epoch, "train_loss": total / steps_per_epoch, "val_spike_auroc": auc, "val_mae60": mae60,
                        "seconds": time.time() - t0})
        log(f"Epoch {epoch}: train loss {total / steps_per_epoch:.4f} | val spike AUROC {auc:.4f} | val MAE@60 {mae60:.1f} "
            f"| {time.time() - t0:.0f}s")
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log("Early stopping")
                break
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg, "static_features": STATIC, "static_mean": mean.tolist(),
                "static_std": std.tolist(), "seq_len": SEQ_LEN, "reg_scale": REG_SCALE}, MODELS / "gru.pt")

    # ---------------- evaluation (same metrics as LightGBM)
    log("Evaluating on test patients")
    pc_va, _ = predict(model, seq_all, va)
    pc_te, pr_te = predict(model, seq_all, te)
    preds = te_df[["patient_id", "timestamp", "g_now", "cohort", "setting", "on_sulfonylurea", "on_basal_insulin"] + CLS + REG].copy()
    metrics = {"run": {"machine": platform.node(), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                       "torch": torch.__version__, "python": platform.python_version(), "quick": quick,
                       "rows_train": len(tr_df), "config": cfg, "history": history,
                       "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                                    capture_output=True, text=True).stdout.strip() or None}}
    curves = {}
    for k, (name, target) in enumerate(zip(["spike", "severe", "hypo"], CLS)):
        y_va, y_te = va_df[target].to_numpy(), te_df[target].to_numpy()
        mv, mt = ~np.isnan(y_va), ~np.isnan(y_te)
        thr = threshold_for_sensitivity(y_va[mv], pc_va[mv, k], 0.80)
        res = binary_report(y_te[mt], pc_te[mt, k], thr, te_df.patient_id.to_numpy()[mt], te_df.timestamp[mt])
        preds[f"p_{name}"] = np.where(mt, pc_te[:, k], np.nan)
        if name in ("spike", "hypo"):
            scored = preds.assign(**{f"p_{name}": preds[f"p_{name}"].fillna(0)})
            res["event_detection"] = event_detection(scored, f"p_{name}", thr, name)
        sub = {}
        for code, lab in {0: "Normoglycemic", 1: "Prediabetes", 2: "T2D"}.items():
            mask = mt & (te_df.cohort.to_numpy() == code)
            if mask.sum() > 100 and 0 < y_te[mask].sum() < mask.sum():
                sub[f"cohort={lab}"] = {"auroc": float(roc_auc_score(y_te[mask], pc_te[mask, k])), "n": int(mask.sum())}
        res["subgroups"] = sub
        res["calibration"] = calibration(y_te[mt], pc_te[mt, k]).round(4).reset_index().to_dict("records")
        metrics[name] = res
        curves[name] = (y_te[mt], pc_te[mt, k])
        log(f"  {name}: AUROC {res['auroc']:.3f}, AUPRC {res['auprc']:.3f}, sens {res['sensitivity']:.2f}, "
            f"spec {res['specificity']:.2f}, PPV {res['ppv']:.2f}, alerts/patient-day {res['alerts_per_patient_day']:.2f}")
    for k, (name, target) in enumerate(zip(["fc30", "fc60", "fc90", "fc120"], REG)):
        y = te_df[target].to_numpy()
        m = ~np.isnan(y)
        p = te_df.g_now.to_numpy() + pr_te[:, k]
        zones = pd.Series(clarke_zones(y[m], p[m])).value_counts(normalize=True).reindex(list("ABCDE"), fill_value=0)
        metrics[name] = {"mae": float(np.mean(np.abs(p[m] - y[m]))), "rmse": float(np.sqrt(np.mean((p[m] - y[m]) ** 2))),
                         "mard_pct": float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100),
                         "baseline_persistence_mae": float(np.mean(np.abs(te_df.g_now.to_numpy()[m] - y[m]))),
                         "clarke_zones_pct": (zones * 100).round(2).to_dict(), "n": int(m.sum())}
        preds[f"pred_{name}"] = p
        log(f"  {name}: MAE {metrics[name]['mae']:.1f} (persistence {metrics[name]['baseline_persistence_mae']:.1f}), "
            f"Clarke A+B {zones['A'] * 100 + zones['B'] * 100:.1f}%")

    preds.to_parquet(RESULTS / "test_predictions.parquet", index=False)
    metrics["run"]["total_minutes"] = round((time.time() - t_start) / 60, 1)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(metrics, curves)
    log(f"Done in {metrics['run']['total_minutes']} min. Results -> {RESULTS}")


def write_report(m, curves):
    hist = pd.DataFrame(m["run"]["history"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(hist.epoch, hist.train_loss, marker="o"); ax[0].set(title="Training loss", xlabel="epoch")
    ax[1].plot(hist.epoch, hist.val_spike_auroc, marker="o", color="#c2410c"); ax[1].set(title="Validation spike AUROC", xlabel="epoch")
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[2].plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y, p):.3f})")
    ax[2].plot([0, 1], [0, 1], "k:", lw=0.6); ax[2].set(title="ROC on test patients", xlabel="FPR", ylabel="TPR"); ax[2].legend()
    chart = img(fig)
    cls_rows = "".join(
        f"<tr><td>{n}</td><td>{m[n]['auroc']:.3f}</td><td>{m[n]['auprc']:.3f}</td><td>{m[n]['sensitivity']:.0%}</td>"
        f"<td>{m[n]['specificity']:.0%}</td><td>{m[n]['ppv']:.0%}</td><td>{m[n]['alerts_per_patient_day']:.2f}</td></tr>"
        for n in ("spike", "severe", "hypo"))
    fc_rows = "".join(
        f"<tr><td>{n[2:]} min</td><td>{m[n]['mae']:.1f}</td><td>{m[n]['rmse']:.1f}</td><td>{m[n]['baseline_persistence_mae']:.1f}</td>"
        f"<td>{m[n]['clarke_zones_pct']['A'] + m[n]['clarke_zones_pct']['B']:.1f}%</td></tr>"
        for n in ("fc30", "fc60", "fc90", "fc120"))
    r = m["run"]
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>GRU Results</title>
<style>body{{font-family:system-ui,Segoe UI,sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;color:#1e293b}}
table{{border-collapse:collapse;width:100%;margin:12px 0}} td,th{{border-bottom:1px solid #e2e8f0;padding:6px 8px;text-align:left}}
.muted{{color:#64748b}} img{{max-width:100%}}</style></head><body>
<h1>GRU Digital Twin &mdash; Test Results</h1>
<p class="muted">{r['device']} &middot; torch {r['torch']} &middot; {r['rows_train']:,} training rows &middot; {len(r['history'])} epochs
&middot; {r['total_minutes']} min{' &middot; <b>QUICK MODE</b>' if r['quick'] else ''}. All numbers are on held-out test patients.</p>
<h2>Adverse-event alerts (next 2 h, threshold for 80% sensitivity on validation)</h2>
<table><tr><th>Event</th><th>AUROC</th><th>AUPRC</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>Alerts / patient-day</th></tr>{cls_rows}</table>
<h2>Glucose forecast</h2>
<table><tr><th>Horizon</th><th>MAE</th><th>RMSE</th><th>Persistence MAE</th><th>Clarke A+B</th></tr>{fc_rows}</table>
<h2>Training</h2>{chart}</body></html>"""
    (RESULTS / "report.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seq-len", type=int, default=72, help="history length in 5-min steps (72 = 6 h)")
    ap.add_argument("--t2d-weight", type=float, default=1.0)
    ap.add_argument("--hypo-weight", type=float, default=1.0)
    ap.add_argument("--oracle-meals", action="store_true", help="diagnostic: true carbs for every meal")
    ap.add_argument("--tag", default="", help="experiment name for output folders")
    main(ap.parse_args())
