"""
Convert the public ShanghaiT2DM dataset into tidy CSVs used to validate the simulator.

Source: Zhao Q. et al., "Chinese diabetes datasets for data-driven machine learning",
Scientific Data 10, 35 (2023). figshare doi:10.6084/m9.figshare.20444397 (CC BY 4.0).
100 T2D patients (109 recordings), CGM every 15 min for 3-14 days, with meal records.

Usage:
    python src/process_shanghai.py   # reads data/external/shanghai/raw, writes data/external/shanghai/*.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "external" / "shanghai"
SU_NAMES = ("gliclazide", "glimepiride", "glipizide", "gliquidone", "glibenclamide", "gliben")
INSULIN_NAMES = ("insulin", "novolin", "humulin", "gansulin", "lantus", "levemir", "novorapid", "humalog",
                 "glargine", "detemir", "aspart", "lispro", "degludec", "tresiba", "toujeo", "basalin", "novomix")


def cgm_metrics(g: pd.Series) -> dict:
    g = g.dropna()
    return {"cgm_mean_glucose": g.mean(), "gmi_pct": 3.31 + 0.02392 * g.mean(), "cv_pct": g.std() / g.mean() * 100,
            "tir_70_180_pct": g.between(70, 180).mean() * 100, "tar_180_pct": (g > 180).mean() * 100,
            "tar_250_pct": (g > 250).mean() * 100, "tbr_70_pct": (g < 70).mean() * 100,
            "tbr_54_pct": (g < 54).mean() * 100, "n_readings": len(g)}


def main():
    folder = next((SRC / "raw").glob("*")) / "Shanghai_T2DM"
    summary = pd.read_excel(folder.parent / "Shanghai_T2DM_Summary.xlsx")
    summary.columns = [c.strip() for c in summary.columns]

    cgm_rows, meal_rows, metric_rows = [], [], []
    for f in sorted(folder.iterdir()):
        if f.name.startswith("~$") or f.suffix not in (".xls", ".xlsx"):
            continue
        rec = f.stem
        df = pd.read_excel(f)
        cgm_col = next(c for c in df.columns if str(c).strip().upper().startswith("CGM"))
        diet_col = next((c for c in df.columns if str(c).strip().lower().startswith("dietary intake")), None)
        df["timestamp"] = pd.to_datetime(df["Date"], errors="coerce")
        df["glucose_mgdl"] = pd.to_numeric(df[cgm_col], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        cgm_rows.append(df[["timestamp", "glucose_mgdl"]].assign(record_id=rec))
        if diet_col is not None:
            meals = df.loc[df[diet_col].notna() & (df[diet_col].astype(str).str.strip() != ""), ["timestamp"]]
            meal_rows.append(meals.assign(record_id=rec))
        metric_rows.append({"record_id": rec, **cgm_metrics(df["glucose_mgdl"])})

    cgm = pd.concat(cgm_rows)[["record_id", "timestamp", "glucose_mgdl"]]
    meals = pd.concat(meal_rows)[["record_id", "timestamp"]]
    metrics = pd.DataFrame(metric_rows)

    s = summary.rename(columns={"Patient Number": "record_id"})
    agents = s["Hypoglycemic Agents"].astype(str).str.lower()
    s["hba1c_pct"] = pd.to_numeric(s["HbA1c (mmol/mol)"], errors="coerce") / 10.929 + 2.15  # IFCC -> NGSP %
    s["on_sulfonylurea"] = agents.apply(lambda a: any(k in a for k in SU_NAMES))
    s["on_insulin"] = agents.apply(lambda a: any(k in a for k in INSULIN_NAMES))
    s = s[["record_id", "Age (years)", "BMI (kg/m2)", "Duration of diabetes (years)", "hba1c_pct",
           "on_sulfonylurea", "on_insulin"]].rename(columns={"Age (years)": "age", "BMI (kg/m2)": "bmi",
                                                            "Duration of diabetes (years)": "diabetes_duration_yrs"})
    metrics = metrics.merge(s, on="record_id", how="left").round(2)

    cgm.to_csv(SRC / "t2dm_cgm.csv", index=False)
    meals.to_csv(SRC / "t2dm_meals.csv", index=False)
    metrics.to_csv(SRC / "t2dm_summary.csv", index=False)
    print(f"{len(metrics)} recordings, {len(cgm):,} CGM readings, {len(meals):,} meal records")
    print(metrics[["hba1c_pct", "cgm_mean_glucose", "gmi_pct", "cv_pct", "tir_70_180_pct", "tar_250_pct",
                   "tbr_70_pct"]].describe().round(2).loc[["mean", "50%", "min", "max"]])
    print(f"on insulin: {metrics.on_insulin.mean():.0%}, on sulfonylurea: {metrics.on_sulfonylurea.mean():.0%}")


if __name__ == "__main__":
    main()
