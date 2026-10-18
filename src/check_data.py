"""
Check that the synthetic data on this machine matches the reference copy exactly.

The datasets are committed to git and can also be regenerated from fixed seeds.
This script compares a fingerprint (row counts + checksums) against data/fingerprint.json,
so results from different laptops are comparable.

Usage:
    python src/check_data.py            # verify
    python src/check_data.py --write    # (maintainers only) record a new reference fingerprint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "fingerprint.json"


def digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def fingerprint() -> dict:
    fp = {}
    for name in ["patients", "labs", "medications", "diagnoses", "twin_params"]:
        df = pd.read_csv(ROOT / "data" / "ehr" / f"{name}.csv")
        fp[f"ehr/{name}"] = {"rows": len(df), "sha": digest(df.round(3))}
    for name in ["events", "daily_summary"]:
        df = pd.read_csv(ROOT / "data" / "wearable" / f"{name}.csv")
        fp[f"wearable/{name}"] = {"rows": len(df), "sha": digest(df.round(3))}
    for name, col in [("cgm", "glucose_mgdl"), ("vitals", "heart_rate_bpm")]:
        s = pq.read_table(ROOT / "data" / "wearable" / f"{name}.parquet", columns=[col]).column(col).to_pandas()
        fp[f"wearable/{name}"] = {"rows": len(s), "sha": digest(s.fillna(-1).astype("int32").to_frame())}
    return fp


def main(write: bool):
    fp = fingerprint()
    if write:
        REF.write_text(json.dumps(fp, indent=2))
        print(f"Reference fingerprint written to {REF}")
        return
    ref = json.loads(REF.read_text())
    bad = [k for k in ref if fp.get(k) != ref[k]]
    for k in ref:
        print(f"  {'OK      ' if k not in bad else 'MISMATCH'} {k}: {fp.get(k)}")
    if bad:
        print("\nData differs from the reference. Check you used the same seeds and package versions "
              "(see requirements.txt). Training still works, but results will not match the reference exactly.")
        sys.exit(1)
    print("\nAll data matches the reference. Safe to train.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    main(ap.parse_args().write)
