"""
Daily IBM calibration logger.

Usage:
    python log_calibration.py                 # snapshot all visible backends now
    python log_calibration.py --backfill 60   # also try to fetch the past 60 days
    python log_calibration.py --flatten       # rebuild tidy CSVs from saved JSON
    python log_calibration.py --compress-old  # gzip any old uncompressed .json snapshots

Raw snapshots go to data/raw/<backend>/<last_update>.json.gz
Tidy tables go to data/tidy/qubits.csv and data/tidy/gates.csv
"""

import argparse
import csv
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from qiskit_ibm_runtime import QiskitRuntimeService

RAW = Path("data/raw")
TIDY = Path("data/tidy")


def save_snapshot(backend_name, props):
    """Save one BackendProperties snapshot; skip if this calibration is already saved."""
    d = props.to_dict()
    stamp = str(d.get("last_update_date", "unknown")).replace(":", "-").replace(" ", "T")
    out = RAW / backend_name / f"{stamp}.json.gz"
    if out.exists() or out.with_suffix("").exists():  # skip if saved (gz or old plain json)
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    d["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    with gzip.open(out, "wt") as fh:
        json.dump(d, fh, default=str)
    return True


def load_snapshot(f):
    if f.suffix == ".gz":
        with gzip.open(f, "rt") as fh:
            return json.load(fh)
    return json.loads(f.read_text())


def snapshot_now(service):
    backends = service.backends()
    print("Backends found:", [b.name for b in backends])
    for b in backends:
        try:
            props = b.properties(refresh=True)
            if props is None:
                print(f"{b.name}: no properties available")
                continue
            new = save_snapshot(b.name, props)
            print(f"{b.name}: {'saved' if new else 'already have this calibration'}")
        except Exception as e:
            print(f"{b.name}: failed ({e})")


def backfill(service, days):
    """Try historical queries. May raise NotImplementedError on the cloud channel."""
    now = datetime.now(timezone.utc)
    for b in service.backends():
        got = 0
        for k in range(1, days + 1):
            try:
                props = b.properties(datetime=now - timedelta(days=k))
            except NotImplementedError:
                print(f"{b.name}: historical queries not supported here; log daily instead")
                return
            except Exception as e:
                print(f"{b.name} day -{k}: failed ({e})")
                continue
            if props is not None and save_snapshot(b.name, props):
                got += 1
        print(f"{b.name}: backfilled {got} new snapshots")


def compress_old():
    """Convert old plain .json snapshots to .json.gz (about 10x smaller)."""
    n = 0
    for f in RAW.glob("*/*.json"):
        with gzip.open(f.with_name(f.name + ".gz"), "wt") as fh:
            fh.write(f.read_text())
        f.unlink()
        n += 1
    print(f"compressed {n} old snapshots")


def flatten():
    """Turn raw JSON into two long-format CSVs: per-qubit and per-gate properties."""
    TIDY.mkdir(parents=True, exist_ok=True)
    qrows, grows = [], []
    for f in sorted(list(RAW.glob("*/*.json")) + list(RAW.glob("*/*.json.gz"))):
        d = load_snapshot(f)
        backend, snap = f.parent.name, d.get("last_update_date")
        for q, params in enumerate(d.get("qubits", [])):
            for p in params:
                qrows.append([backend, snap, q, p.get("name"), p.get("value"), p.get("unit"), p.get("date")])
        for g in d.get("gates", []):
            for p in g.get("parameters", []):
                grows.append([backend, snap, g.get("gate"), "-".join(map(str, g.get("qubits", []))),
                              p.get("name"), p.get("value"), p.get("unit"), p.get("date")])
    with open(TIDY / "qubits.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "snapshot", "qubit", "param", "value", "unit", "measured_at"])
        w.writerows(qrows)
    with open(TIDY / "gates.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "snapshot", "gate", "qubits", "param", "value", "unit", "measured_at"])
        w.writerows(grows)
    print(f"{len(qrows)} qubit rows, {len(grows)} gate rows written to {TIDY}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0)
    ap.add_argument("--flatten", action="store_true")
    ap.add_argument("--compress-old", action="store_true")
    args = ap.parse_args()

    if args.compress_old:
        compress_old()
    elif args.flatten:
        flatten()
    else:
        service = QiskitRuntimeService()  # uses the account saved with save_account
        snapshot_now(service)
        if args.backfill:
            backfill(service, args.backfill)
        flatten()
