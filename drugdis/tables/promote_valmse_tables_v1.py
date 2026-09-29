#!/usr/bin/env python3
"""Promote audited valMSE T10/T13 staging tables into a canonical table set.

The operation is deliberately narrow and recoverable: current files and the
registry are copied into a new backup directory, all recorded hashes are
verified, and only T10, T13 and their TABLES.json entries are replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TARGETS = ("T10", "T13")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(value: dict, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix=destination.name + ".", suffix=".tmp",
        dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=1)
        handle.write("\n")
    try:
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = parse_args()
    canonical = args.canonical_dir.resolve()
    staging = args.staging_dir.resolve()
    backup = args.backup_dir.resolve()
    if backup.exists():
        raise SystemExit(f"refusing to reuse backup directory: {backup}")

    registry_path = canonical / "TABLES.json"
    audit_path = staging / "identity_audit.json"
    required = [canonical, staging, registry_path, audit_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))
    with registry_path.open() as handle:
        registry = json.load(handle)
    with audit_path.open() as handle:
        audit = json.load(handle)
    if audit.get("status") != "pass" or audit.get("t07_t10_profiles_checked") != 20 \
            or audit.get("t07_t13_m0_profiles_checked") != 10:
        raise AssertionError("staging identity audit did not pass the complete grid")

    before = {}
    after = {}
    for key in TARGETS:
        current = canonical / registry[key]["file"]
        staged_record = audit["outputs"][key]
        staged = Path(staged_record["file"])
        if staged.parent != staging:
            raise AssertionError(f"{key} audit points outside staging directory: {staged}")
        if not current.exists() or not staged.exists():
            raise FileNotFoundError((current, staged))
        current_hash = sha256_file(current)
        staged_hash = sha256_file(staged)
        if current_hash != registry[key]["sha256"]:
            raise AssertionError(f"{key} canonical file does not match TABLES.json")
        if staged_hash != staged_record["sha256"]:
            raise AssertionError(f"{key} staged file does not match the identity audit")
        before[key] = {"file": current.name, "rows": registry[key]["rows"],
                       "sha256": current_hash}
        with staged.open() as handle:
            rows = max(sum(1 for _ in handle) - 1, 0)
        after[key] = {"file": current.name, "rows": rows, "sha256": staged_hash}

    backup.mkdir(parents=True)
    shutil.copy2(registry_path, backup / "TABLES.json")
    for key in TARGETS:
        shutil.copy2(canonical / before[key]["file"], backup / before[key]["file"])

    for key in TARGETS:
        staged = Path(audit["outputs"][key]["file"])
        destination = canonical / before[key]["file"]
        atomic_copy(staged, destination)
        registry[key] = {
            **after[key],
            "selection_rule": "frozen_valMSE",
            "identity_audit_sha256": sha256_file(audit_path),
        }
    atomic_json(registry, registry_path)

    promotion = {
        "schema_version": "rise.canonical_table_promotion.v1",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "minimum validation MSE_raw",
        "identity_audit": str(audit_path),
        "identity_audit_sha256": sha256_file(audit_path),
        "backup_dir": str(backup),
        "before": before,
        "after": after,
    }
    atomic_json(promotion, canonical / "RESULT_IDENTITY_valmse_v1.json")

    for key in TARGETS:
        if sha256_file(canonical / after[key]["file"]) != after[key]["sha256"]:
            raise AssertionError(f"post-promotion checksum failed for {key}")
    print(json.dumps(promotion, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
