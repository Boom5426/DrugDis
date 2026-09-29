#!/usr/bin/env python3
"""Promote one audited staging table with checksum checks and a full backup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--staged-file", type=Path, required=True)
    parser.add_argument("--audit-file", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--metadata-key", required=True)
    parser.add_argument("--metadata-value", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(value: dict, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", prefix=destination.name + ".",
                                     suffix=".tmp", dir=destination.parent,
                                     delete=False) as handle:
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
    canonical_dir = args.canonical_dir.resolve()
    staged_file = args.staged_file.resolve()
    audit_file = args.audit_file.resolve()
    backup_dir = args.backup_dir.resolve()
    registry_file = canonical_dir / "TABLES.json"
    required = [canonical_dir, staged_file, audit_file, registry_file]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))
    if backup_dir.exists():
        raise SystemExit(f"refusing to reuse backup directory: {backup_dir}")

    with registry_file.open() as handle:
        registry = json.load(handle)
    with audit_file.open() as handle:
        audit = json.load(handle)
    if audit.get("status") != "pass":
        raise AssertionError("audit status is not pass")
    if args.key not in registry:
        raise KeyError(args.key)

    current_file = canonical_dir / registry[args.key]["file"]
    current_hash = sha256_file(current_file)
    staged_hash = sha256_file(staged_file)
    if current_hash != registry[args.key]["sha256"]:
        raise AssertionError("current canonical table does not match TABLES.json")
    audited_new_hash = audit.get("new_t03_sha256")
    if audited_new_hash and staged_hash != audited_new_hash:
        raise AssertionError("staged table does not match its audit")

    backup_dir.mkdir(parents=True)
    shutil.copy2(current_file, backup_dir / current_file.name)
    shutil.copy2(registry_file, backup_dir / registry_file.name)
    shutil.copy2(audit_file, backup_dir / audit_file.name)

    with staged_file.open() as handle:
        rows = max(sum(1 for _ in handle) - 1, 0)
    atomic_copy(staged_file, current_file)
    registry[args.key] = {
        "file": current_file.name,
        "rows": rows,
        "sha256": staged_hash,
        args.metadata_key: args.metadata_value,
        "audit_sha256": sha256_file(audit_file),
    }
    atomic_json(registry, registry_file)
    if sha256_file(current_file) != staged_hash:
        raise AssertionError("post-promotion checksum failed")
    print(json.dumps({
        "status": "pass", "key": args.key,
        "before_sha256": current_hash, "after_sha256": staged_hash,
        "backup_dir": str(backup_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
