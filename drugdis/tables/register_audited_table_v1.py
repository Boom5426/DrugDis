#!/usr/bin/env python3
"""Register one new audited canonical table without replacing existing files."""

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
    parser.add_argument("--description", required=True)
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
    if args.key in json.load(registry_file.open()):
        raise SystemExit(f"refusing to replace existing registry key: {args.key}")
    destination = canonical_dir / staged_file.name
    if destination.exists() or backup_dir.exists():
        raise SystemExit("refusing to replace an existing destination or backup")

    with registry_file.open() as handle:
        registry = json.load(handle)
    with audit_file.open() as handle:
        audit = json.load(handle)
    if audit.get("status") != "pass":
        raise AssertionError("audit status is not pass")
    staged_hash = sha256_file(staged_file)
    if audit.get("new_table_sha256") != staged_hash:
        raise AssertionError("staged table does not match audit hash")
    for key, record in registry.items():
        if sha256_file(canonical_dir / record["file"]) != record["sha256"]:
            raise AssertionError(f"existing canonical table {key} fails its registry hash")

    backup_dir.mkdir(parents=True)
    shutil.copy2(registry_file, backup_dir / registry_file.name)
    shutil.copy2(audit_file, backup_dir / audit_file.name)
    atomic_copy(staged_file, destination)
    with staged_file.open() as handle:
        rows = max(sum(1 for _ in handle) - 1, 0)
    registry[args.key] = {
        "file": destination.name, "rows": rows, "sha256": staged_hash,
        "description": args.description, "audit_sha256": sha256_file(audit_file),
    }
    atomic_json(registry, registry_file)
    print(json.dumps({"status": "pass", "key": args.key, "file": destination.name,
                      "rows": rows, "sha256": staged_hash,
                      "backup_dir": str(backup_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
