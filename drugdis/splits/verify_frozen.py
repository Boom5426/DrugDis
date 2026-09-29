"""Step 2.1: verify the frozen substrate and manifests by checksum only.

No recomputation. Every hash recorded at freeze time is recomputed from disk and
compared. A mismatch is a reproducibility blocker and stops the rerun.
"""
from __future__ import annotations
import hashlib, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

SUB = str(paths.CODE / "splits") + "/"
MAN = str(paths.MANIFESTS) + "/"


def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_list(items):
    h = hashlib.sha256()
    for x in items:
        h.update(str(x).encode()); h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    rec = json.load(open(MAN + "MANIFEST.json"))
    ok, bad = [], []
    print("=== inputs recorded at freeze time ===")
    for k, want in rec["input_hashes"].items():
        got = sha256_file(paths.resolve_config(rec["config"])[k])
        (ok if got == want else bad).append((k, want, got))
        print(f"  {'OK ' if got == want else 'BAD'} {k:22s} {want[:16]}  {os.path.basename(rec['config'][k])}")
    print(f"\n  {'OK ' if sha256_file(SUB + 'make_manifests.py') == rec['generator_sha256'] else 'BAD'} "
          f"generator make_manifests.py {rec['generator_sha256'][:16]}")
    if sha256_file(SUB + "make_manifests.py") != rec["generator_sha256"]:
        bad.append(("generator", rec["generator_sha256"], sha256_file(SUB + "make_manifests.py")))

    print("\n=== manifests ===")
    for key, want in rec["manifest_sha256"].items():
        m = json.load(open(MAN + key + ".json"))
        for split, w in want.items():
            got = sha256_list(m[split])
            (ok if got == w else bad).append((f"{key}/{split}", w, got))
        n = {s: len(m[s]) for s in ("train", "val", "test")}
        good = all(sha256_list(m[s]) == want[s] for s in want)
        print(f"  {'OK ' if good else 'BAD'} {key:16s} entities {n}")

    frozen_dir = paths.data_dir() / "omics_baseline_frozen"
    prov = json.load(open(frozen_dir / "ccle_anchored_provenance.json"))
    got = sha256_file(frozen_dir / os.path.basename(prov["output"]))
    (ok if got == prov["output_sha256"] else bad).append(("ccle_anchored", prov["output_sha256"], got))
    print(f"\n=== frozen transcriptome ===")
    print(f"  {'OK ' if got == prov['output_sha256'] else 'BAD'} "
          f"baseline_mrna_ccle_anchored.parquet {prov['output_sha256'][:16]} "
          f"({prov['n_samples']} samples x {prov['n_genes']} genes)")
    print(f"\n{len(ok)} checks passed, {len(bad)} failed")
    json.dump({"passed": len(ok), "failed": len(bad),
               "failures": [{"what": w, "expected": e, "got": g} for w, e, g in bad]},
              open(paths.work_dir() / "substrate_verification.json", "w"), indent=1)
    if bad:
        print("BLOCKER: checksum mismatch, the rerun must not proceed")
        return 2
    print("frozen substrate and manifests verified by checksum. No recomputation performed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
