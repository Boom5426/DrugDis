"""ECFP4 fingerprints (Morgan radius 2, 2,048 bits) for every compound.

Converted from the analysis notebook SMILES_Encoder.ipynb (cells 4 and 7 to 9),
logic unchanged. The compounds are the unique SMILES of drug_response.parquet;
each is stripped, a charged tin atom [Sn-] is neutralised to [Sn] (the one
valence fix the notebook applies), canonicalised with RDKit, and fingerprinted.
The output is a pickle of {canonical SMILES: float32 vector of length 2,048}.
Compounds whose SMILES RDKit cannot parse get no fingerprint and therefore do
not enter the benchmark dataset.

    python drugdis/data/encode_ecfp4.py \
        --out "$DRUGDIS_DATA/Molecule_Embeddings/ECFP4_emb2048.pickle"

The other eleven compound representations of the benchmark were computed with
the published models listed in the manuscript and are distributed as files in
the Hugging Face dataset (Molecule_Embeddings/).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

RADIUS = 2
N_BITS = 2048


def canonical_unique(smiles):
    """Notebook cell 4: strip, neutralise [Sn-], canonicalise, deduplicate."""
    valid, invalid = {}, []
    for smi in smiles:
        s = smi.strip()
        if "[Sn-]" in s:
            s = s.replace("[Sn-]", "[Sn]")
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            invalid.append(smi)
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can not in valid:
            valid[can] = mol
    return list(valid.keys()), invalid


def ecfp4(smiles, radius: int = RADIUS, n_bits: int = N_BITS):
    """Notebook cell 7: parse again, canonicalise, fingerprint each unique molecule."""
    fpgen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    mols, invalid = {}, set()
    for smi in set(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.add(smi)
            continue
        can = Chem.MolToSmiles(mol, canonical=True)
        if can not in mols:
            mols[can] = mol
    out = {can: np.array(fpgen.GetFingerprintAsNumPy(mol), dtype=np.float32)
           for can, mol in mols.items()}
    return out, sorted(invalid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default=None,
                    help="default: $DRUGDIS_DATA/drug_response.parquet")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    response = a.response or str(paths.data_dir() / "drug_response.parquet")
    smiles = set(pd.read_parquet(response)["SMILES"].tolist())
    canon, bad = canonical_unique(smiles)
    print(f"{len(smiles):,} unique SMILES, {len(canon):,} canonical, {len(bad)} unparsable", flush=True)
    fps, bad2 = ecfp4(canon)
    print(f"{len(fps):,} fingerprints, {len(bad2)} unparsable on the second pass", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(fps, fh)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
