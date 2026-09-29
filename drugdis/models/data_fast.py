"""Phase 2A-2 data pipeline.

Identical panel construction, filtering and split to
`rise/phase2a1/train_m1.py` (which itself follows `legacy/src_copy`), with one
implementation change: instead of assembling every batch row by row in Python,
the drug and sample feature matrices are materialised once on the GPU and a
batch is gathered by index.

This is a pure execution change. The DataLoader is constructed the same way over
a dataset of the same length, so the sampler consumes the global RNG identically
and yields the same permutation; the gathered rows carry the same float32 values
the per-sample path produced. `validate_equivalence.py` checks this against the
frozen Phase 2A-1 M0 run rather than assuming it.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

CLIO_DIR = str(paths.data_dir()) + "/"
MASTER_TABLE_PATH = os.path.join(CLIO_DIR, "master_table.parquet")
DRUG_RESPONSE_PATH = os.path.join(CLIO_DIR, "drug_response.parquet")
MOL_DIR = CLIO_DIR + "Molecule_Embeddings/"
GENE_DIR = CLIO_DIR + "Gene_Embeddings/"
DRUG_FM_PATH_MAP = {"ECFP4": MOL_DIR + "ECFP4_emb2048.pickle",
                    "KPGT": MOL_DIR + "KPGT_emb2304.pickle"}
GENOMICS_FM_PATH_MAP = {
    "baseline": CLIO_DIR + "omics_baseline/baseline_mrna_common_genes.parquet",
    "SCimilarity": GENE_DIR + "SCimilarity_embeddings.parquet"}

DRUG = "SMILES"
SAMPLE = "Sample_ID"


class IndexDataset(Dataset):
    """Returns row indices. Same length as the panel, so the sampler behaves
    exactly as it did over the per-sample dataset."""

    def __init__(self, n: int):
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> int:
        return i


class Panel:
    """One split of the response table with its GPU-resident index arrays."""

    def __init__(self, df: pd.DataFrame, drug_row: np.ndarray, sample_row: np.ndarray,
                 dev: torch.device):
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.drug_idx = torch.as_tensor(drug_row, dtype=torch.long, device=dev)
        self.sample_idx = torch.as_tensor(sample_row, dtype=torch.long, device=dev)
        self.y = torch.as_tensor(self.df["Sensitivity"].to_numpy(np.float32), device=dev)
        self.targets = self.y.clone()          # what the loss is trained against
        self.dev = dev

    def loader(self, batch_size: int, shuffle: bool, drop_last: bool,
               generator: torch.Generator | None = None) -> DataLoader:
        """With an explicit `generator` the shuffle stream depends only on that
        generator's seed, not on how many draws the arm's architecture consumed
        at initialisation. Every arm at a given seed then sees the identical
        sequence of minibatch permutations, which is what makes the paired
        comparison test the named causal factor and nothing else."""
        return DataLoader(IndexDataset(self.n), batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=drop_last, generator=generator)


class FeatureStore:
    def __init__(self, drug_mat: torch.Tensor, gene_mat: torch.Tensor):
        self.D = drug_mat
        self.G = gene_mat
        self.drug_dim = int(drug_mat.shape[1])
        self.gene_dim = int(gene_mat.shape[1])

    def gather(self, panel: Panel, idx: torch.Tensor):
        idx = idx.to(panel.drug_idx.device, non_blocking=True)
        return self.D[panel.drug_idx[idx]], self.G[panel.sample_idx[idx]], panel.targets[idx]


def build_frozen(substrate_config: str, manifest_dir: str, split_type: str, seed: int,
                 drug_fm: str, dev: torch.device, verbose: bool = True,
                 drug_rep_path: str | None = None, gene_rep_path: str | None = None,
                 allow_partial_drug_rep: bool = False):
    """Frozen-substrate loader: the panel and the split both come from disk.

    The panel is built by the same code that produced the manifests
    (`rise/substrate/make_manifests.py`), so there is one definition of
    eligibility, and the train/val/test entity lists are read rather than
    recomputed. Nothing here depends on dataframe row order."""
    import json
    sys.path.insert(0, str(paths.CODE / "splits"))
    import make_manifests as mm

    cfg = paths.load_config(substrate_config)
    dg = mm.load_transcriptome(cfg)
    panel = mm.build_panel(cfg, dg)
    with open(os.path.join(manifest_dir, f"{split_type}_seed{seed}.json")) as fh:
        man = json.load(fh)
    col = SAMPLE if split_type == "LCLO" else DRUG
    parts = {k: panel[panel[col].astype(str).isin(set(man[k]))].reset_index(drop=True)
             for k in ("train", "val", "test")}
    assert sum(len(v) for v in parts.values()) == len(panel), "manifest does not partition the panel"
    if verbose:
        print(f"  frozen substrate {cfg['substrate_version']} | manifest {split_type} seed {seed}")
        print(f"  pairs: train={len(parts['train']):,} val={len(parts['val']):,} "
              f"test={len(parts['test']):,}", flush=True)

    dpath = drug_rep_path or DRUG_FM_PATH_MAP[drug_fm]
    with open(dpath, "rb") as f:
        drug_feat = pickle.load(f)
    used_drugs = pd.Index(pd.unique(panel[DRUG]))
    used_samples = pd.Index(pd.unique(panel[SAMPLE]))
    missing = [k for k in used_drugs if k not in drug_feat]
    if missing and not allow_partial_drug_rep:
        raise AssertionError(f"{len(missing)} panel drugs absent from "
                             f"{os.path.basename(dpath)}; pass allow_partial_drug_rep to "
                             f"evaluate on the covered subset instead")
    if missing:
        # The frozen manifests are not modified. The uncovered drugs are dropped
        # from every split, and the drop is recorded so the row can be flagged as
        # evaluated on fewer rows and therefore not strictly rank-comparable.
        keep = set(drug_feat.keys())
        before = {k: len(v) for k, v in parts.items()}
        parts = {k: v[v[DRUG].isin(keep)].reset_index(drop=True) for k, v in parts.items()}
        panel = panel[panel[DRUG].isin(keep)].reset_index(drop=True)
        used_drugs = pd.Index(pd.unique(panel[DRUG]))
        used_samples = pd.Index(pd.unique(panel[SAMPLE]))
        if verbose:
            print(f"  PARTIAL COVERAGE: {len(missing):,} of {len(missing)+len(used_drugs):,} "
                  f"panel drugs absent; rows "
                  f"{ {k: f'{before[k]:,}->{len(parts[k]):,}' for k in parts} }", flush=True)
    dmat = np.stack([np.asarray(drug_feat[k], dtype=np.float32) for k in used_drugs])
    del drug_feat
    if gene_rep_path:
        # The panel and its eligibility still come from the frozen CCLE substrate;
        # only the feature matrix is replaced, and the aligned embeddings cover
        # exactly the frozen samples.
        gsrc = pd.read_parquet(gene_rep_path, engine="fastparquet")
        gsrc.index = gsrc.index.astype(str)
    else:
        gsrc = dg
    gmat = gsrc.loc[used_samples].to_numpy(np.float32)
    assert gmat.shape[0] == len(used_samples), (gmat.shape, len(used_samples))
    gmat = np.nan_to_num(gmat, nan=0.0)
    drug_pos = pd.Series(np.arange(len(used_drugs), dtype=np.int64), index=used_drugs)
    sample_pos = pd.Series(np.arange(len(used_samples), dtype=np.int64), index=used_samples)
    store = FeatureStore(torch.as_tensor(dmat, device=dev), torch.as_tensor(gmat, device=dev))
    del dmat, gmat
    panels = {k: Panel(v, drug_pos.loc[v[DRUG]].to_numpy(),
                       sample_pos.loc[v[SAMPLE]].to_numpy(), dev)
              for k, v in parts.items()}
    if verbose:
        print(f"  features on {dev}: drug {tuple(store.D.shape)} gene {tuple(store.G.shape)}",
              flush=True)
    return store, panels["train"], panels["val"], panels["test"]


def build(drug_fm: str, genomics_fm: str, split_type: str, seed: int,
          test_split_ratio: float, val_split_ratio: float, dev: torch.device,
          verbose: bool = True):
    """Legacy loader, kept so Phase 2A-2 runs stay reproducible."""
    df_master = pd.read_parquet(MASTER_TABLE_PATH, engine="fastparquet")
    df_raw = pd.read_parquet(DRUG_RESPONSE_PATH, engine="fastparquet")
    df_response = df_raw.groupby([DRUG, SAMPLE])["Sensitivity"].mean().reset_index()
    del df_raw

    with open(DRUG_FM_PATH_MAP[drug_fm], "rb") as f:
        drug_feat = pickle.load(f)
    dg = pd.read_parquet(GENOMICS_FM_PATH_MAP[genomics_fm],
                         engine="fastparquet").fillna(0.0).astype(np.float32)
    # The genomics table has a non-unique index: 2,923 rows for 2,173 distinct
    # Sample_IDs, and the duplicate rows are not equal. The published pipeline
    # builds its lookup with dict(zip(dg.index, dg.values)), so the LAST row of a
    # duplicated Sample_ID wins. Reproduced here exactly; changing it would change
    # the substrate and break the paired comparison against M0.
    n_rows_raw = len(dg)
    dg = dg[~dg.index.duplicated(keep="last")]
    if verbose and n_rows_raw != len(dg):
        print(f"  genomics table: {n_rows_raw} rows -> {len(dg)} unique Sample_IDs "
              f"(duplicates resolved last-wins, as in the published pipeline)", flush=True)
    gene_keys = set(dg.index)

    df_f = df_response[df_response[DRUG].isin(drug_feat.keys())
                       & df_response[SAMPLE].isin(gene_keys)]
    cell_ids = df_master[df_master["Model_Type"] == "Cell Line"].index
    df_domain = df_f[df_f[SAMPLE].isin(cell_ids)]

    split_col = SAMPLE if split_type == "LCLO" else DRUG
    uniq = df_domain[split_col].unique()
    val_ratio = val_split_ratio / (1.0 - test_split_ratio)
    trval, test_ids = train_test_split(uniq, test_size=test_split_ratio, random_state=seed)
    train_ids, val_ids = train_test_split(trval, test_size=val_ratio, random_state=seed)
    parts = {}
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        parts[name] = df_domain[df_domain[split_col].isin(ids)].reset_index(drop=True)
    if verbose:
        print(f"  pairs: train={len(parts['train']):,} val={len(parts['val']):,} "
              f"test={len(parts['test']):,}", flush=True)

    # feature matrices, restricted to keys the panel actually uses
    used_drugs = pd.Index(pd.unique(df_domain[DRUG]))
    used_samples = pd.Index(pd.unique(df_domain[SAMPLE]))
    dmat = np.stack([np.asarray(drug_feat[k], dtype=np.float32) for k in used_drugs])
    del drug_feat
    gmat = dg.loc[used_samples].to_numpy(np.float32)
    assert gmat.shape[0] == len(used_samples), (gmat.shape, len(used_samples))
    del dg
    drug_pos = pd.Series(np.arange(len(used_drugs), dtype=np.int64), index=used_drugs)
    sample_pos = pd.Series(np.arange(len(used_samples), dtype=np.int64), index=used_samples)

    store = FeatureStore(torch.as_tensor(dmat, device=dev),
                         torch.as_tensor(gmat, device=dev))
    del dmat, gmat
    panels = {k: Panel(v, drug_pos.loc[v[DRUG]].to_numpy(),
                       sample_pos.loc[v[SAMPLE]].to_numpy(), dev)
              for k, v in parts.items()}
    if verbose:
        print(f"  features on {dev}: drug {tuple(store.D.shape)} gene {tuple(store.G.shape)}",
              flush=True)
    return store, panels["train"], panels["val"], panels["test"]
