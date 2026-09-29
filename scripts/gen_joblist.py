"""Deterministic job list for the representation benchmark, split into disjoint
slices so parallel workers never race on the same configuration."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "drugdis"))
import paths  # noqa: E402
MOL = str(paths.data_dir() / "Molecule_Embeddings")
ALN = str(paths.work_dir() / "aligned_ccle")
OUT = str(paths.runs_dir() / "PaperRerun" / "ECFP4__baseline")
GENE = ["raw", "scGPT", "CellPLM", "BulkFormer", "Geneformer", "scFoundation",
        "SCimilarity", "STATE", "Tahoe_x1", "tGPT", "UCE", "Cell2Sentence"]
DRUG = [("ChemBERTa2", 384), ("Chemprop", 300), ("InfoAlign", 300), ("KPGT", 2304),
        ("MolCLR", 512), ("MolT5", 768), ("Mole_BERT", 300), ("Ouroboros", 2048),
        ("UniMol", 512), ("GeminiMol", 2048), ("UniMolV2", 1024)]
PARTIAL = {"GeminiMol", "UniMolV2"}

jobs = []
for sd in (3407, 3408, 3409):
    for sp in ("LCLO", "LSO"):
        for g in GENE:
            tag = f"gene-{g}"
            extra = "" if g == "raw" else f" --gene_rep_path {ALN}/{g}_ccle_aligned.parquet"
            jobs.append((tag, sp, sd, extra))
        for nm, dim in DRUG:
            tag = f"drug-{nm}"
            extra = f" --drug_rep_path {MOL}/{nm}_emb{dim}.pickle"
            if nm in PARTIAL:
                extra += " --allow_partial_drug_rep"
            jobs.append((tag, sp, sd, extra))

todo = [j for j in jobs
        if not os.path.exists(f"{OUT}/{j[1]}/{j[0]}__M0_seed{j[2]}/results.json")]
print(f"total {len(jobs)} configurations, {len(jobs)-len(todo)} already complete, "
      f"{len(todo)} remaining", file=sys.stderr)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
which = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i, (tag, sp, sd, extra) in enumerate(todo):
    if i % n == which:
        print(f"{tag}|{sp}|{sd}|{extra}")
