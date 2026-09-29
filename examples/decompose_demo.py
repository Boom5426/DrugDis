"""DrugDis on a synthetic response table, no download needed.

    python examples/decompose_demo.py

Builds 80 compounds x 120 samples with 30% of pairs observed. The response is a
compound marginal plus a sample marginal plus a smaller drug-sample interaction
plus noise. Two hypothetical predictors are then scored component by component:
one reproduces only the marginals, the other also half of the interaction.

The numbers are synthetic. They illustrate the software, not results of the
manuscript's benchmark.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drugdis.evaluate import component_profile, decompose  # noqa: E402

rng = np.random.default_rng(7)
n_drug, n_sample = 80, 120
pairs = [(d, s) for d in range(n_drug) for s in range(n_sample) if rng.random() < 0.3]
frame = pd.DataFrame(pairs, columns=["SMILES", "Sample_ID"])
frame["SMILES"] = "drug" + frame["SMILES"].astype(str)
frame["Sample_ID"] = "sample" + frame["Sample_ID"].astype(str)

drug_effect = dict(zip(frame["SMILES"].unique(), rng.normal(0, 1.0, n_drug)))
sample_effect = dict(zip(frame["Sample_ID"].unique(), rng.normal(0, 0.4, n_sample)))
additive = frame["SMILES"].map(drug_effect) + frame["Sample_ID"].map(sample_effect)
interaction = rng.normal(0, 0.5, len(frame))
frame["Sensitivity"] = additive + interaction + rng.normal(0, 0.2, len(frame))

hy, my, summary = decompose(frame)
print(f"{summary['n_pairs']:,} observed pairs, {summary['n_drugs']} compounds, "
      f"{summary['n_samples']} samples, {summary['components']} connected component(s)")
print(f"additive component:    {summary['share_additive_pct']:5.1f}% of response variance "
      f"on {100 * summary['d_H_over_N']:.1f}% of the degrees of freedom")
print(f"interaction component: {summary['share_interaction_pct']:5.1f}% "
      f"(corr with the additive component {summary['corr_additive_interaction']:+.1e})\n")

noise = rng.normal(0, 0.2, len(frame))
predictors = {"marginals only": additive + noise,
              "marginals + half the interaction": additive + 0.5 * interaction + noise}
rows = {name: component_profile(frame, yhat=p.to_numpy()) for name, p in predictors.items()}
table = pd.DataFrame(rows).T[["rawPCC", "sharedPCC", "intPCC", "A_int", "R2_interaction",
                              "MSE_raw", "E_shared", "E_interaction"]]
print(table.to_string(float_format=lambda v: f"{v:.3f}"))
print("\nBoth predictors track the total response; only the second recovers the interaction.")
