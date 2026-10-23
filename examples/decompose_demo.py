"""DrugDis on a synthetic response table, no download needed.

    python examples/decompose_demo.py

Builds 80 compounds x 120 samples with 30% of pairs observed. The response is a
compound marginal plus a sample marginal plus a smaller drug-sample interaction
plus noise. Three hypothetical predictors are then scored component by component:
one reproduces only the marginals (with drug- and sample-level error, so it stays
additive), one adds pair-level noise that carries no interaction information, and
one also recovers half of the interaction.

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

drugs, samples = frame["SMILES"].unique(), frame["Sample_ID"].unique()
drug_effect = dict(zip(drugs, rng.normal(0, 1.0, n_drug)))
sample_effect = dict(zip(samples, rng.normal(0, 0.4, n_sample)))
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

marginal_error = (frame["SMILES"].map(dict(zip(drugs, rng.normal(0, 0.2, n_drug))))
                  + frame["Sample_ID"].map(dict(zip(samples, rng.normal(0, 0.2, n_sample)))))
pair_noise = rng.normal(0, 0.2, len(frame))
predictors = {"marginals only": additive + marginal_error,
              "+ pair-level noise": additive + pair_noise,
              "+ half the interaction": additive + 0.5 * interaction + pair_noise}
rows = {name: component_profile(frame, yhat=p.to_numpy()) for name, p in predictors.items()}
fields = ["rawPCC", "sharedPCC", "intPCC", "A_int", "emits_interaction", "R2_interaction",
          "MSE_raw", "E_shared", "E_interaction"]


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    return "undefined" if np.isnan(v) else f"{v:.3f}"


table = pd.DataFrame({name: [fmt(r[f]) for f in fields] for name, r in rows.items()}, index=fields)
print(table.to_string())
print("\nThe first emits no interaction, so intPCC is undefined. The second emits"
      "\nuninformative interaction and scores near zero. Only the third recovers it.")
