# Canonical result tables

Every quantitative result of the manuscript is read from one of these 24 tables.
`TABLES.json` records each file's row count and SHA-256 (`CHECKSUMS.sha256` holds
the same hashes in `sha256sum` format), and `scripts/reproduce_tables.sh`
rebuilds all of them from the processed data and the training runs.

Column names are code field names: `rawPCC`, `sharedPCC` and `intPCC` are the
total-response, additive-component and interaction-component correlations;
`E_shared` and `E_interaction` the additive- and interaction-component errors;
`A_int` the interaction amplitude. Model arms keep their code keys: M3 and M4 are
the manuscript's M1 and M2. [PROJECT_STRUCTURE.md](../../PROJECT_STRUCTURE.md)
maps the other code names to the manuscript's terms.

| table | contents | built by | used in |
| :--- | :--- | :--- | :--- |
| T01 | benchmark dataset definition and checksums | `drugdis/tables/build_tables.py` | Fig. 1b; Supplementary Table 1 |
| T02 | variance decomposition of the benchmark dataset and each test set | `drugdis/tables/build_tables.py` | Fig. 1c,e, 4b; Supplementary Table 2 |
| T03 | cross-assay reproducibility, GDSC1 vs GDSC2 | `drugdis/tables/rebuild_measurement_bootstrap_v2.py` | Fig. 2e, 4e; Supplementary Table 6 |
| T04 | representation benchmark, every run | `drugdis/tables/build_tables_bench.py` | source of T05, T06 |
| T05, T06 | representation benchmark profiles, LCLO and LSO | `drugdis/tables/build_tables_bench.py` | Fig. 3a–d,h; Supplementary Tables 8, 9 |
| T07 | M0 and M4 component-wise recovery profiles | `drugdis/tables/build_tables.py` | Fig. 1d,f, 4c; Supplementary Table 20 |
| T08 | zero-shot organoid evaluation of M0 and M4 | `drugdis/tables/build_tables.py` | Fig. 4c,d,h, 5b,c; Supplementary Tables 21, 22 |
| T09 | ranking agreement across decoders | `drugdis/tables/build_tables.py` | Fig. 3f; Supplementary Table 12 |
| T10 | component-wise error geometry | `drugdis/tables/rebuild_valmse_identity_tables_v1.py` | Fig. 4a,d–f; Supplementary Tables 13, 14 |
| T12 | ranking resolution | `drugdis/tables/separability.py` | Fig. 3a–e; Supplementary Table 10 |
| T13 | calibrated null predictors | `drugdis/tables/rebuild_valmse_identity_tables_v1.py` | Fig. 2d; Supplementary Table 5 |
| T15 | organoid input-compatibility check | `drugdis/tables/build_T15.py` | Fig. 4g; Supplementary Table 15 |
| T16 | decoder profiles | `drugdis/tables/build_T16.py` | Fig. 3g; Supplementary Table 11 |
| T17 | distributions of the response and its components | `drugdis/tables/build_T17.py` | Fig. 1e |
| T18 | interaction R² per run | `drugdis/tables/build_interaction_r2_table.py` | Fig. 5d; Supplementary Table 18 |
| T19 | published evaluation views under prespecified perturbations | `drugdis/tables/run_nm_validation_packages_ab.py` | Fig. 2b,c,f; Supplementary Table 4 |
| T20 | per-resource sensitivity | `drugdis/tables/run_nm_validation_packages_ab.py` | Fig. 2f; Supplementary Table 7 |
| T21, T22 | GDSC1 selection and independent GDSC2 test | `drugdis/tables/run_nm_validation_packages_ab.py` | Fig. 5e–h; Supplementary Tables 19, 17 |
| T23 | matched M0/M3/M4 comparison | `drugdis/tables/audit_current_substrate_m3.py` | Fig. 5a–d; Supplementary Table 16 |
| T24 | zero-shot organoid evaluation of M3 | `drugdis/organoid/evaluate_current_m3_pdo.py` | Fig. 5b–d; Supplementary Table 22 |
| T25 | organoid panel counts | `drugdis/tables/build_T25_pdo_panel_counts.py` | Fig. 1b |
| T26 | decomposition within each response resource | `drugdis/tables/build_T26_resource_decomposition.py` | Supplementary Table 3 |

Figure and table numbers refer to the PDFs in [`manuscript/`](../../manuscript/).
