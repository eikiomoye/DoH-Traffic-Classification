# DNS-over-HTTPS classification: reviewed research code

Author: **Ikiomoye Douglas Emmanuel**

Paper: [Comparative Classification of DNS-over-HTTPS Traffic on the CIRACIC-DoHBrw-2020 Dataset Using Logistic Regression, Random Forest and XGBoost](https://doi.org/10.63158/journalisi.v8i5.1914), *Journal of Information Systems and Informatics*, 8(5), 6503-6534, 2026.

## Status

This package contains a reviewed derivative of the original notebook supplied by the author on 9 November 2026. The original embedded Python runner is retained under `provenance/`, and saved tabular displays are retained as source evidence. The edited runner and cleaned notebook have not been run on the full dataset in this review. Original result CSVs and the dataset audit have now been supplied and pass 36 independent consistency checks. These are historical outputs, not new measurements.

See [docs/REVIEW.md](docs/REVIEW.md) for the code review and remaining provenance gaps. The review identifies an interrupted execution record and a difference between the manuscript's Python version and the notebook's saved environment. The result CSVs and dataset audit are included. The run-specific environment and completion records are still missing. Do not silently rewrite the publication's environment claims.

## Files

| Path | Contents |
| --- | --- |
| `CIRA_DoH_Layer1_Reviewed.ipynb` | Cleaned Colab notebook, without outputs or temporary upload cells |
| `run_experiment.py` | Extracted and corrected command-line runner |
| `provenance/original_runner.py` | Original embedded runner, unchanged |
| `provenance/source_identity.json` | SHA-256 identities of supplied notebook and original runner |
| `provenance/saved_display_outputs.json` | Source notebook's saved plain-text displays; some tables are truncated |
| `tests/test_review.py` | Isolated helper tests and static notebook/protocol consistency checks |
| `results/original/` | Eight supplied result/audit files, retained byte-for-byte |
| `results/validation_report.json` | 36 result consistency checks |
| `tools/verify_saved_results.py` | Recalculate and reconcile the saved statistics |
| `docs/REVIEW.md` | Findings, changes and remaining evidence gaps |
| `CITATION.cff`, `paper.bib` | Citation metadata for the paper |

## Running

The notebook installs its packages in Colab. Alternatively, install the dependencies into your own virtual environment:

```bash
python -m pip install numpy pandas pyarrow scipy scikit-learn xgboost matplotlib joblib requests
python run_experiment.py --data /path/to/L1-DoH-NonDoH.parquet --workdir cira_reviewed_run --seeds 42 52 62 --n-jobs -1 --importance-sample 20000 --importance-repeats 3
```

The runner expects string labels `DoH` and `NonDoH`, then maps them to 1 and 0. It verifies the full dataset dimensions and counts before fitting. Use a new work directory for each run; existing non-empty outputs are rejected. The notebook additionally uses KaggleHub for data acquisition. Packages installed without version pins constitute a new environment, not a reproduction of a verified historical environment.

The supplied notebook obtains a cleaned derivative from `dhoogla/cicdohbrw2020`. Official dataset context is available at https://www.unb.ca/cic/datasets/dohbrw-2020.html. No dataset is bundled. Do not substitute raw CIC exports for the cleaned parquet file and claim identical preprocessing.

## Validation

```bash
python -m unittest discover -s tests -v
```

The review tests isolate the actual helper functions through Python AST extraction, allowing metrics and audit behaviour to be checked without importing unavailable XGBoost or reading the million-row parquet. These checks do not exercise model fitting, downloads, full CLI execution, plotting or the complete experiment. Syntax and notebook structure were checked. See the review report for exact scope.

Model configurations, training folds, seeds, scoring formulas, class weighting, estimator `.predict()` behaviour and timing boundaries are retained. This is a maintenance derivative, not a fresh experiment. The legacy output column `pr_auc` means scikit-learn Average Precision, not trapezoidal PR-AUC.

## Research provenance and release

This repository publishes the reviewed code alongside the supplied historical results. It does not claim that the full benchmark was rerun during the review. The supplied CSVs and audit are included as author-provided historical results. The original run-specific environment/configuration/completion JSONs should be added when available. Do not put the full dataset, API tokens or large fitted models in Git.

Software release terms for the original code remain to be selected by the author. The paper's CC BY licence does not automatically establish the licence for this software or the dataset.

## Saved-result reconciliation

Run `python tools/verify_saved_results.py` from the repository root. This checks the supplied tables and refreshes `results/validation_report.json`; it does not train models. All 36 checks passed during this review. AP and ROC-AUC are reconciled across seeds, but cannot be recomputed from predictions because probability-level output was not supplied. The dataset SHA-256 recorded by the original audit is `a6e20990d1ee21e1bb118bc4cf87eb886d26755ef5e271bb3b99bd8448009410`. This identifies the audit's claimed input; the underlying parquet bytes have not been independently hashed here.

The GitHub Actions **Review checks** workflow runs the isolated helper tests and saved-result reconciliation on Python 3.12. It does not run the full benchmark.
