# Pre-publication notebook review

Reviewed on 9 November 2026 against the supplied manuscript and original notebook. The uploaded original is unchanged. This package is a separate reviewed derivative. This report records the review completed before GitHub publication.

## Main assessment

The executable model settings and displayed aggregate results are consistent with the paper's main three-seed, three-fold comparison. The notebook is a credible source of original experiment code, but its saved execution state does not prove one uninterrupted successful run. It needs provenance clarification before an exact reproduction claim.

## Findings and treatment

| Finding | Evidence | Treatment |
| --- | --- | --- |
| Stale five-fold/five-seed wording | Embedded runner docstring, `--skip-cv` help and display caption | Corrected to three folds and seeds 42, 52, 62; executable settings already used these values |
| Saved interruption | Full-run cell has a KeyboardInterrupt, while later cells display complete-looking three-seed results | Removed saved outputs from clean notebook; retained historical plain-text tables separately; result tables now supplied and reconciled; completion/environment records still needed |
| Environment discrepancy | Saved environment cell reports Python 3.13.15; supplied manuscript says Python 3.12 | Recorded as unresolved; cannot infer the original training interpreter from an independently run environment cell |
| Unnecessary third-party uploads | Appended code posts the dataset to temporary hosting services; one version is code and one is markdown | Removed those appended cells; core experiment's explicitly requested downloads remain |
| Constant-feature checks not enforced | Audit records constant and >=99.9%-dominant features but does not append them to validation failures | Added validation failures; changes acceptance of defective inputs, not the intended clean dataset protocol |
| Non-numeric columns can crash before a useful audit | Float conversion precedes the non-numeric failure report | Infinity check now selects numeric columns; non-numeric predictors are reported as failed validation |
| Stale outputs can be mistaken for a new run | Runner writes into an existing output directory | Added a non-empty-output guard; use a fresh `--workdir` |
| Misleading paper-run flag | Original flag only excludes smoke tests, even if CV or importance is skipped | Flag now requires the complete default three-seed protocol, CV, importance sample and repeats |
| Full held-out importance sample fails | `train_test_split(train_size=len(test))` is invalid | Use the full held-out partition directly when sample size equals its length |
| Missing explicit data path can fall back to another file | Original path finder continues after an invalid `--data` | Explicit invalid path now fails rather than substituting another dataset |
| Metric label is ambiguous | `pr_auc` calls `average_precision_score` | Kept CSV names for compatibility, corrected the permutation plot axis to Average Precision and explained the legacy name |
| Notebook state and clutter | Appended duplicate environment snippets, errors, session metadata and result displays | Cleaned copy retains core 20 cells, removes outputs and session metadata, and re-embeds the reviewed runner |

## Agreement with the paper

- Full dataset checks: 1,019,117 records, 29 predictors, 30 columns including Label, 750,456 NonDoH and 268,661 DoH records.
- 80:20 stratified hold-out splits with seeds 42, 52 and 62.
- Three-fold stratified CV with shuffling and each experiment seed as the fold RNG.
- StandardScaler is inside the Logistic Regression pipeline, so it is fitted separately within training folds.
- Logistic Regression: C=1, lbfgs, 1,000 iterations, balanced class weights.
- Random Forest: 100 trees, Gini, unrestricted depth, square-root feature sampling, balanced_subsample.
- XGBoost: 200 rounds, learning rate 0.05, depth 5, histogram trees, regularisation and sampling settings matching the supplied methods.
- XGBoost's class ratio is calculated from the outer development partition, then held fixed inside its CV folds. This excludes held-out test labels but uses the outer development class distribution. The original semantics are retained and documented rather than silently changed to fold-specific weights.
- Seed-42 models share the same stratified 20,000-record importance sample, sample RNG 42, permutation RNG 42, three repeats and AP scoring.

## Saved results agree at the reported precision

The following values are historical displays from the uploaded notebook, not measurements made in this review.

| Quantity | Notebook display | Rounded paper value |
| --- | ---: | ---: |
| Random Forest macro F1 | 0.996499 | 0.9965 |
| Random Forest MCC | 0.993002 | 0.9930 |
| Random Forest FPR | 0.001073 | 0.0011 |
| XGBoost Average Precision | 0.999376 | 0.9994 |
| Random Forest mean training seconds | 743.417114 | 743.42 |
| XGBoost mean training seconds | 31.399257 | 31.40 |
| Logistic Regression FPR | 0.119729 | 0.1197 |

These matches support consistency between the displayed results and manuscript. They do not independently verify the dataset hash, execution history or training environment. All code cells have null execution counts despite retained outputs, so execution ordering cannot be reconstructed from counts.

## Numerical choices deliberately preserved

The runner uses each estimator's `.predict()` for labels, with `.predict_proba()` computed separately for ranking metrics. It does not explicitly threshold a single shared probability vector. Exactly-0.5 ties can have estimator-specific treatment. Timing therefore includes both label and probability prediction calls. Altering this now could change reported labels or prediction times; a future refactor should be a separately validated experiment.

The original output already stores tn/fp/fn/tp by seed in `test_metrics_by_seed.csv`, in addition to summed confusion matrices. There is no need to invent missing per-seed matrices from aggregate values.

The parquet version is not pinned by the original KaggleHub call. Recording a hash identifies newly retrieved bytes, but comparison with the recorded original audit hash is needed to check agreement with the claimed training input. The supplied audit now provides that hash. The source's XGBoost 3.4.1 and scikit-learn 1.6.1 environment display is retained as evidence, not independently verified package-installation history.

## Tests and limits

Syntax compilation, notebook structure and isolated helper tests were performed. Helper checks cover valid/invalid dataset audits, constant and near-constant rejection, non-numeric failure reporting, metrics, statistical summaries, hashing and explicit input paths. Static checks verify the three-fold/three-seed defaults, absence of saved outputs and agreement between the notebook's embedded runner and standalone script. Tests use small synthetic frames, not the benchmark dataset.

The full experiment, original downloads and estimator fits were not rerun. XGBoost and pyarrow were unavailable in the local review environment. No data has been uploaded to external services during review, and no remote code from the notebook has been executed. A simple pattern scan found no obvious API token or private-key strings in the cell sources; this is not a guarantee that every possible secret format was detected.

## Update after receipt of the result files

Eight original result/audit files were supplied after the initial review. They are included byte-for-byte under `results/original/`, with their own SHA-256 identities recorded separately. All 36 reconciliation checks pass. These checks cover 27 fold rows, nine model/seed test rows, all reported summary means/standard deviations/t confidence intervals, confusion-derived metrics, summed matrices, the 29-feature schema and 87 permutation entries.

The original audit records dataset SHA-256:

`a6e20990d1ee21e1bb118bc4cf87eb886d26755ef5e271bb3b99bd8448009410`

It reports a 107,043,097-byte parquet file, 1,019,117 records, 29 predictors and no validation defects. These are claims in the supplied audit, corroborated where possible by the result tables. The raw parquet was not supplied, so its bytes and data-quality checks were not independently rerun.

AP and ROC-AUC cannot be independently recalculated from confusion matrices. Their summary arithmetic agrees with the per-seed metrics; probability-level outputs would be required to re-evaluate those metrics themselves. Mean training time is 743.417114 seconds for Random Forest and 31.399257 seconds for XGBoost, which supports the paper's approximately 95.8% time reduction at the supplied precision. Hardware timing itself has not been reproduced.

## Remaining provenance gaps

The run-specific `environment.json`, `experiment_manifest.json` and `model_configurations.json` were not included. Those files could establish which recorded environment, settings and completion status accompanied these results. The Python 3.12 versus 3.13.15 discrepancy remains unresolved. Saved outputs with a KeyboardInterrupt can coexist with results from an earlier run, so these files do not establish one uninterrupted run of the uploaded notebook.

Publication can describe the package accurately as author-supplied experiment code, a documented maintenance derivative and supplied historical results. It should not claim that the reviewer reran the full benchmark or verified the physical dataset bytes. Retain the environment discrepancy until the original run record resolves it.
