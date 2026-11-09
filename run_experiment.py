#!/usr/bin/env python3
"""
Reviewed derivative of the supplied CIRA-CIC-DoHBrw-2020 Layer 1 runner.
Original model settings and numerical evaluation are retained. See docs/REVIEW.md.

Task:
    DoH versus NonDoH HTTPS traffic classification.

Models:
    Logistic Regression, Random Forest and XGBoost.

Protocol:
    Stratified 80% development and 20% untouched test split.
    Stratified three-fold cross-validation within the development set.
    Three fixed seeds: 42, 52 and 62.
    Class weighting is derived from the development data.
    Standardisation is fitted within the Logistic Regression pipeline only.

The script never fabricates or fills experimental results. It validates the
dataset before model fitting and writes all measured outputs to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import scipy
import sklearn
import xgboost
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


EXPECTED_ROWS = 1_019_117
EXPECTED_COLUMNS = 30
EXPECTED_FEATURES = 29
EXPECTED_LABEL_COUNTS = {"DoH": 268_661, "NonDoH": 750_456}
SEEDS = [42, 52, 62]
POSITIVE_LABEL = "DoH"
NEGATIVE_LABEL = "NonDoH"
TARGET_COLUMN = "Label"
PARQUET_NAME = "L1-DoH-NonDoH.parquet"

DOWNLOAD_URLS = [
    (
        "ResearchGate archive",
        "https://www.researchgate.net/profile/Laurens-Dhooge/"
        "publication/367633971_CIRA-CIC-DoHBrw2020-Improved/"
        "links/63d954be62d2a24f92e27561/"
        "CIRA-CIC-DoHBrw2020-Improved.zip",
    ),
    (
        "Kaggle public dataset API",
        "https://www.kaggle.com/api/v1/datasets/download/"
        "dhoogla/cicdohbrw2020",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CIRA-CIC-DoHBrw-2020 Layer 1 experiment."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to L1-DoH-NonDoH.parquet.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path("cira_doh_run"),
        help="Directory for downloaded data and experimental outputs.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the public improved archive when the parquet file is absent.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Threads used within Random Forest and XGBoost. Default: all available.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="Repeated split and model seeds.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip three-fold development cross-validation. Not suitable for the paper run.",
    )
    parser.add_argument(
        "--skip-importance",
        action="store_true",
        help="Skip permutation importance.",
    )
    parser.add_argument(
        "--importance-sample",
        type=int,
        default=20_000,
        help="Maximum held-out records used for permutation importance.",
    )
    parser.add_argument(
        "--importance-repeats",
        type=int,
        default=3,
        help="Permutation repetitions for each feature.",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save fitted seed-42 estimators with joblib.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Use a stratified sample of 50,000 records and the first seed only. "
            "Outputs are diagnostic and must not be reported as paper results."
        ),
    )
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        zf.extractall(destination)


def stream_download(url: str, destination: Path) -> None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/150 Safari/537.36"
        )
    }
    with requests.get(
        url,
        headers=headers,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if destination.stat().st_size < 1_000_000:
        head = destination.read_bytes()[:300].lower()
        if b"<html" in head or b"<!doctype html" in head:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"The server returned an HTML page rather than the archive "
                f"(content type: {content_type or 'unknown'})."
            )


def locate_or_download_dataset(
    requested_path: Path | None,
    workdir: Path,
    permit_download: bool,
) -> Path:
    candidates: list[Path] = []
    if requested_path is not None:
        if not requested_path.is_file():
            raise FileNotFoundError(f"Explicit dataset path does not exist: {requested_path}")
        return requested_path.resolve()
    candidates.extend(
        [
            Path(PARQUET_NAME),
            workdir / PARQUET_NAME,
            workdir / "data" / PARQUET_NAME,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    recursively_found = list(workdir.rglob(PARQUET_NAME))
    if recursively_found:
        return recursively_found[0].resolve()

    if not permit_download:
        raise FileNotFoundError(
            f"{PARQUET_NAME} was not found. Supply --data PATH or add --download."
        )

    data_dir = workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for source_name, url in DOWNLOAD_URLS:
        archive = data_dir / "CIC-DoHBrw2020-Improved.zip"
        archive.unlink(missing_ok=True)
        try:
            print(f"Downloading from {source_name}...")
            stream_download(url, archive)
            if not zipfile.is_zipfile(archive):
                raise RuntimeError("Downloaded object is not a valid ZIP archive.")
            safe_extract_zip(archive, data_dir)
            found = list(data_dir.rglob(PARQUET_NAME))
            if not found:
                raise FileNotFoundError(
                    f"{PARQUET_NAME} was not present in the downloaded archive."
                )
            return found[0].resolve()
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            archive.unlink(missing_ok=True)

    raise RuntimeError(
        "The public archive could not be retrieved automatically. "
        "Place L1-DoH-NonDoH.parquet in the working directory.\n"
        + "\n".join(errors)
    )


def audit_dataset(df: pd.DataFrame, dataset_path: Path) -> tuple[list[str], pd.Series]:
    label_counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]

    numeric_frame = df[feature_columns]
    non_numeric = [
        c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    infinite_cells = int(
        np.isinf(numeric_frame.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64, copy=False)).sum()
    )
    constant_features = [
        c for c in feature_columns if df[c].nunique(dropna=False) <= 1
    ]
    near_constant_features = []
    for column in feature_columns:
        proportions = df[column].value_counts(normalize=True, dropna=False)
        if not proportions.empty and float(proportions.iloc[0]) >= 0.999:
            near_constant_features.append(
                {
                    "feature": column,
                    "dominant_proportion": float(proportions.iloc[0]),
                }
            )

    audit = {
        "dataset_file": dataset_path.name,
        "sha256": sha256_file(dataset_path),
        "file_size_bytes": dataset_path.stat().st_size,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "predictor_count": len(feature_columns),
        "target": TARGET_COLUMN,
        "label_counts": label_counts,
        "missing_cells": missing_cells,
        "exact_duplicate_rows": duplicate_rows,
        "infinite_numeric_cells": infinite_cells,
        "non_numeric_predictors": non_numeric,
        "constant_features": constant_features,
        "near_constant_features_99_9_percent": near_constant_features,
        "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
    }

    failures = []
    if df.shape[0] != EXPECTED_ROWS:
        failures.append(f"rows={df.shape[0]}, expected {EXPECTED_ROWS}")
    if df.shape[1] != EXPECTED_COLUMNS:
        failures.append(f"columns={df.shape[1]}, expected {EXPECTED_COLUMNS}")
    if len(feature_columns) != EXPECTED_FEATURES:
        failures.append(
            f"predictors={len(feature_columns)}, expected {EXPECTED_FEATURES}"
        )
    if label_counts != EXPECTED_LABEL_COUNTS:
        failures.append(
            f"label_counts={label_counts}, expected {EXPECTED_LABEL_COUNTS}"
        )
    if missing_cells != 0:
        failures.append(f"missing_cells={missing_cells}, expected 0")
    if duplicate_rows != 0:
        failures.append(f"duplicate_rows={duplicate_rows}, expected 0")
    if infinite_cells != 0:
        failures.append(f"infinite_cells={infinite_cells}, expected 0")
    if non_numeric:
        failures.append(f"non_numeric_predictors={non_numeric}")
    if constant_features:
        failures.append(f"constant_features={constant_features}")
    if near_constant_features:
        failures.append(f"near_constant_features={near_constant_features}")

    if failures:
        audit["validation_status"] = "FAILED"
        audit["validation_failures"] = failures
    else:
        audit["validation_status"] = "PASSED"
        audit["validation_failures"] = []

    return feature_columns, pd.Series(audit, dtype=object)


def build_models(seed: int, scale_pos_weight: float, n_jobs: int) -> dict[str, Any]:
    logistic = Pipeline(
        steps=[
            ("standardise", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        bootstrap=True,
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
        random_state=seed,
    )

    xgboost_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        n_jobs=n_jobs,
        random_state=seed,
        verbosity=1,
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "XGBoost": xgboost_model,
    }


def probability_scores(estimator: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X))
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError("Estimator provides neither predict_proba nor decision_function.")


def calculate_test_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else np.nan

    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "precision_doh": precision_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "recall_doh": recall_score(
            y_true, predictions, pos_label=1, zero_division=0
        ),
        "macro_f1": f1_score(y_true, predictions, average="macro"),
        "mcc": matthews_corrcoef(y_true, predictions),
        "pr_auc": average_precision_score(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "false_positive_rate": fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics, cm


def summarise_repetitions(results: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision_doh",
        "recall_doh",
        "macro_f1",
        "mcc",
        "pr_auc",
        "roc_auc",
        "false_positive_rate",
        "training_time_seconds",
        "prediction_time_seconds",
    ]
    rows = []
    for model_name, group in results.groupby("model", sort=False):
        n = len(group)
        t_critical = stats.t.ppf(0.975, df=n - 1) if n > 1 else np.nan
        for metric in metric_columns:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            ci_half = (
                float(t_critical * std / np.sqrt(n))
                if n > 1 and np.isfinite(std)
                else np.nan
            )
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "n_repetitions": n,
                    "mean": mean,
                    "standard_deviation": std,
                    "ci95_lower": mean - ci_half if np.isfinite(ci_half) else np.nan,
                    "ci95_upper": mean + ci_half if np.isfinite(ci_half) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def create_plots(
    test_results: pd.DataFrame,
    summary: pd.DataFrame,
    confusion_totals: dict[str, np.ndarray],
    importance_frame: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for metric in ["macro_f1", "mcc", "pr_auc", "false_positive_rate"]:
        metric_summary = summary[summary["metric"] == metric].copy()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        errors = np.vstack(
            [
                metric_summary["mean"] - metric_summary["ci95_lower"],
                metric_summary["ci95_upper"] - metric_summary["mean"],
            ]
        )
        ax.bar(
            metric_summary["model"],
            metric_summary["mean"],
            yerr=errors,
            capsize=5,
        )
        ax.set_ylabel(metric.replace("_", " ").upper())
        ax.set_title(f"Test {metric.replace('_', ' ').title()} across three seeds")
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(plot_dir / f"test_{metric}.png", dpi=220)
        plt.close(fig)

    for model_name, matrix in confusion_totals.items():
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=[NEGATIVE_LABEL, POSITIVE_LABEL],
        )
        display.plot(ax=ax, values_format="d")
        ax.set_title(f"{model_name}, summed test confusion matrix")
        fig.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        fig.savefig(plot_dir / f"confusion_{safe_name}.png", dpi=220)
        plt.close(fig)

    if importance_frame is not None and not importance_frame.empty:
        for model_name, group in importance_frame.groupby("model", sort=False):
            top = group.sort_values("importance_mean", ascending=False).head(15)
            top = top.sort_values("importance_mean", ascending=True)
            fig, ax = plt.subplots(figsize=(8.5, 6.5))
            ax.barh(
                top["feature"],
                top["importance_mean"],
                xerr=top["importance_std"],
                capsize=3,
            )
            ax.set_xlabel("Decrease in held-out Average Precision after permutation")
            ax.set_title(f"{model_name}, permutation importance")
            fig.tight_layout()
            safe_name = model_name.lower().replace(" ", "_")
            fig.savefig(
                plot_dir / f"permutation_importance_{safe_name}.png",
                dpi=220,
            )
            plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.importance_sample < 1 or args.importance_repeats < 1:
        raise ValueError("Importance sample and repeats must be positive.")
    if args.n_jobs == 0 or len(args.seeds) != len(set(args.seeds)):
        raise ValueError("n-jobs cannot be zero and seeds must be unique.")
    args.workdir.mkdir(parents=True, exist_ok=True)
    output_dir = args.workdir / "outputs"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty. Use a new --workdir to avoid mixing runs.")
    output_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("default")
    start_wall = time.time()

    dataset_path = locate_or_download_dataset(
        args.data,
        args.workdir,
        args.download,
    )
    print(f"Reading {dataset_path}...")
    df = pd.read_parquet(dataset_path)

    feature_columns, audit_series = audit_dataset(df, dataset_path)
    audit = audit_series.to_dict()
    write_json(output_dir / "dataset_audit.json", audit)
    if audit["validation_status"] != "PASSED":
        raise ValueError(
            "Dataset validation failed. See outputs/dataset_audit.json. "
            + "; ".join(audit["validation_failures"])
        )

    feature_summary = df[feature_columns].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    feature_summary["dtype"] = [str(df[c].dtype) for c in feature_summary.index]
    feature_summary["unique_values"] = [
        int(df[c].nunique(dropna=False)) for c in feature_summary.index
    ]
    feature_summary.to_csv(output_dir / "feature_summary.csv", index_label="feature")

    if args.smoke_test:
        print("SMOKE TEST: results are not suitable for the paper.")
        df, _ = train_test_split(
            df,
            train_size=min(50_000, len(df)),
            stratify=df[TARGET_COLUMN],
            random_state=args.seeds[0],
        )
        active_seeds = [args.seeds[0]]
    else:
        active_seeds = args.seeds

    X = df[feature_columns]
    y = df[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}).astype(np.int8)

    cv_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    confusion_totals: dict[str, np.ndarray] = {}
    seed_42_models: dict[str, Any] = {}
    seed_42_test: tuple[pd.DataFrame, pd.Series] | None = None
    config_record: dict[str, Any] = {}

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision_doh": "precision",
        "recall_doh": "recall",
        "macro_f1": "f1_macro",
        "mcc": "matthews_corrcoef",
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
    }

    for seed in active_seeds:
        print(f"\nSeed {seed}: creating stratified 80/20 split...")
        X_dev, X_test, y_dev, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            stratify=y,
            random_state=seed,
        )

        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=y_dev,
        )
        negative_count = int((y_dev == 0).sum())
        positive_count = int((y_dev == 1).sum())
        scale_pos_weight = negative_count / positive_count

        models = build_models(seed, scale_pos_weight, args.n_jobs)
        if not config_record:
            config_record = {
                "seeds": active_seeds,
                "split": {
                    "development_fraction": 0.80,
                    "test_fraction": 0.20,
                    "stratified": True,
                    "seed_controls_split_and_estimator": True,
                },
                "cross_validation": {
                    "folds": 3,
                    "stratified": True,
                    "shuffle": True,
                    "performed": not args.skip_cv,
                },
                "class_imbalance": {
                    "negative_class": NEGATIVE_LABEL,
                    "positive_class": POSITIVE_LABEL,
                    "logistic_regression": "class_weight='balanced'",
                    "random_forest": "class_weight='balanced_subsample'",
                    "xgboost": "scale_pos_weight = development NonDoH / DoH",
                    "example_development_class_weights": {
                        NEGATIVE_LABEL: float(class_weights[0]),
                        POSITIVE_LABEL: float(class_weights[1]),
                    },
                    "example_xgboost_scale_pos_weight": scale_pos_weight,
                },
                "models": {
                    name: estimator.get_params(deep=True)
                    for name, estimator in models.items()
                },
                "decision_threshold": 0.5,
                "positive_class": POSITIVE_LABEL,
                "primary_metrics": ["macro_f1", "mcc", "pr_auc"],
                "additional_metrics": [
                    "accuracy",
                    "balanced_accuracy",
                    "precision_doh",
                    "recall_doh",
                    "roc_auc",
                    "false_positive_rate",
                    "training_time_seconds",
                    "prediction_time_seconds",
                ],
            }

        for model_name, estimator in models.items():
            print(f"Seed {seed}, {model_name}")

            if not args.skip_cv:
                cv = StratifiedKFold(
                    n_splits=3,
                    shuffle=True,
                    random_state=seed,
                )
                cv_result = cross_validate(
                    clone(estimator),
                    X_dev,
                    y_dev,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=1,
                    error_score="raise",
                )
                for fold_index in range(3):
                    row = {
                        "seed": seed,
                        "model": model_name,
                        "fold": fold_index + 1,
                        "fit_time_seconds": cv_result["fit_time"][fold_index],
                        "score_time_seconds": cv_result["score_time"][fold_index],
                    }
                    for metric_name in scoring:
                        row[metric_name] = cv_result[
                            f"test_{metric_name}"
                        ][fold_index]
                    cv_rows.append(row)

            fit_start = time.perf_counter()
            estimator.fit(X_dev, y_dev)
            training_time = time.perf_counter() - fit_start

            prediction_start = time.perf_counter()
            predictions = estimator.predict(X_test)
            probabilities = probability_scores(estimator, X_test)
            prediction_time = time.perf_counter() - prediction_start

            metric_values, cm = calculate_test_metrics(
                y_test,
                predictions,
                probabilities,
            )
            test_rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "development_records": len(X_dev),
                    "test_records": len(X_test),
                    "development_non_doh": negative_count,
                    "development_doh": positive_count,
                    "xgboost_scale_pos_weight": (
                        scale_pos_weight if model_name == "XGBoost" else np.nan
                    ),
                    "training_time_seconds": training_time,
                    "prediction_time_seconds": prediction_time,
                    **metric_values,
                }
            )
            confusion_totals.setdefault(model_name, np.zeros((2, 2), dtype=np.int64))
            confusion_totals[model_name] += cm

            if seed == 42:
                seed_42_models[model_name] = estimator
                seed_42_test = (X_test, y_test)

    cv_frame = pd.DataFrame(cv_rows)
    if not cv_frame.empty:
        cv_frame.to_csv(output_dir / "cv_fold_metrics.csv", index=False)
        cv_summary = (
            cv_frame.groupby(["model"], sort=False)
            .agg(
                {
                    col: ["mean", "std"]
                    for col in [
                        "accuracy",
                        "balanced_accuracy",
                        "precision_doh",
                        "recall_doh",
                        "macro_f1",
                        "mcc",
                        "pr_auc",
                        "roc_auc",
                        "fit_time_seconds",
                        "score_time_seconds",
                    ]
                }
            )
        )
        cv_summary.columns = [
            "_".join(parts) for parts in cv_summary.columns.to_flat_index()
        ]
        cv_summary.reset_index().to_csv(
            output_dir / "cv_metrics_summary.csv",
            index=False,
        )

    test_frame = pd.DataFrame(test_rows)
    test_frame.to_csv(output_dir / "test_metrics_by_seed.csv", index=False)
    test_summary = summarise_repetitions(test_frame)
    test_summary.to_csv(output_dir / "test_metrics_summary.csv", index=False)

    confusion_rows = []
    for model_name, matrix in confusion_totals.items():
        tn, fp, fn, tp = matrix.ravel()
        confusion_rows.append(
            {
                "model": model_name,
                "summed_tn": int(tn),
                "summed_fp": int(fp),
                "summed_fn": int(fn),
                "summed_tp": int(tp),
            }
        )
    pd.DataFrame(confusion_rows).to_csv(
        output_dir / "summed_confusion_matrices.csv",
        index=False,
    )

    importance_frame: pd.DataFrame | None = None
    if (
        not args.skip_importance
        and seed_42_models
        and seed_42_test is not None
    ):
        X_test_42, y_test_42 = seed_42_test
        sample_n = min(args.importance_sample, len(X_test_42))
        if sample_n == len(X_test_42):
            X_imp, y_imp = X_test_42, y_test_42
        else:
            X_imp, _, y_imp, _ = train_test_split(
                X_test_42,
                y_test_42,
                train_size=sample_n,
                stratify=y_test_42,
                random_state=42,
            )
        importance_rows = []
        for model_name, estimator in seed_42_models.items():
            print(f"Permutation importance: {model_name}")
            result = permutation_importance(
                estimator,
                X_imp,
                y_imp,
                scoring="average_precision",
                n_repeats=args.importance_repeats,
                random_state=42,
                n_jobs=args.n_jobs,
            )
            for feature, mean_value, std_value in zip(
                feature_columns,
                result.importances_mean,
                result.importances_std,
            ):
                importance_rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "importance_mean": mean_value,
                        "importance_std": std_value,
                        "scoring": "average_precision",
                        "held_out_sample_records": sample_n,
                        "permutation_repeats": args.importance_repeats,
                        "seed": 42,
                    }
                )
        importance_frame = pd.DataFrame(importance_rows)
        importance_frame.to_csv(
            output_dir / "permutation_importance.csv",
            index=False,
        )

    write_json(output_dir / "model_configurations.json", config_record)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }
    write_json(output_dir / "environment.json", environment)

    create_plots(
        test_frame,
        test_summary,
        confusion_totals,
        importance_frame,
        output_dir,
    )

    if args.save_models and seed_42_models:
        model_dir = output_dir / "models_seed_42"
        model_dir.mkdir(parents=True, exist_ok=True)
        for model_name, estimator in seed_42_models.items():
            safe_name = model_name.lower().replace(" ", "_")
            joblib.dump(estimator, model_dir / f"{safe_name}.joblib")

    manifest = {
        "status": "completed",
        "paper_run": (not args.smoke_test and not args.skip_cv
                      and not args.skip_importance and args.seeds == SEEDS
                      and args.importance_sample == 20_000
                      and args.importance_repeats == 3),
        "dataset_validation": audit["validation_status"],
        "dataset_sha256": audit["sha256"],
        "seeds": active_seeds,
        "cross_validation_performed": not args.skip_cv,
        "permutation_importance_performed": importance_frame is not None,
        "elapsed_wall_seconds": time.time() - start_wall,
        "output_files": sorted(
            str(path.relative_to(args.workdir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    write_json(output_dir / "experiment_manifest.json", manifest)

    print(f"\nCompleted. Outputs: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
