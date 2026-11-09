"""Isolated helper checks, not a complete benchmark or model-training test."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics


ROOT = Path(__file__).resolve().parents[1]


def load_helpers():
    """Compile the actual helper definitions without importing XGBoost."""
    tree = ast.parse((ROOT / "run_experiment.py").read_text())
    functions = {"sha256_file", "audit_dataset", "calculate_test_metrics",
                 "summarise_repetitions", "locate_or_download_dataset"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    env = {"Path": Path, "Any": Any, "np": np, "pd": pd, "stats": stats,
           "hashlib": hashlib, "TARGET_COLUMN": "Label", "PARQUET_NAME": "L1-DoH-NonDoH.parquet"}
    for name in ("confusion_matrix", "accuracy_score", "balanced_accuracy_score", "precision_score",
                 "recall_score", "f1_score", "matthews_corrcoef", "average_precision_score", "roc_auc_score"):
        env[name] = getattr(metrics, name)
    exec(compile(module, str(ROOT / "run_experiment.py"), "exec"), env)
    return env


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.env = load_helpers()

    def audit(self, frame):
        # Small, synthetic dimensions only; this does not certify benchmark data.
        self.env.update(EXPECTED_ROWS=len(frame), EXPECTED_COLUMNS=frame.shape[1],
                        EXPECTED_FEATURES=frame.shape[1] - 1,
                        EXPECTED_LABEL_COUNTS=frame.Label.value_counts().to_dict())
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "fixture.parquet"
            p.write_bytes(b"synthetic audit fixture, not a parquet dataset")
            return self.env["audit_dataset"](frame, p)[1].to_dict()

    def frame(self, n=100):
        rng = np.random.default_rng(123)
        f = pd.DataFrame({"Duration": rng.normal(size=n), "PacketLengthMode": rng.normal(size=n)})
        f["Label"] = ["NonDoH"] * (n // 2) + ["DoH"] * (n - n // 2)
        return f

    def test_clean_audit_passes(self):
        result = self.audit(self.frame())
        self.assertEqual(result["validation_status"], "PASSED")
        self.assertEqual(result["dataset_file"], "fixture.parquet")

    def test_constant_column_fails(self):
        f = self.frame(); f["Duration"] = 1
        result = self.audit(f)
        self.assertEqual(result["validation_status"], "FAILED")
        self.assertTrue(any("constant_features=" in item for item in result["validation_failures"]))

    def test_near_constant_column_fails(self):
        f = self.frame(1000); f["Duration"] = [1] * 999 + [2]
        result = self.audit(f)
        self.assertEqual(result["validation_status"], "FAILED")
        self.assertTrue(any("near_constant_features=" in item for item in result["validation_failures"]))

    def test_non_numeric_predictor_produces_audit_failure(self):
        f = self.frame(); f["Duration"] = [str(i) for i in range(len(f))]
        result = self.audit(f)
        self.assertEqual(result["validation_status"], "FAILED")
        self.assertEqual(result["non_numeric_predictors"], ["Duration"])

    def test_infinite_predictor_produces_audit_failure(self):
        f = self.frame(); f.loc[0, "Duration"] = np.inf
        result = self.audit(f)
        self.assertEqual(result["validation_status"], "FAILED")
        self.assertEqual(result["infinite_numeric_cells"], 1)

    def test_metric_ap_and_confusion_counts(self):
        y = pd.Series([0, 0, 1, 1]); p = np.array([.1, .6, .4, .9])
        result, cm = self.env["calculate_test_metrics"](y, (p >= .5).astype(int), p)
        self.assertAlmostEqual(result["pr_auc"], 5 / 6)
        self.assertEqual(cm.tolist(), [[1, 1], [1, 1]])
        self.assertEqual(result["false_positive_rate"], .5)

    def test_summary_uses_sample_sd_and_t_interval(self):
        names = ["accuracy", "balanced_accuracy", "precision_doh", "recall_doh", "macro_f1", "mcc",
                 "pr_auc", "roc_auc", "false_positive_rate", "training_time_seconds", "prediction_time_seconds"]
        f = pd.DataFrame([{**dict.fromkeys(names, v), "model": "test"} for v in (.6, .7, .8)])
        result = self.env["summarise_repetitions"](f).iloc[0]
        self.assertAlmostEqual(result["mean"], .7)
        self.assertAlmostEqual(result["standard_deviation"], .1)
        self.assertAlmostEqual(result["ci95_upper"], .7 + stats.t.ppf(.975, 2) * .1 / np.sqrt(3))

    def test_hash_identifies_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "data"; p.write_bytes(b"abc")
            self.assertEqual(self.env["sha256_file"](p), hashlib.sha256(b"abc").hexdigest())

    def test_invalid_explicit_path_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "L1-DoH-NonDoH.parquet").write_bytes(b"alternate")
            with self.assertRaises(FileNotFoundError):
                self.env["locate_or_download_dataset"](root / "missing.parquet", root, False)

    def test_notebook_runner_matches_script_and_outputs_removed(self):
        n = json.loads((ROOT / "CIRA_DoH_Layer1_Reviewed.ipynb").read_text())
        self.assertEqual(len(n["cells"]), 20)
        for c in n["cells"]:
            if c["cell_type"] == "code":
                self.assertEqual(c["outputs"], [])
                self.assertIsNone(c["execution_count"])
            self.assertNotIn("0x0.st", "".join(c["source"]))
            self.assertNotIn("temp.sh", "".join(c["source"]))
        tree = ast.parse("".join(n["cells"][9]["source"]))
        writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "write_text"]
        self.assertEqual(ast.literal_eval(writes[0].args[0]), (ROOT / "run_experiment.py").read_text())

    def test_protocol_defaults(self):
        tree = ast.parse((ROOT / "run_experiment.py").read_text())
        seeds = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "SEEDS" for t in node.targets)]
        self.assertEqual(seeds, [[42, 52, 62]])
        folds = [ast.literal_eval(kw.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "StratifiedKFold"
                 for kw in node.keywords if kw.arg == "n_splits"]
        self.assertEqual(folds, [3])


if __name__ == "__main__":
    unittest.main()
