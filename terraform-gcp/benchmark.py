#!/usr/bin/env python3
"""Train and benchmark LightGBM on the credit-card fraud dataset."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM and record fraud-detection benchmark metrics."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "ml-benchmark" / "creditcard.csv",
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="Path for the JSON result",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def timed_predict(model: lgb.LGBMClassifier, rows: pd.DataFrame) -> float:
    start = time.perf_counter()
    model.predict_proba(rows)[:, 1]
    return time.perf_counter() - start


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {args.data}. Complete README step 4.3 first."
        )

    print(f"Loading dataset: {args.data}")
    load_started = time.perf_counter()
    data = pd.read_csv(args.data)
    load_seconds = time.perf_counter() - load_started

    if "Class" not in data.columns:
        raise ValueError("Expected a target column named 'Class'.")

    features = data.drop(columns="Class")
    target = data["Class"].astype(np.int8)

    # Keep the final test set completely separate from the validation data used
    # for LightGBM early stopping.
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=args.seed,
        stratify=target,
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.20,
        random_state=args.seed,
        stratify=y_train_val,
    )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
        verbosity=-1,
    )

    print(
        f"Training on {len(x_train):,} rows; validating on "
        f"{len(x_valid):,} rows; testing on {len(x_test):,} rows"
    )
    train_started = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    training_seconds = time.perf_counter() - train_started

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int8)

    # Warm up prediction code before measuring latency and throughput.
    model.predict_proba(x_test.iloc[:1000])

    single_row = x_test.iloc[[0]]
    latency_runs = np.array([timed_predict(model, single_row) for _ in range(100)])
    latency_ms = float(np.median(latency_runs) * 1000)

    batch = x_test.iloc[: min(1000, len(x_test))]
    batch_seconds = timed_predict(model, batch)
    throughput_rows_per_second = float(len(batch) / batch_seconds)

    results = {
        "dataset": {
            "path": str(args.data.resolve()),
            "total_rows": int(len(data)),
            "feature_count": int(features.shape[1]),
            "fraud_rows": int(target.sum()),
            "train_rows": int(len(x_train)),
            "validation_rows": int(len(x_valid)),
            "test_rows": int(len(x_test)),
        },
        "timing": {
            "load_data_seconds": round(load_seconds, 6),
            "training_seconds": round(training_seconds, 6),
            "inference_latency_one_row_median_ms": round(latency_ms, 6),
            "inference_batch_1000_seconds": round(batch_seconds, 6),
            "inference_throughput_rows_per_second": round(
                throughput_rows_per_second, 3
            ),
        },
        "model": {
            "algorithm": "LightGBM LGBMClassifier",
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
        },
        "metrics": {
            "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 6),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 6),
            "f1_score": round(float(f1_score(y_test, predictions)), 6),
            "precision": round(float(precision_score(y_test, predictions)), 6),
            "recall": round(float(recall_score(y_test, predictions)), 6),
        },
        "environment": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== LightGBM benchmark result ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved result to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
