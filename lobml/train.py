"""Reproducible model selection and held-out evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, HORIZON, PURGE_ROWS, build_features, load_data, split_masks

CLASSES = [-1, 0, 1]
SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scores(model, x: pd.DataFrame, y: pd.Series) -> dict:
    prediction = model.predict(x)
    probabilities = model.predict_proba(x)
    return {
        "rows": int(len(y)),
        "macro_f1": round(float(f1_score(y, prediction, labels=CLASSES, average="macro", zero_division=0)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, prediction)), 5),
        "log_loss": round(float(log_loss(y, probabilities, labels=model.classes_)), 5),
        "class_support": {str(c): int((y == c).sum()) for c in CLASSES},
        "confusion_matrix": confusion_matrix(y, prediction, labels=CLASSES).tolist(),
    }


def _chart(matrix: list[list[int]], daily: dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    array = np.asarray(matrix)
    axes[0].imshow(array, cmap="Blues")
    axes[0].set(xlabel="Predicted", ylabel="True", title="Held-out confusion matrix")
    axes[0].set_xticks(range(3), ["down", "flat", "up"])
    axes[0].set_yticks(range(3), ["down", "flat", "up"])
    for row in range(3):
        for col in range(3):
            axes[0].text(col, row, str(array[row, col]), ha="center", va="center", color="black")
    days = sorted(daily, key=int)
    axes[1].bar(days, [daily[day]["macro_f1"] for day in days], color="#2463a5")
    axes[1].set(xlabel="Trading day", ylabel="Macro F1", title="Held-out performance by day", ylim=(0, 1))
    fig.savefig(out, dpi=170)
    plt.close(fig)


def _markdown(report: dict) -> str:
    val = report["validation"]
    test = report["test"]
    rows = [
        "# Results",
        "",
        f"Input SHA-256: `{report['data_sha256']}`",
        "",
        "| Model | Validation macro F1 |",
        "|---|---:|",
    ]
    rows.extend(f"| {name} | {score['macro_f1']:.3f} |" for name, score in val.items())
    rows += [
        "",
        f"Selected model: **{report['selected_model']}** (chosen on day 7 only).",
        "",
        "| Held-out days 8–10 | Value |",
        "|---|---:|",
        f"| Macro F1 | {test['macro_f1']:.3f} |",
        f"| Balanced accuracy | {test['balanced_accuracy']:.3f} |",
        f"| Multiclass log loss | {test['log_loss']:.3f} |",
        f"| Samples | {test['rows']:,} |",
        "",
        "![Held-out evaluation](evaluation.png)",
        "",
        "These are historical classification results, not a trading backtest or evidence of profitability.",
        "The normalized/derived FI-2010 benchmark has no executable bid/ask timestamps, queue position,",
        "fees, slippage, or market impact in this project. See [limitations](../docs/LIMITATIONS.md).",
        "",
    ]
    return "\n".join(rows)


def train(data_path: str | Path, out_dir: str | Path) -> dict:
    data_path = Path(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    x = build_features(frame)
    masks = split_masks(frame)
    y = frame[HORIZON].astype(int)
    x_train, y_train = x.loc[masks["train"]], y.loc[masks["train"]]
    x_val, y_val = x.loc[masks["validation"]], y.loc[masks["validation"]]
    x_test, y_test = x.loc[masks["test"]], y.loc[masks["test"]]
    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise ValueError("One or more temporal partitions are empty")

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=500, random_state=SEED),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,
            max_depth=5,
            min_samples_leaf=80,
            learning_rate=0.06,
            l2_regularization=0.1,
            early_stopping=False,
            random_state=SEED,
        ),
    }
    validation = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation[name] = _scores(model, x_val, y_val)
    # Only day 7 selects the model. Days 8–10 are read for evaluation once.
    selected_name = max(models, key=lambda name: validation[name]["macro_f1"])
    selected = models[selected_name]
    test = _scores(selected, x_test, y_test)
    daily = {
        str(day): _scores(selected, x_test.loc[frame.loc[masks["test"], "DAY"].eq(day).to_numpy()],
                          y_test.loc[frame.loc[masks["test"], "DAY"].eq(day).to_numpy()])
        for day in (8, 9, 10)
    }
    report = {
        "dataset": "FI-2010 NoAuction DecPre, derived CSV by seanahmad/fi2010",
        "data_sha256": _sha256(data_path),
        "target": HORIZON,
        "classes": CLASSES,
        "seed": SEED,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "purge_rows_per_stock_day": PURGE_ROWS,
        "split_days": {"train": [1, 2, 3, 4, 5, 6], "validation": [7], "test": [8, 9, 10]},
        "split_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "features": FEATURES,
        "validation": validation,
        "selected_model": selected_name,
        "test": test,
        "test_by_day": daily,
    }
    joblib.dump({"model": selected, "features": FEATURES, "target": HORIZON, "version": 1,
                 "runtime": report["runtime"]}, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _chart(test["confusion_matrix"], daily, out_dir / "evaluation.png")
    (out_dir / "RESULTS.md").write_text(_markdown(report), encoding="utf-8")
    return report
