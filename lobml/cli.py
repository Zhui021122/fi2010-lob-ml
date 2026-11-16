"""Command-line interface for download, validation, training, and batch inference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .data import FEATURES, build_features, load_data

DATA_URL = (
    "https://raw.githubusercontent.com/seanahmad/fi2010/"
    "360fef5a74332c2e2a5bf4ee92ad1a4b2b4fb311/data/data.csv"
)
DATA_SHA256 = "abcf332e6e6c8eb1b29494ff93081204e289ad7592af4e6bed11dd3085247403"
DATA_REPO = "https://github.com/seanahmad/fi2010.git"
DATA_COMMIT = "360fef5a74332c2e2a5bf4ee92ad1a4b2b4fb311"


def _download_with_git(temporary: Path) -> None:
    if shutil.which("git") is None:
        raise RuntimeError("Direct download failed and Git is unavailable")
    with tempfile.TemporaryDirectory(prefix="fi2010-", dir=temporary.parent) as directory:
        commands = [
            ["git", "init", "-q", directory],
            ["git", "-C", directory, "remote", "add", "origin", DATA_REPO],
            ["git", "-C", directory, "fetch", "-q", "--depth", "1", "origin", DATA_COMMIT],
        ]
        for command in commands:
            subprocess.run(command, check=True)
        with temporary.open("wb") as file:
            subprocess.run(
                ["git", "-C", directory, "show", "FETCH_HEAD:data/data.csv"],
                check=True,
                stdout=file,
            )


def download(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        try:
            with urllib.request.urlopen(DATA_URL, timeout=60) as response, temporary.open("wb") as file:
                shutil.copyfileobj(response, file)
        except (OSError, urllib.error.URLError):
            temporary.unlink(missing_ok=True)
            _download_with_git(temporary)
        digest = hashlib.sha256()
        with temporary.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded CSV checksum mismatch")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def predict(data_path: Path, model_path: Path, output: Path) -> None:
    import joblib

    bundle = joblib.load(model_path)
    if bundle.get("features") != FEATURES or bundle.get("version") != 1:
        raise ValueError("Incompatible model bundle")
    frame = load_data(data_path, labeled=False)
    features = build_features(frame)
    model = bundle["model"]
    classes = [int(item) for item in model.classes_]
    probabilities = model.predict_proba(features)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row", "stock", "day", "prediction", *[f"prob_{c}" for c in classes]])
        for index, (stock, day, predicted, probs) in enumerate(
            zip(frame["STOCK"], frame["DAY"], model.predict(features), probabilities, strict=True)
        ):
            writer.writerow([index, int(stock), int(day), int(predicted), *[float(p) for p in probs]])


def main() -> None:
    parser = argparse.ArgumentParser(prog="lobml", description="FI-2010 order-book ML benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download", help="Download a pinned public FI-2010 derivative")
    fetch.add_argument("--output", type=Path, default=Path("data/fi2010.csv"))
    check = commands.add_parser("validate", help="Validate a labeled CSV")
    check.add_argument("--data", type=Path, default=Path("data/fi2010.csv"))
    fit = commands.add_parser("train", help="Train and evaluate the benchmark")
    fit.add_argument("--data", type=Path, default=Path("data/fi2010.csv"))
    fit.add_argument("--out", type=Path, default=Path("artifacts"))
    infer = commands.add_parser("predict", help="Predict from ordered order-book rows")
    infer.add_argument("--data", type=Path, required=True)
    infer.add_argument("--model", type=Path, default=Path("artifacts/model.joblib"))
    infer.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()
    if args.command == "download":
        download(args.output)
        print(f"Verified dataset: {args.output}")
    elif args.command == "validate":
        frame = load_data(args.data)
        print(json.dumps({"rows": len(frame), "stocks": int(frame.STOCK.nunique()), "days": sorted(frame.DAY.unique().tolist())}))
    elif args.command == "train":
        from .train import train

        report = train(args.data, args.out)
        print(json.dumps({"selected_model": report["selected_model"], "test": report["test"]}, indent=2))
    elif args.command == "predict":
        predict(args.data, args.model, args.output)
        print(f"Predictions: {args.output}")


if __name__ == "__main__":
    main()
