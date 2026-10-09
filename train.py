"""S6E10: reproducible probability predictions with out-of-fold validation.

Runs locally or as a private Kaggle script. No API credential belongs here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

COMPETITION = "playground-series-s6e10"
TARGET = "satisfaction"
ID = "id"


def encode_target(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("Training labels contain missing values.")
    if set(values.unique()).issubset({0, 1}):
        result = values.astype("int8")
    else:
        mapping = {"satisfied": 1, "dissatisfied": 0,
                   "neutral or dissatisfied": 0, "0": 0, "1": 1}
        result = values.astype(str).str.strip().str.lower().map(mapping)
        if result.isna().any():
            raise ValueError("Unrecognized labels; inspect the actual competition data.")
        result = result.astype("int8")
    if result.nunique() != 2:
        raise ValueError("Both binary classes are required.")
    return result


def read_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if list(sample.columns) != [ID, TARGET]:
        raise ValueError("Unexpected submission schema; expected id,satisfaction.")
    for frame in (train, test, sample):
        if ID not in frame or frame[ID].isna().any() or not frame[ID].is_unique:
            raise ValueError("IDs must exist, be nonmissing, and be unique.")
    if TARGET not in train or TARGET in test:
        raise ValueError("The target must appear only in the training file.")
    features = [name for name in train if name not in (ID, TARGET)]
    if set(features) != set(test.columns) - {ID}:
        raise ValueError("Training/test feature sets differ.")
    if len(sample) != len(test) or set(sample[ID]) != set(test[ID]):
        raise ValueError("Sample submission IDs differ from test IDs.")
    return train, test, sample, features, encode_target(train[TARGET])


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["fe_missing_count"] = frame.isna().sum(axis=1).astype("int16")
    canonical = {re.sub(r"[^a-z0-9]+", "", name.lower()): name for name in frame}
    rating_names = ["inflightwifiservice", "departurearrivaltimeconvenient",
                    "easeofonlinebooking", "gatelocation", "foodanddrink",
                    "onlineboarding", "seatcomfort", "inflightentertainment",
                    "onboardservice", "legroomservice", "baggagehandling",
                    "checkinservice", "inflightservice", "cleanliness"]
    rating_cols = [canonical[name] for name in rating_names if name in canonical
                   and pd.api.types.is_numeric_dtype(frame[canonical[name]])]
    if rating_cols:
        ratings = frame[rating_cols]
        result["fe_rating_mean"] = ratings.mean(axis=1)
        result["fe_rating_min"] = ratings.min(axis=1)
        result["fe_rating_max"] = ratings.max(axis=1)
        result["fe_rating_zero_count"] = ratings.eq(0).sum(axis=1)
        result["fe_rating_low_count"] = ratings.le(2).sum(axis=1)
    departure = canonical.get("departuredelayinminutes")
    arrival = canonical.get("arrivaldelayinminutes")
    if departure and arrival:
        result["fe_delay_difference"] = frame[arrival] - frame[departure]
        result["fe_delay_total"] = frame[arrival] + frame[departure]
    for name in result.select_dtypes(include="number"):
        result[name] = result[name].replace([np.inf, -np.inf], np.nan)
    return result


def prepare_features(train_x: pd.DataFrame, test_x: pd.DataFrame):
    train_x, test_x = engineer(train_x), engineer(test_x)
    cats = list(train_x.select_dtypes(include=["object", "string", "category", "bool"]).columns)
    for frame in (train_x, test_x):
        for name in cats:
            frame[name] = frame[name].astype("string").fillna("__MISSING__").astype(str)
    lgb_train, lgb_test = train_x.copy(), test_x.copy()
    for name in cats:
        # The vocabulary uses only unlabeled values, never targets.
        vocabulary = pd.Index(pd.concat([train_x[name], test_x[name]]).unique())
        dtype = pd.CategoricalDtype(categories=vocabulary)
        lgb_train[name] = lgb_train[name].astype(dtype)
        lgb_test[name] = lgb_test[name].astype(dtype)
    return train_x, test_x, lgb_train, lgb_test, cats


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame):
    if list(frame.columns) != [ID, TARGET] or len(frame) != len(sample):
        raise ValueError("Submission has the wrong shape or columns.")
    if not frame[ID].equals(sample[ID]):
        raise ValueError("Submission IDs must follow the official sample order.")
    probability = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("Submission probabilities must be finite and within [0,1].")


def run(data_dir: Path, output_dir: Path, folds=3, iterations=1800, seed=42,
        threads=4, feature_engineering=True):
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test, sample, features, y = read_data(data_dir)
    if not 2 <= folds <= int(y.value_counts().min()):
        raise ValueError("Invalid fold count for the class sizes.")
    if feature_engineering:
        cb_x, cb_test, lg_x, lg_test, cats = prepare_features(train[features], test[features])
    else:
        # Identical preprocessing, with the engineered columns removed.
        cb_x, cb_test, lg_x, lg_test, cats = prepare_features(train[features], test[features])
        cb_x, cb_test = cb_x[features], cb_test[features]
        lg_x, lg_test = lg_x[features], lg_test[features]
    print(json.dumps({"train_rows": len(train), "test_rows": len(test),
                      "features": len(cb_x.columns), "categorical": cats,
                      "positive_rate": float(y.mean())}), flush=True)
    oof = {name: np.full(len(train), np.nan) for name in ("catboost", "lightgbm")}
    prediction = {name: np.zeros(len(test)) for name in oof}
    scores, importance = [], []
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(cb_x, y), 1):
        print(f"Fold {fold}/{folds}: training CatBoost", flush=True)
        cb = CatBoostClassifier(iterations=iterations, depth=7, learning_rate=0.055,
                                loss_function="Logloss", eval_metric="AUC",
                                l2_leaf_reg=5, random_seed=seed + fold,
                                thread_count=threads, allow_writing_files=False,
                                verbose=False)
        cb.fit(cb_x.iloc[fit_idx], y.iloc[fit_idx], cat_features=cats,
               eval_set=(cb_x.iloc[val_idx], y.iloc[val_idx]),
               early_stopping_rounds=120, verbose=250)
        oof["catboost"][val_idx] = cb.predict_proba(cb_x.iloc[val_idx])[:, 1]
        prediction["catboost"] += cb.predict_proba(cb_test)[:, 1] / folds
        print(f"Fold {fold}/{folds}: training LightGBM", flush=True)
        lg = lgb.LGBMClassifier(n_estimators=iterations + iterations // 2,
                               learning_rate=0.035, num_leaves=63,
                               max_depth=-1, min_child_samples=80,
                               colsample_bytree=0.85, subsample=0.85,
                               subsample_freq=1, reg_lambda=4,
                               random_state=seed + fold, n_jobs=threads,
                               verbosity=-1, deterministic=True, force_col_wise=True)
        lg.fit(lg_x.iloc[fit_idx], y.iloc[fit_idx],
               eval_set=[(lg_x.iloc[val_idx], y.iloc[val_idx])], eval_metric="auc",
               callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(300)])
        oof["lightgbm"][val_idx] = lg.predict_proba(lg_x.iloc[val_idx])[:, 1]
        prediction["lightgbm"] += lg.predict_proba(lg_test)[:, 1] / folds
        score = {"fold": fold,
                 "catboost_auc": float(roc_auc_score(y.iloc[val_idx], oof["catboost"][val_idx])),
                 "lightgbm_auc": float(roc_auc_score(y.iloc[val_idx], oof["lightgbm"][val_idx])),
                 "catboost_iterations": int(cb.get_best_iteration() + 1),
                 "lightgbm_iterations": int(lg.best_iteration_)}
        scores.append(score)
        importance.append(pd.DataFrame({"feature": cb_x.columns,
                                        "catboost": cb.feature_importances_,
                                        "lightgbm_gain": lg.booster_.feature_importance("gain")}))
        print(json.dumps(score), flush=True)
        pd.DataFrame({ID: train[ID], TARGET: y, **oof}).to_csv(output_dir / "oof.csv", index=False)
    if any(not np.isfinite(values).all() for values in oof.values()):
        raise ValueError("Incomplete out-of-fold predictions.")
    blends = [{"catboost_weight": weight,
               "auc": float(roc_auc_score(y, weight * oof["catboost"] +
                                          (1 - weight) * oof["lightgbm"]))}
              for weight in (0.0, 0.25, 0.5, 0.75, 1.0)]
    chosen = max(blends, key=lambda item: item["auc"])
    weight = chosen["catboost_weight"]
    test_probability = weight * prediction["catboost"] + (1 - weight) * prediction["lightgbm"]
    by_id = pd.Series(test_probability, index=test[ID])
    submission = sample.copy()
    submission[TARGET] = submission[ID].map(by_id)
    validate_submission(submission, sample)
    submission_path = output_dir / "submission.csv"
    submission.to_csv(submission_path, index=False)
    pd.DataFrame({ID: test[ID], **prediction}).to_csv(output_dir / "test_predictions.csv", index=False)
    pd.concat(importance).groupby("feature").mean().sort_values("catboost", ascending=False).to_csv(
        output_dir / "feature_importance.csv")
    report = {"competition": COMPETITION, "data_source": "input_csv_files",
              "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                               for name in ("train.csv", "test.csv", "sample_submission.csv")},
              "train_rows": len(train), "test_rows": len(test), "feature_count": len(cb_x.columns),
              "features": list(cb_x.columns), "folds": folds, "seed": seed,
              "iterations": iterations, "feature_engineering": feature_engineering,
              "fold_scores": scores, "blend_scores": blends, "selected_blend": chosen,
              "validation_note": "Blend weights were selected on these same OOF predictions; leaderboard score remains unknown.",
              "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),
              "seconds": round(time.time() - started, 2),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("pandas", "numpy", "scikit-learn", "catboost", "lightgbm")}}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_blend": chosen, "submission": str(submission_path),
                      "seconds": report["seconds"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    default_data = Path("data")
    if Path("/kaggle/input").is_dir():
        matches = [path.parent for path in Path("/kaggle/input").rglob("sample_submission.csv")
                   if (path.parent / "train.csv").is_file() and (path.parent / "test.csv").is_file()]
        if len(matches) != 1:
            raise ValueError("Could not locate exactly one competition data directory.")
        default_data = matches[0]
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working")
                        if Path("/kaggle/working").is_dir() else Path("outputs"))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--raw-features", action="store_true")
    args = parser.parse_args()
    run(args.data, args.output, args.folds, args.iterations, args.seed,
        args.threads, not args.raw_features)


if __name__ == "__main__":
    main()
