"""First three-fold LightGBM submission while the full ensemble is training."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from train import read_data, prepare_features, validate_submission

root = Path(__file__).resolve().parent
data = root / "data"
output = root / "outputs-first"
output.mkdir(exist_ok=True)
start = time.time()
train, test, sample, features, y = read_data(data)
_, _, x, xt, cats = prepare_features(train[features], test[features])
oof = np.full(len(train), np.nan)
test_predictions = np.zeros(len(test))
fold_scores = []
for fold, (fit, val) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(x, y), 1):
    print(f"LightGBM baseline fold {fold}/3", flush=True)
    model = lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.035, num_leaves=63,
                              min_child_samples=80, colsample_bytree=0.85, subsample=0.85,
                              subsample_freq=1, reg_lambda=4, random_state=42 + fold,
                              n_jobs=8, verbosity=-1, deterministic=True, force_col_wise=True)
    model.fit(x.iloc[fit], y.iloc[fit], eval_set=[(x.iloc[val], y.iloc[val])], eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(300)])
    oof[val] = model.predict_proba(x.iloc[val])[:, 1]
    test_predictions += model.predict_proba(xt)[:, 1] / 3
    score = {"fold": fold, "auc": float(roc_auc_score(y.iloc[val], oof[val])),
             "best_iteration": int(model.best_iteration_)}
    fold_scores.append(score)
    print(json.dumps(score), flush=True)
if not np.isfinite(oof).all():
    raise ValueError("Incomplete OOF coverage.")
submission = sample.copy()
submission["satisfaction"] = submission.id.map(pd.Series(test_predictions, index=test.id))
validate_submission(submission, sample)
path = output / "submission.csv"
submission.to_csv(path, index=False)
pd.DataFrame({"id": train.id, "satisfaction": y, "lightgbm": oof}).to_csv(output / "oof.csv", index=False)
report = {"competition": "playground-series-s6e10", "data_source": "input_csv_files",
          "input_sha256": {name: hashlib.sha256((data / name).read_bytes()).hexdigest()
                           for name in ("train.csv", "test.csv", "sample_submission.csv")},
          "submission_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
          "folds": 3, "seed": 42, "iterations": 1200, "fold_scores": fold_scores,
          "train_rows": len(train), "test_rows": len(test),
          "selected_blend": {"catboost_weight": 0, "auc": float(roc_auc_score(y, oof))},
          "seconds": round(time.time() - start, 2),
          "versions": {name: importlib.metadata.version(name)
                       for name in ("pandas", "numpy", "scikit-learn", "lightgbm")}}
(output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report["selected_blend"]), flush=True)
import manage
manage.OUTPUT = output
manage.submit()
