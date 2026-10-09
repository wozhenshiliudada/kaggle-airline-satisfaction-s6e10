"""Record only aggregate diagnostics from the official downloaded data."""
from pathlib import Path
import json
from train import read_data

root = Path(__file__).resolve().parent
train, test, sample, features, y = read_data(root / "data")
report = {"competition": "playground-series-s6e10", "train_rows": len(train),
          "test_rows": len(test), "raw_features": features,
          "positive_rate": float(y.mean()), "target_values": [str(value) for value in train.satisfaction.unique()],
          "train_test_ids_overlap": len(set(train.id).intersection(test.id)),
          "duplicate_feature_rows_train": int(train[features].duplicated().sum()),
          "columns": {name: {"dtype": str(train[name].dtype),
                             "train_missing": int(train[name].isna().sum()),
                             "test_missing": int(test[name].isna().sum()),
                             "unique_train": int(train[name].nunique())}
                      for name in features}}
(root / "data-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
