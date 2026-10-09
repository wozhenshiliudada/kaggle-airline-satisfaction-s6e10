"""Exercise real models on synthetic data; this is not a competition result."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from train import encode_target, run, validate_submission

root = Path(__file__).resolve().parent
data = root / "smoke-data"
output = root / "smoke-output"
data.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
n = 900
frame = pd.DataFrame({
    "id": np.arange(n), "Gender": rng.choice(["Male", "Female"], n),
    "Customer Type": rng.choice(["Loyal Customer", "disloyal Customer"], n),
    "Age": rng.integers(18, 80, n),
    "Type of Travel": rng.choice(["Business travel", "Personal Travel"], n),
    "Class": rng.choice(["Business", "Eco", "Eco Plus"], n),
    "Flight Distance": rng.integers(200, 5000, n),
    "Inflight wifi service": rng.integers(0, 6, n),
    "Online boarding": rng.integers(0, 6, n),
    "Seat comfort": rng.integers(0, 6, n),
    "Departure Delay in Minutes": rng.integers(0, 150, n),
    "Arrival Delay in Minutes": rng.integers(0, 150, n).astype(float),
})
latent = (frame["Online boarding"] + frame["Seat comfort"]
          + 1.5 * frame["Class"].eq("Business") + rng.normal(0, 1.8, n))
target = (latent > 5.5).astype(int)
frame.loc[rng.choice(n, 25, replace=False), "Arrival Delay in Minutes"] = np.nan
frame.loc[rng.choice(n, 15, replace=False), "Customer Type"] = None
train = frame.iloc[:700].copy()
train["satisfaction"] = target.iloc[:700].to_numpy()
test = frame.iloc[700:].copy()
test.loc[test.index[:3], "Class"] = "Unseen cabin"
sample = pd.DataFrame({"id": rng.permutation(test["id"].to_numpy()), "satisfaction": 0.5})
train.to_csv(data / "train.csv", index=False)
test.to_csv(data / "test.csv", index=False)
sample.to_csv(data / "sample_submission.csv", index=False)
report = run(data, output, folds=2, iterations=60, threads=2)
submission = pd.read_csv(output / "submission.csv")
validate_submission(submission, sample)
assert report["selected_blend"]["auc"] > 0.65
assert "id" not in report["features"] and "satisfaction" not in report["features"]
oof = pd.read_csv(output / "oof.csv")
assert oof[["catboost", "lightgbm"]].notna().all().all()
assert encode_target(pd.Series(["satisfied", "neutral or dissatisfied"])).tolist() == [1, 0]
for invalid in (submission.assign(satisfaction=np.nan),
                submission.assign(satisfaction=1.1), submission.iloc[::-1].reset_index(drop=True)):
    try:
        validate_submission(invalid, sample)
    except ValueError:
        pass
    else:
        raise AssertionError("An invalid prediction file was accepted.")
report["data_source"] = "synthetic_smoke_test_not_a_competition_score"
(output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("PASS: real CatBoost/LightGBM training, OOF coverage, unseen categories, missing values, shuffled IDs, and invalid-submission rejection.")
print("These data and scores are SYNTHETIC, not Kaggle competition results.")
