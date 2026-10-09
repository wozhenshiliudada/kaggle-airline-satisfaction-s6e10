"""Two-stage audit for freshly trained development/holdout comparisons.

Stage ``dev`` chooses among fixed carriers using development predictions only.
Stage ``finalize`` checks that locked choice on the excluded holdout exactly once.
The development stack cross-fit is a selection proxy, not nested base-model CV.
No historical OOF is consumed and this module never uploads a submission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
COMPETITION = "playground-series-s6e10"
FIXED_STACK = dict(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
NOTE = ("All supervised members are freshly trained with the same excluded "
        "H1/H2 holdouts. Development meta-model cross-fit reuses overlapping base-model "
        "training folds and is not fully nested CV. Every training label has "
        "participated in earlier project research; this is a newly excluded "
        "retraining holdouts, not pristine data. H2 was prospectively reserved "
        "before round16; it is used only for the predeclared fallback round17. "
        "Auxiliary feature construction "
        "may be unlabeled transductive as disclosed in the protocol.")
TRANSFER_NOTE = ("The locked comparison validates the freshly retrained baseline-plus-carrier "
                 "structure. Reusing the historical full-training baseline and a new full-training "
                 "five-fold carrier changes training coverage and native fold composition. "
                 "Transferred coefficients remain fixed from development; this holdout does not "
                 "provide an unbiased evaluation of that final production CSV. Any historical "
                 "OOF compatibility gate is adaptive and must be reported separately.")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def path(value):
    result = Path(value)
    if not result.is_absolute():
        result = ROOT / result
    result = result.resolve()
    result.relative_to(ROOT.resolve())
    return result


def load(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def save(value, obj):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def rel(value):
    return str(path(value).relative_to(ROOT)).replace("\\", "/")


def check_probability(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (shape,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid prediction shape/finite values: {name}")
    if ((value < 0) | (value > 1)).any():
        raise ValueError(f"Invalid probability range: {name}")
    return value


def check_idx(value, total, name):
    value = np.asarray(value)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"Index must be one-dimensional integers: {name}")
    if len(np.unique(value)) != len(value) or (value < 0).any() or (value >= total).any():
        raise ValueError(f"Duplicate/out-of-range index: {name}")
    return value.astype(np.int64)


def assert_same(actual, expected, name):
    if not np.array_equal(actual, expected):
        raise ValueError("Alignment mismatch: " + name)


def paired(y, old, new):
    def parts(p):
        positive, negative = p[y == 1], p[y == 0]
        sp, sn = np.sort(positive), np.sort(negative)
        vp = (np.searchsorted(sn, positive, side="left") +
              np.searchsorted(sn, positive, side="right")) / (2 * len(sn))
        vn = 1 - (np.searchsorted(sp, negative, side="left") +
                  np.searchsorted(sp, negative, side="right")) / (2 * len(sp))
        return vp, vn
    a, b = parts(old)
    c, d = parts(new)
    se = float(np.sqrt(np.var(c - a, ddof=1) / len(a) +
                       np.var(d - b, ddof=1) / len(b)))
    delta = float(roc_auc_score(y, new) - roc_auc_score(y, old))
    return dict(delta=delta, fixed_prediction_se=se,
                approx_95ci=[delta - 1.96 * se, delta + 1.96 * se],
                limitation="Fixed prediction, independent-case approximation; "
                "training variability, prior adaptive research, and model "
                "selection uncertainty are not corrected.")


def read_inputs(manifest):
    protocol = load(manifest)
    if protocol["competition"] != COMPETITION or protocol["folds"] != 3:
        raise ValueError("Wrong competition/fold protocol")
    if protocol["dev_min_delta"] != 1e-5:
        raise ValueError("Development selection gate changed")
    if protocol.get("stack_parameters", FIXED_STACK) != FIXED_STACK:
        raise ValueError("Fixed logistic stack parameters changed")
    actual_hashes = {k: sha(ROOT / "data" / k) for k in
                     ("train.csv", "test.csv", "sample_submission.csv")}
    if actual_hashes != protocol["input_sha256"]:
        raise ValueError("Competition input changed")
    fingerprints = {rel(manifest): sha(path(manifest)),
                    "round16_evaluate.py": sha(Path(__file__))}
    for mapping in (protocol["source_code_sha256"],
                    protocol["protected_file_sha256"]):
        for name, digest in mapping.items():
            if sha(path(name)) != digest:
                raise ValueError("Frozen/protected file changed: " + name)
            fingerprints[rel(name)] = digest
    if protocol.get("external_sha256"):
        if sha(ROOT / "data-original" / "data.csv") != protocol["external_sha256"]:
            raise ValueError("Original data changed")
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    if list(sample) != ["id", "satisfaction"] or not sample.id.is_unique or not tr.id.is_unique:
        raise ValueError("Invalid official ID/schema")
    split_file = path(protocol["split_path"])
    if sha(split_file) != protocol["split_sha256"]:
        raise ValueError("Frozen split changed")
    fingerprints[rel(split_file)] = sha(split_file)
    with np.load(split_file, allow_pickle=False) as z:
        dev = check_idx(z["dev_idx"], len(tr), "development")
        hold1 = check_idx(z["hold1_idx"], len(tr), "holdout one")
        hold2 = check_idx(z["hold2_idx"], len(tr), "holdout two")
        hold = np.concatenate([hold1, hold2])
        fold_id = np.asarray(z["dev_fold_id"])
        assert_same(z["train_id"], tr.id.to_numpy(), "split training ID")
        assert_same(z["test_id"], sample.id.to_numpy(), "split test ID")
    if len(dev) + len(hold) != len(tr) or len(np.intersect1d(dev, hold)):
        raise ValueError("Development and holdout must partition all train rows")
    if len(np.unique(hold)) != len(hold):
        raise ValueError("H1 and H2 overlap")
    if abs(len(dev) / len(tr) - .8) > 3 / len(tr) or any(
            abs(len(b) / len(tr) - .1) > 3 / len(tr) for b in (hold1, hold2)):
        raise ValueError("Frozen split must have 80/10/10 row proportions")
    if fold_id.shape != (len(dev),) or not np.issubdtype(fold_id.dtype, np.integer) or set(np.unique(fold_id)) != {0, 1, 2}:
        raise ValueError("Invalid development fold assignment")
    # No holdout label statistic is computed at this stage.
    ydev = tr.satisfaction.iloc[dev].to_numpy(dtype=np.int8)
    if set(np.unique(ydev)) != {0, 1}:
        raise ValueError("Development labels must contain both binary classes")
    base = list(protocol["baseline_members"])
    supplied = protocol["candidate_members"]
    carriers = dict(supplied) if isinstance(supplied, dict) else {n: n for n in supplied}
    if list(carriers) != ["raw", "raw13", "raw30"] or base != ["baseline_current"]:
        raise ValueError("Fixed baseline/carrier list changed")
    if set(base) & set(carriers.values()) or len(set(carriers.values())) != 3:
        raise ValueError("Baseline and three carriers must be separate members")
    if set(protocol["models"]) != set(base) | set(carriers.values()):
        raise ValueError("Model list contains missing or unplanned members")
    pred = {}
    for name, folder in protocol["models"].items():
        if isinstance(folder, dict):
            folder = folder["path"]
        folder = path(folder)
        cfg_file, aggregate = folder / "configuration.json", folder / "summary.npz"
        cfg = load(cfg_file)
        if cfg["competition"] != COMPETITION or cfg["folds"] != 3:
            raise ValueError("Native configuration mismatch: " + name)
        if cfg["input_sha256"] != actual_hashes or cfg["split_sha256"] != protocol["split_sha256"]:
            raise ValueError("Native data/split mismatch: " + name)
        if cfg.get("external_sha256") != protocol.get("external_sha256"):
            raise ValueError("Native external data mismatch: " + name)
        if not cfg["holdout_excluded_from_supervised_training_and_early_stopping"]:
            raise ValueError("Holdout exclusion not declared: " + name)
        for source, digest in cfg["source_code_sha256"].items():
            if sha(path(source)) != digest:
                raise ValueError("Native source changed: " + source)
            fingerprints[rel(source)] = digest
        fingerprints[rel(cfg_file)] = sha(cfg_file)
        fingerprints[rel(aggregate)] = sha(aggregate)
        with np.load(aggregate, allow_pickle=False) as z:
            assert_same(z["dev_idx"], dev, name + " development index")
            assert_same(z["hold_idx"], hold, name + " holdout index")
            assert_same(z["test_id"], sample.id.to_numpy(), name + " test ID")
            po = check_probability(z["dev_oof"], len(dev), name + " development")
            ph = check_probability(z["hold_pred"], len(hold), name + " holdout")
            pt = check_probability(z["test_pred"], len(sample), name + " test")
        coverage = np.zeros(len(dev), dtype=np.int8)
        hold_avg = np.zeros(len(hold))
        test_avg = np.zeros(len(sample))
        for f in range(3):
            native = folder / f"fold-{f + 1}.npz"
            fingerprints[rel(native)] = sha(native)
            val_pos = np.flatnonzero(fold_id == f)
            train_pos = np.flatnonzero(fold_id != f)
            with np.load(native, allow_pickle=False) as z:
                assert_same(z["val_idx"], dev[val_pos], name + " validation fold")
                assert_same(z["train_idx"], dev[train_pos], name + " training fold")
                assert_same(z["hold_idx"], hold, name + " fold holdout index")
                assert_same(z["test_id"], sample.id.to_numpy(), name + " fold test ID")
                vp = check_probability(z["val_pred"], len(val_pos), name + " fold val")
                hp = check_probability(z["hold_pred"], len(hold), name + " fold hold")
                tp = check_probability(z["test_pred"], len(sample), name + " fold test")
            if not np.allclose(vp, po[val_pos], rtol=0, atol=1e-12):
                raise ValueError("Native OOF aggregation mismatch: " + name)
            coverage[val_pos] += 1
            hold_avg += hp / 3
            test_avg += tp / 3
        if not np.all(coverage == 1):
            raise ValueError("Development OOF coverage is not exactly once")
        if not np.allclose(hold_avg, ph, rtol=0, atol=1e-12) or not np.allclose(test_avg, pt, rtol=0, atol=1e-12):
            raise ValueError("Native three-fold averaging mismatch: " + name)
        pred[name] = dict(dev=po, hold=ph, test=pt)
    active = protocol["active_holdout"]
    if active not in ("hold1", "hold2") or protocol["hold_min_delta"] != 1e-5:
        raise ValueError("Predeclared active holdout/submission gate changed")
    active_pos = np.arange(len(hold1)) if active == "hold1" else np.arange(len(hold1), len(hold))
    return protocol, fingerprints, tr, sample, dev, hold, fold_id, ydev, pred, active_pos


def stack(features, y, fold_id):
    cross = np.empty(len(y))
    fold_models = []
    for f in range(3):
        a, b = fold_id != f, fold_id == f
        model = LogisticRegression(**FIXED_STACK).fit(features[a], y[a])
        cross[b] = model.predict_proba(features[b])[:, 1]
        fold_models.append(dict(fold=f + 1, coefficients=model.coef_[0].tolist(),
                                intercept=float(model.intercept_[0]),
                                fitted_rows=int(a.sum()), evaluated_rows=int(b.sum()),
                                iterations=model.n_iter_.tolist()))
    final = LogisticRegression(**FIXED_STACK).fit(features, y)
    return cross, final, fold_models


def matrix(pred, names, key):
    return np.column_stack([logit(np.clip(pred[n][key], 1e-6, 1 - 1e-6)) for n in names])


def run(manifest, stage):
    protocol, fingerprints, tr, sample, dev, hold, fold_id, y, pred, active_pos = read_inputs(manifest)
    analysis = path(protocol["analysis_dir"])
    lock_file, final_file = analysis / "development-selection.json", analysis / "final-comparison.json"
    base_names = list(protocol["baseline_members"])
    supplied = protocol["candidate_members"]
    carriers = dict(supplied) if isinstance(supplied, dict) else {n: n for n in supplied}
    named = {"baseline": base_names, **{k: base_names + [v]
                                      for k, v in carriers.items()}}
    if stage == "dev":
        if lock_file.exists() or final_file.exists():
            raise ValueError("Development selection already recorded; refusing to replace locked coefficients")
        cross = {"baseline": pred["baseline_current"]["dev"]}
        models, fit_info = {}, {}
        for name, members in named.items():
            if name == "baseline":
                continue
            cross[name], models[name], fit_info[name] = stack(matrix(pred, members, "dev"), y, fold_id)
        def score(p):
            return dict(auc=float(roc_auc_score(y, p)),
                        block_auc=[float(roc_auc_score(y[fold_id == f], p[fold_id == f])) for f in range(3)])
        baseline = score(cross["baseline"])
        rows = []
        for name in carriers:
            current = score(cross[name])
            delta = current["auc"] - baseline["auc"]
            blocks = [a - b for a, b in zip(current["block_auc"], baseline["block_auc"])]
            rows.append(dict(candidate=name, **current, delta=delta, block_deltas=blocks,
                             eligible=delta >= protocol["dev_min_delta"] and min(blocks) > 0))
        selected = max((r for r in rows if r["eligible"]), key=lambda r: r["auc"], default=None)
        # The fitted hold/test predictions are generated without looking at hold labels.
        analysis.mkdir(parents=True, exist_ok=True)
        predicted = analysis / "locked-stack-predictions.npz"
        np.savez_compressed(predicted, dev_idx=dev, hold_idx=hold, test_id=sample.id.to_numpy(),
            **{name + "_dev": value for name, value in cross.items()},
            baseline_hold=pred["baseline_current"]["hold"],
            baseline_test=pred["baseline_current"]["test"],
            **{name + "_hold": models[name].predict_proba(matrix(pred, members, "hold"))[:, 1]
               for name, members in named.items() if name != "baseline"},
            **{name + "_test": models[name].predict_proba(matrix(pred, members, "test"))[:, 1]
               for name, members in named.items() if name != "baseline"})
        result = dict(competition=COMPETITION, stage="development_selection", folds=3,
            protocol_path=rel(manifest), source_fingerprints=fingerprints,
            locked_prediction_path=rel(predicted), locked_prediction_sha256=sha(predicted),
            baseline=baseline, candidates=rows, selected_candidate=selected,
            stack_parameters=FIXED_STACK, development_crossfit=fit_info,
            deployment_stack={k: dict(members=named[k], coefficients=m.coef_[0].tolist(),
                intercept=float(m.intercept_[0]), iterations=m.n_iter_.tolist()) for k, m in models.items()},
            counts=dict(train=len(tr), development=len(dev), excluded_holdouts=len(hold),
                        active_holdout=len(active_pos), active_holdout_name=protocol["active_holdout"], test=len(sample),
                        development_oof_coverage_count=1,
                        development_fraction=len(dev) / len(tr)),
            holdout_labels_evaluated=False, validation_note=NOTE)
        save(lock_file, result)
        print(json.dumps({k: result[k] for k in ("stage", "baseline", "candidates", "selected_candidate", "counts")}, indent=2))
        return
    if stage != "finalize":
        raise ValueError("Unknown stage")
    if final_file.exists():
        raise ValueError("Holdout comparison already finalized; refusing a second evaluation")
    lock = load(lock_file)
    if lock["source_fingerprints"] != fingerprints or lock["protocol_path"] != rel(manifest):
        raise ValueError("Development choice or source artifacts changed after locking")
    selected = lock["selected_candidate"]
    if selected is None:
        raise ValueError("No development-eligible candidate; do not inspect holdout")
    predicted = path(lock["locked_prediction_path"])
    if sha(predicted) != lock["locked_prediction_sha256"]:
        raise ValueError("Locked stack prediction changed")
    name = selected["candidate"]
    with np.load(predicted, allow_pickle=False) as z:
        assert_same(z["dev_idx"], dev, "locked development")
        assert_same(z["hold_idx"], hold, "locked holdout")
        assert_same(z["test_id"], sample.id.to_numpy(), "locked test")
        base_h = check_probability(z["baseline_hold"], len(hold), "baseline hold")
        selected_h = check_probability(z[name + "_hold"], len(hold), "selected hold")
        selected_d = check_probability(z[name + "_dev"], len(dev), "selected dev")
        selected_t = check_probability(z[name + "_test"], len(sample), "selected test")
    # Only the protocol's predeclared active holdout is scored; the other half's
    # labels are not used in selection, diagnostics, early stopping, or gates.
    active_idx = hold[active_pos]
    base_h, selected_h = base_h[active_pos], selected_h[active_pos]
    yh = tr.satisfaction.iloc[active_idx].to_numpy(dtype=np.int8)
    if set(np.unique(yh)) != {0, 1}:
        raise ValueError("Holdout must contain both binary classes")
    before = float(roc_auc_score(yh, base_h))
    after = float(roc_auc_score(yh, selected_h))
    delta = after - before
    eligible = delta >= protocol["hold_min_delta"]
    hashes = np.fromiter((int(hashlib.sha256(str(v).encode()).hexdigest()[:16], 16) % 3
                         for v in tr.id.iloc[active_idx]), dtype=np.int8, count=len(active_idx))
    hold_blocks = []
    for f in range(3):
        b = hashes == f
        hold_blocks.append(dict(block=f + 1, rows=int(b.sum()),
            baseline_auc=float(roc_auc_score(yh[b], base_h[b])),
            selected_auc=float(roc_auc_score(yh[b], selected_h[b])),
            delta=float(roc_auc_score(yh[b], selected_h[b]) - roc_auc_score(yh[b], base_h[b]))))
    result = dict(competition=COMPETITION, stage="final_holdout_check", folds=3,
        input_sha256=protocol["input_sha256"], external_sha256=protocol.get("external_sha256"),
        script_sha256=sha(Path(__file__)), source_fingerprints=fingerprints,
        development_selection_sha256=sha(lock_file), locked_prediction_sha256=sha(predicted),
        baseline=lock["baseline"], development_candidates=lock["candidates"],
        selected_development_candidate=selected,
        selected_blend=dict(candidate=name, auc=selected["auc"],
                            auc_role="development_meta_crossfit_only", development_delta=selected["delta"],
                            heldout_auc=after, heldout_delta=delta),
        heldout=dict(baseline_auc=before, selected_auc=after, delta=delta,
                     min_delta=protocol["hold_min_delta"], passed=eligible,
                     paired_fixed_prediction_approximation=paired(yh, base_h, selected_h),
                     hash_blocks_diagnostic_only=hold_blocks),
        counts=lock["counts"], deployment_stack=lock["deployment_stack"],
        eligible_for_submission=eligible, uploaded=False,
        eligible_for_production_transfer=eligible, production_transfer_limitation=TRANSFER_NOTE,
        model_description="Fresh 80-percent development models, fixed logit stack, excluded holdout checked",
        validation_note=NOTE + " Holdout labels are used only to accept/reject the development-selected "
                        "candidate; hash blocks and confidence interval are diagnostic, not selection rules.")
    pd.DataFrame(dict(id=tr.id.iloc[dev].to_numpy(), satisfaction=y,
                      prediction=selected_d)).to_csv(analysis / "development-oof.csv", index=False)
    pd.DataFrame(dict(id=tr.id.iloc[active_idx].to_numpy(), satisfaction=yh,
                      baseline_prediction=base_h, prediction=selected_h)).to_csv(
                          analysis / "holdout-predictions.csv", index=False)
    if eligible:
        out = path(protocol["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        submission = sample.copy()
        submission.satisfaction = selected_t
        submission.to_csv(out / "submission.csv", index=False)
        result["submission_sha256"] = sha(out / "submission.csv")
        result["development_oof_sha256"] = sha(analysis / "development-oof.csv")
        result["holdout_prediction_sha256"] = sha(analysis / "holdout-predictions.csv")
        save(out / "metrics.json", result)
    save(final_file, result)
    print(json.dumps({k: result[k] for k in ("stage", "selected_blend", "heldout", "eligible_for_submission")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("dev", "finalize"), required=True)
    args = parser.parse_args()
    run(args.manifest, args.stage)
