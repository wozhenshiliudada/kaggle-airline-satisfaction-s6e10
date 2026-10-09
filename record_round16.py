"""Audit rounds16/17 production outputs and reconcile one official submission.

This module never trains or uploads. Validation distinguishes the excluded
retraining comparison from the adaptive historical full-OOF compatibility gate.
Finalization updates the protected best pointer only for a strictly higher
official public score, and is deliberately a one-time operation per round.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
COMPETITION = "playground-series-s6e10"
BASE = "outputs-round13/submission.csv"
BASE_SHA = "b3451e0310cf0892ebd89cab80f400a5c0fe7fe3725ff93965e0499bf9225f93"
BASE_SCORE = Decimal("0.96132")
CAUTION = ("The development cross-fit covers only the frozen 80 percent development "
           "rows. H1/H2 validate the newly retrained baseline-plus-carrier structure, "
           "with checkpoint selection on development validation. Production transfers "
           "the coefficients unchanged to the historical full-training mixed 3/5/10-fold "
           "baseline and a new five-fold carrier. The historical full-OOF gate is adaptive; "
           "the locked comparison is not an unbiased evaluation of the final production CSV.")


def sha(value):
    digest = hashlib.sha256()
    with Path(value).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path(value):
    value = Path(value)
    if not value.is_absolute():
        value = ROOT / value
    value = value.resolve()
    value.relative_to(ROOT.resolve())
    return value


def relative(value):
    return str(path(value).relative_to(ROOT)).replace("\\", "/")


def load(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def save(value, obj):
    target = path(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def inputs():
    return {name: sha(ROOT / "data" / name) for name in
            ("train.csv", "test.csv", "sample_submission.csv")}


def assert_same(actual, expected, message):
    if not np.array_equal(actual, expected):
        raise ValueError("Alignment mismatch: " + message)


def close(actual, expected, message, tol=1e-12):
    if not np.allclose(actual, expected, rtol=0, atol=tol):
        raise ValueError("Numerical mismatch: " + message)


def probabilities(values, rows, message):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (rows,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid probability vector: " + message)
    return values


def submission(csv, sample):
    if list(csv) != ["id", "satisfaction"] or len(csv) != len(sample):
        raise ValueError("Submission schema changed")
    assert_same(csv.id.to_numpy(), sample.id.to_numpy(), "official submission ID order")
    return probabilities(csv.satisfaction, len(sample), "submission")


def current_best():
    best = load("best-submission.json")
    if (best.get("competition") != COMPETITION or best["path"] != BASE or
            best["sha256"] != BASE_SHA or sha(path(BASE)) != BASE_SHA or
            Decimal(str(best["public_score"])) != BASE_SCORE):
        raise ValueError("Protected official baseline changed")
    return best


def verify_fingerprints(mapping):
    for name, digest in mapping.items():
        if sha(path(name)) != digest:
            raise ValueError("Frozen source changed: " + name)


def locked_audit(round_number, metrics):
    declared = metrics["locked_validation"]
    required_paths = dict(protocol_path=f"experiments/round{round_number}-locked/protocol.json",
                          development_selection_path=f"analyses/round{round_number}-locked/development-selection.json",
                          final_comparison_path=f"analyses/round{round_number}-locked/final-comparison.json")
    for key, expected in required_paths.items():
        if relative(declared[key]) != expected:
            raise ValueError("Locked audit path changed: " + key)
    protocol = load(declared["protocol_path"])
    development = load(declared["development_selection_path"])
    final = load(declared["final_comparison_path"])
    if (protocol["competition"] != COMPETITION or protocol["round"] != round_number or
            protocol["folds"] != 3 or protocol["epochs"] != (4 if round_number == 16 else 8) or
            protocol["dev_min_delta"] != 1e-5 or protocol["hold_min_delta"] != 1e-5 or
            protocol["active_holdout"] != ("hold1" if round_number == 16 else "hold2")):
        raise ValueError("Predeclared locked protocol changed")
    if protocol["input_sha256"] != inputs() or protocol["external_sha256"] != metrics["external_sha256"]:
        raise ValueError("Locked training data changed")
    verify_fingerprints(development["source_fingerprints"])
    if final["source_fingerprints"] != development["source_fingerprints"]:
        raise ValueError("Locked sources changed before holdout evaluation")
    for key in ("development_selection_path", "final_comparison_path", "locked_prediction_path"):
        digest_key = key.removesuffix("_path") + "_sha256"
        if sha(path(declared[key])) != declared[digest_key]:
            raise ValueError("Locked artifact fingerprint mismatch: " + key)
    if sha(path(declared["development_selection_path"])) != final["development_selection_sha256"]:
        raise ValueError("Development lock changed after the holdout check")
    if (declared["locked_prediction_path"] != development["locked_prediction_path"] or
            declared["locked_prediction_sha256"] != development["locked_prediction_sha256"] or
            declared["locked_prediction_sha256"] != final["locked_prediction_sha256"]):
        raise ValueError("Frozen prediction mapping changed")
    selected = development["selected_candidate"]
    if not selected or not selected["eligible"] or selected["delta"] < 1e-5 or min(selected["block_deltas"]) <= 0:
        raise ValueError("Development gate did not pass")
    mode = selected["candidate"]
    if mode not in ("raw", "raw13", "raw30") or final["selected_development_candidate"] != selected:
        raise ValueError("Holdout altered the development input choice")
    if not final["eligible_for_production_transfer"] or not final["heldout"]["passed"] or final["heldout"]["delta"] < 1e-5:
        raise ValueError("Excluded active holdout did not pass")
    if not declared["holdout_passed"] or not declared["eligible_for_production_transfer"]:
        raise ValueError("Production did not preserve the locked holdout gate")
    close(declared["development_delta"], selected["delta"], "copied development gain")
    close(declared["holdout_delta"], final["heldout"]["delta"], "copied holdout gain")
    if declared["active_holdout"] != protocol["active_holdout"]:
        raise ValueError("Wrong active holdout")
    stack = development["deployment_stack"][mode]
    locked = metrics["production_coefficient_lock"]
    if locked["candidate"] != mode or locked["members"] != ["baseline_current", mode]:
        raise ValueError("Wrong two-column stack")
    if locked["members"] != stack["members"] or locked["coefficients"] != stack["coefficients"] or locked["intercept"] != stack["intercept"]:
        raise ValueError("Production refitted or modified locked coefficients")
    if metrics["frozen_stack"] != stack or final["deployment_stack"] != development["deployment_stack"]:
        raise ValueError("Complete fitted development stack changed")
    if locked["clip"] != [1e-6, 1 - 1e-6] or len(locked["coefficients"]) != 2:
        raise ValueError("Fixed logit mapping changed")
    if not np.isfinite(locked["coefficients"] + [locked["intercept"]]).all():
        raise ValueError("Invalid locked coefficients")
    return mode, protocol, development, final, locked


def validate(round_number):
    folder = ROOT / f"outputs-round{round_number}"
    if (folder / "validation.json").exists():
        raise ValueError("This round was already validated; refusing to replace its audit")
    metrics_file = folder / "metrics.json"
    metrics = load(metrics_file)
    best = current_best()
    if metrics["competition"] != COMPETITION or metrics["folds"] != 5 or metrics["native_folds"] != 5:
        raise ValueError("Wrong production competition/folds")
    if (metrics["source_status"] != {"carrier": "completed_five_folds"} or
            metrics["evaluation_blocks"] != 5 or metrics["historical_native_folds"] != [3, 5, 10]):
        raise ValueError("Native/historical validation description changed")
    if metrics.get("round", round_number) != round_number or metrics["input_sha256"] != inputs():
        raise ValueError("Wrong production round/inputs")
    if sha(ROOT / "data-original" / "data.csv") != metrics["external_sha256"]:
        raise ValueError("Production external data changed")
    if metrics["script_sha256"] != sha(ROOT / "round16_production.py"):
        raise ValueError("Production script changed")
    verify_fingerprints(metrics["feature_code_sha256"])
    verify_fingerprints(metrics["source_fingerprints"])
    mode, protocol, development, final, coef = locked_audit(round_number, metrics)
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    y = tr.satisfaction.to_numpy(dtype=np.int8)
    if set(np.unique(y)) != {0, 1} or not tr.id.is_unique or not sample.id.is_unique:
        raise ValueError("Invalid training labels or IDs")
    sub_file, oof_file = folder / "submission.csv", folder / "oof.csv"
    if sha(sub_file) != metrics["submission_sha256"] or sha(oof_file) != metrics["oof_sha256"]:
        raise ValueError("Formal production CSV changed")
    pt = submission(pd.read_csv(sub_file), sample)
    oof = pd.read_csv(oof_file)
    if list(oof) != ["id", "satisfaction", "prediction"]:
        raise ValueError("Formal OOF schema changed")
    assert_same(oof.id.to_numpy(), tr.id.to_numpy(), "formal OOF IDs")
    assert_same(oof.satisfaction.to_numpy(), y, "formal OOF labels")
    p = probabilities(oof.prediction, len(tr), "formal full-training OOF")
    old_oof = pd.read_csv(ROOT / "outputs-round13" / "oof.csv")
    assert_same(old_oof.id.to_numpy(), tr.id.to_numpy(), "protected baseline OOF IDs")
    assert_same(old_oof.satisfaction.to_numpy(), y, "protected baseline OOF labels")
    old = probabilities(old_oof.prediction, len(tr), "protected baseline OOF")
    old_test = submission(pd.read_csv(path(BASE)), sample)
    if (metrics["protected_baseline_oof_sha256"] != sha(ROOT / "outputs-round13" / "oof.csv") or
            metrics["protected_baseline_submission_sha256"] != BASE_SHA):
        raise ValueError("Production used a different protected baseline")
    native_folder = path(metrics["native_carrier_folder"])
    if relative(native_folder) != f"experiments/round{round_number}-carrier-{mode}-5fold":
        raise ValueError("Unexpected production model folder")
    cfg = load(native_folder / "configuration.json")
    if cfg["competition"] != COMPETITION or cfg["folds"] != 5 or cfg["input_sha256"] != inputs():
        raise ValueError("Native carrier configuration mismatch")
    if cfg["external_sha256"] != metrics["external_sha256"] or cfg["script_sha256"] != metrics["script_sha256"]:
        raise ValueError("Native carrier source changed")
    if cfg["epochs"] != protocol["epochs"] or cfg["mode"] != mode:
        raise ValueError("Native carrier input/training budget changed")
    if cfg["production_coefficient_lock"] != coef or cfg["locked_validation"] != metrics["locked_validation"]:
        raise ValueError("Native production coefficient/audit changed")
    if cfg["frozen_stack"] != metrics["frozen_stack"] or not cfg["full_training_coverage"]:
        raise ValueError("Native full-training stack declaration changed")
    verify_fingerprints(cfg["source_code_sha256"])
    native_oof = pd.read_csv(native_folder / "oof.csv")
    native_sub = pd.read_csv(native_folder / "submission.csv")
    assert_same(native_oof.id.to_numpy(), tr.id.to_numpy(), "native OOF IDs")
    assert_same(native_oof.satisfaction.to_numpy(), y, "native OOF labels")
    carrier = probabilities(native_oof.prediction, len(tr), "native carrier OOF")
    carrier_test = submission(native_sub, sample)
    fold_auc, counts, averaged = [], np.zeros(len(tr), dtype=np.int8), np.zeros(len(sample))
    checkpoints = {}
    splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(tr, y))
    for f, (a, b) in enumerate(splits, 1):
        native_file = native_folder / f"fold-{f}.npz"
        digest = sha(native_file)
        if metrics["source_fingerprints"].get(relative(native_file)) != digest:
            raise ValueError("Native five-fold checkpoint was not fingerprinted")
        checkpoints[relative(native_file)] = digest
        detail = load(native_folder / f"fold-{f}.json")
        if detail["checkpoint_sha256"] != digest or detail["fold"] != f or detail["source_audit"] != cfg["source_audit"]:
            raise ValueError("Native checkpoint source/training receipt changed")
        with np.load(native_file, allow_pickle=False) as z:
            assert_same(z["train_idx"], a, "native training positions")
            assert_same(z["val_idx"], b, "native validation positions")
            assert_same(z["val_id"], tr.id.iloc[b].to_numpy(), "native validation IDs")
            assert_same(z["test_id"], sample.id.to_numpy(), "native test IDs")
            if len(z["hold_idx"]) != 0:
                raise ValueError("Production native holdout must be empty after full refit")
            if z["hold_pred"].shape != (0,):
                raise ValueError("Production independent holdout prediction must be empty")
            pv = probabilities(z["val_pred"], len(b), "native fold validation")
            tp = probabilities(z["test_pred"], len(sample), "native fold test")
            close(pv, carrier[b], "native single-coverage OOF")
            averaged += tp / 5
        counts[b] += 1
        fold_auc.append(float(roc_auc_score(y[b], p[b])))
    if not np.all(counts == 1):
        raise ValueError("Native full-training OOF coverage is not exactly once")
    close(averaged, carrier_test, "native five-fold test average")
    def rebuild(base_pred, carrier_pred):
        return expit(coef["intercept"] + coef["coefficients"][0] * logit(np.clip(base_pred, *coef["clip"]))
                     + coef["coefficients"][1] * logit(np.clip(carrier_pred, *coef["clip"])))
    close(rebuild(old, carrier), p, "production full-OOF frozen logit mapping")
    close(rebuild(old_test, carrier_test), pt, "production test frozen logit mapping")
    auc, old_auc = float(roc_auc_score(y, p)), float(roc_auc_score(y, old))
    deltas = [candidate - float(roc_auc_score(y[b], old[b])) for candidate, (_, b) in zip(fold_auc, splits)]
    selected = metrics["selected_blend"]
    if (not selected or not metrics["eligible_for_submission"] or
            selected["candidate"] != mode + "_locked_logit" or
            auc - old_auc < 1e-5 or min(deltas) <= 0 or not selected["all_blocks_positive"]):
        raise ValueError("Predeclared historical compatibility gate failed")
    close(selected["auc"], auc, "formal full-OOF AUC")
    close(selected["delta"], auc - old_auc, "formal full-OOF gain")
    close(selected["fold_deltas"], deltas, "five historical evaluation-block gains")
    gate = metrics["compatibility_gate"]
    if gate["min_delta"] != 1e-5 or not gate["passed"] or not gate["all_five_blocks_positive"]:
        raise ValueError("Compatibility gate declaration changed")
    close(gate["delta"], auc - old_auc, "compatibility gain")
    full = metrics["full_oof"]
    if full["rows"] != len(tr) or full["coverage_count"] != 1:
        raise ValueError("Full production OOF counts changed")
    close(full["baseline_auc"], old_auc, "protected full-OOF baseline AUC")
    close(full["selected_auc"], auc, "full production OOF AUC")
    close(full["block_auc"], fold_auc, "full production block AUC")
    close(full["block_deltas"], deltas, "full production block gains")
    result = dict(passed=True, uploaded=False, round=round_number,
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        record_script_sha256=sha(Path(__file__)), metrics_pre_official_sha256=sha(metrics_file),
        submission_sha256=sha(sub_file), oof_sha256=sha(oof_file), rows=len(sample),
        columns=["id", "satisfaction"], input_sha256=inputs(), external_sha256=metrics["external_sha256"],
        preserved_baseline_sha256=BASE_SHA, best_pointer_pre_upload_sha256=sha(ROOT / "best-submission.json"),
        previous_best_public_score=str(BASE_SCORE), previous_best_submission_ref=best["submission_ref"],
        full_oof_rows=len(tr), full_oof_auc=auc, full_oof_delta=auc - old_auc,
        full_oof_fold_deltas=deltas, full_native_carrier_coverage_count=1,
        development_crossfit_rows=development["counts"]["development"],
        development_crossfit_auc=development["selected_candidate"]["auc"],
        development_delta=development["selected_candidate"]["delta"],
        active_holdout=protocol["active_holdout"], excluded_holdout_auc=final["heldout"]["selected_auc"],
        excluded_holdout_delta=final["heldout"]["delta"],
        coefficients_transferred_unchanged=True, production_coefficient_lock=coef,
        id_alignment=True, probabilities_finite=True,
        probability_min=float(pt.min()), probability_max=float(pt.max()),
        checkpoint_sha256=checkpoints, source_fingerprints=metrics["source_fingerprints"],
        feature_code_sha256=metrics["feature_code_sha256"],
        validation_note=CAUTION)
    save(folder / "validation.json", result)
    print(json.dumps({key: result[key] for key in
        ("passed", "round", "rows", "submission_sha256", "full_oof_auc", "full_oof_delta", "excluded_holdout_delta")}, indent=2))


def official_snapshot(value):
    obj = load(value)
    rows = obj["competition"]
    if len(rows) != 1 or rows[0]["ref"] != f"https://www.kaggle.com/competitions/{COMPETITION}" or not rows[0]["userHasEntered"]:
        raise ValueError("Official snapshot belongs to a different competition/account state")
    refs = [str(row["ref"]) for row in obj["submissions"]]
    if len(refs) != len(set(refs)):
        raise ValueError("Duplicate official submission references")
    return obj


def finalize(round_number):
    folder = ROOT / f"outputs-round{round_number}"
    validation_file, metrics_file = folder / "validation.json", folder / "metrics.json"
    validation, metrics = load(validation_file), load(metrics_file)
    if not validation["passed"] or validation["uploaded"] or validation["round"] != round_number:
        raise ValueError("Round was not validated or has already been finalized")
    if validation["record_script_sha256"] != sha(Path(__file__)) or validation["metrics_pre_official_sha256"] != sha(metrics_file):
        raise ValueError("Validated code/metrics changed before receipt reconciliation")
    if validation["submission_sha256"] != metrics["submission_sha256"] or sha(folder / "submission.csv") != metrics["submission_sha256"]:
        raise ValueError("Validated production CSV changed")
    if sha(folder / "oof.csv") != validation["oof_sha256"]:
        raise ValueError("Validated production full-OOF changed")
    verify_fingerprints(validation["feature_code_sha256"])
    # The live receipt is the one intentionally changed file after upload.
    verify_fingerprints({name: digest for name, digest in validation["source_fingerprints"].items()
                         if relative(name) != "last-submission.json"})
    old = current_best()
    if sha(ROOT / "best-submission.json") != validation["best_pointer_pre_upload_sha256"]:
        raise ValueError("Protected best pointer changed before official reconciliation")
    receipt = load("last-submission.json")
    if receipt["sha256"] != metrics["submission_sha256"]:
        raise ValueError("Uploaded receipt belongs to another file")
    before = official_snapshot(f"evidence/round{round_number}-before-upload-official-results.json")
    after = official_snapshot("kaggle-results.json")
    prior_refs = {str(row["ref"]) for row in before["submissions"]}
    now_refs = {str(row["ref"]) for row in after["submissions"]}
    additions = [row for row in after["submissions"] if str(row["ref"]) not in prior_refs]
    if len(additions) != 1 or not prior_refs.issubset(now_refs) or len(now_refs) != len(prior_refs) + 1:
        raise ValueError("Expected exactly one new official submission, with all previous refs retained")
    new = additions[0]
    if new["status"] != "SubmissionStatus.COMPLETE" or new.get("publicScore") in (None, ""):
        raise ValueError("New official score is not Complete")
    score = Decimal(str(new["publicScore"]))
    if not score.is_finite() or not (Decimal("0") <= score <= Decimal("1")):
        raise ValueError("Invalid official public AUC")
    before_time = datetime.fromisoformat(before["checked_at_utc"])
    after_time = datetime.fromisoformat(after["checked_at_utc"])
    if after_time <= before_time:
        raise ValueError("Official after snapshot predates the before snapshot")
    previous_best_rows = [row for row in after["submissions"] if str(row["ref"]) == str(old["submission_ref"])]
    if len(previous_best_rows) != 1 or Decimal(str(previous_best_rows[0]["publicScore"])) != BASE_SCORE:
        raise ValueError("Official protected-best row changed or disappeared")
    if receipt.get("submission_ref") not in (None, new["ref"]):
        raise ValueError("Receipt reference differs from the unique new official submission")
    improved = score > Decimal(str(old["public_score"]))
    metrics.update(uploaded=True, official_status="COMPLETE", official_submission_ref=new["ref"],
        official_public_score=float(score), official_checked_at_utc=after["checked_at_utc"])
    receipt.update(submission_ref=new["ref"], public_score=new["publicScore"], official_status="COMPLETE",
        official_checked_at_utc=after["checked_at_utc"], official_submission=new)
    validation.update(uploaded=True, submission_ref=new["ref"], official_status="COMPLETE",
        public_score=new["publicScore"], new_official_best=improved,
        official_checked_at_utc=after["checked_at_utc"],
        exactly_one_new_official_submission=True,
        before_snapshot_sha256=sha(path(f"evidence/round{round_number}-before-upload-official-results.json")),
        after_snapshot_sha256=sha(ROOT / "kaggle-results.json"))
    # Every check is complete before any receipt, metric, or best-pointer mutation.
    save(metrics_file, metrics)
    save("last-submission.json", receipt)
    save(f"evidence/round{round_number}-submission-receipt.json", receipt)
    shutil.copy2(ROOT / "kaggle-results.json", ROOT / f"evidence/round{round_number}-after-submit-official-results.json")
    if improved:
        save("best-submission.json", dict(competition=COMPETITION,
            path=f"outputs-round{round_number}/submission.csv", sha256=metrics["submission_sha256"],
            submission_ref=new["ref"], public_score=new["publicScore"],
            official_checked_at_utc=after["checked_at_utc"], previous_best=old))
    save(validation_file, validation)
    print(json.dumps(dict(round=round_number, submission_ref=new["ref"], public_score=new["publicScore"],
        new_official_best=improved, previous_best_public_score=old["public_score"],
        checked_at_utc=after["checked_at_utc"], official_competition=after["competition"]), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "finalize"))
    parser.add_argument("--round", type=int, choices=(16, 17), required=True)
    args = parser.parse_args()
    validate(args.round) if args.action == "validate" else finalize(args.round)
