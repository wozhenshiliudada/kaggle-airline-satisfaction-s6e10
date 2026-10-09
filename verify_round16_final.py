"""Reconcile the completed rounds16/17 experiment, artifacts, and documents.

This final audit never trains, scores labels, selects carriers, or uploads.
It uses the saved independent checkpoint audit and recorded selection/receipt
decisions. Only ID columns are read from CSVs containing training/holdout labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COMPETITION = "playground-series-s6e10"
BASE = "outputs-round13/submission.csv"
BASE_SHA = "b3451e0310cf0892ebd89cab80f400a5c0fe7fe3725ff93965e0499bf9225f93"
BASE_SCORE = Decimal("0.96132")
AUTHORIZED_MUTABLE = {"best-submission.json", "last-submission.json"}
PRIMITIVES = ("compact", "route_auc", "route_tuned", "cat", "plain_nn",
              "lgb", "context", "aux", "tabm", "travel")
HASH_CACHE = {}


def path(value):
    result = Path(value)
    result = (ROOT / result if not result.is_absolute() else result).resolve()
    result.relative_to(ROOT.resolve())
    return result


def rel(value):
    return str(path(value).relative_to(ROOT)).replace("\\", "/")


def sha(value):
    target = path(value)
    stat = target.stat()
    key = (str(target), stat.st_size, stat.st_mtime_ns)
    if key not in HASH_CACHE:
        digest = hashlib.sha256()
        with target.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        HASH_CACHE[key] = digest.hexdigest()
    return HASH_CACHE[key]


def load(value):
    return json.loads(path(value).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, message):
    require(np.array_equal(actual, expected), "Alignment mismatch: " + message)


def verify_hashes(mapping, allow_authorized_changes=False):
    for source, expected in mapping.items():
        if allow_authorized_changes and rel(source) in AUTHORIZED_MUTABLE:
            continue
        require(sha(source) == expected, "Saved audited source changed: " + source)


def complete_snapshot(value):
    result = load(value)
    competition = result["competition"]
    require(len(competition) == 1 and competition[0]["ref"] ==
            f"https://www.kaggle.com/competitions/{COMPETITION}" and
            competition[0]["userHasEntered"], "Wrong official competition/account snapshot")
    refs = [str(row["ref"]) for row in result["submissions"]]
    require(len(refs) == len(set(refs)), "Official snapshot contains duplicate refs")
    for row in result["submissions"]:
        require(row["status"] == "SubmissionStatus.COMPLETE" and row.get("publicScore") not in (None, ""),
                "An official submission is still processing or lacks a public score")
        value = Decimal(str(row["publicScore"]))
        require(value.is_finite() and 0 <= value <= 1, "Invalid official public score")
    return result


def checkpoint_audit(round_number):
    audit_path = f"evidence/round{round_number}-checkpoint-audit.json"
    audit = load(audit_path)
    require(audit["passed"] and audit["round"] == round_number and
            audit["baseline_estimator_fits"] == 33 and audit["baseline_development_folds"] == 3 and
            audit["carrier_fits"] == 9, "Independent saved checkpoint audit is incomplete")
    require(audit["script_sha256"] == sha("audit_round16_checkpoints.py"), "Independent audit source changed")
    require(audit["holdout_labels_loaded"] is False and audit["holdout_metrics_computed"] is False and
            audit["development_metrics_computed"] is False, "Integrity audit evaluated label metrics")
    audited = audit["audited_file_sha256"]
    verify_hashes(audited, allow_authorized_changes=True)
    base = "experiments/round16-locked/baseline_current"
    for f in (1, 2, 3):
        for suffix in ("npz", "json"):
            require(f"{base}/fold-{f}.{suffix}" in audited, "Baseline native fold was not independently audited")
        for component in PRIMITIVES:
            for suffix in ("npz", "json"):
                require(f"{base}/components-fold-{f}/{component}.{suffix}" in audited,
                        "Baseline primitive was not independently audited")
    require(f"{base}/summary.npz" in audited, "Combined baseline summary missing from audit")
    return audit_path, audit


def round_artifacts(round_number, official_rows, train_id, split):
    audit_path, audit = checkpoint_audit(round_number)
    protocol_path = f"experiments/round{round_number}-locked/protocol.json"
    protocol = load(protocol_path)
    require(protocol["competition"] == COMPETITION and protocol["round"] == round_number and
            protocol["epochs"] == (4 if round_number == 16 else 8) and protocol["folds"] == 3,
            "Round used an undeclared training budget/protocol")
    active = "hold1" if round_number == 16 else "hold2"
    require(protocol["active_holdout"] == active, "Wrong holdout was active")
    require(protocol["split_path"] == "experiments/round16-locked/split.npz" and
            protocol["split_sha256"] == sha(protocol["split_path"]), "Prospectively reserved split changed")
    verify_hashes(protocol["source_code_sha256"])
    verify_hashes(protocol["protected_file_sha256"], allow_authorized_changes=True)
    analysis = path(f"analyses/round{round_number}-locked")
    dev_path = analysis / "development-selection.json"
    held_path = analysis / "final-comparison.json"
    dev = load(dev_path)
    require(dev["competition"] == COMPETITION and dev["stage"] == "development_selection" and
            dev["holdout_labels_evaluated"] is False, "Development stage consulted heldout labels")
    require(dev["protocol_path"] == protocol_path, "Development decision belongs to another protocol")
    verify_hashes(dev["source_fingerprints"], allow_authorized_changes=True)
    require(dev["locked_prediction_sha256"] == sha(dev["locked_prediction_path"]), "Frozen prediction packet changed")
    require(dev["counts"]["train"] == len(train_id) and
            dev["counts"]["development"] == len(split["dev_idx"]) and
            dev["counts"]["development_oof_coverage_count"] == 1 and
            dev["counts"]["active_holdout_name"] == active and
            dev["counts"]["active_holdout"] == len(split[active + "_idx"]), "Development/heldout count roles changed")
    selected = dev["selected_candidate"]
    final = None
    output = path(f"outputs-round{round_number}")
    production_path = path(f"analyses/round{round_number}-production-comparison.json")
    archive = path(f"evidence/round{round_number}-submission-receipt.json")
    if selected is None:
        require(not held_path.exists(), "No development-eligible candidate may be scored on holdout")
        require(not production_path.exists(), "Production ran without a development-eligible carrier")
        for name in ("submission.csv", "oof.csv", "metrics.json", "validation.json"):
            require(not (output / name).exists(), "Failed development produced a formal candidate")
        require(not archive.exists(), "Failed development has an upload record")
        return dict(round=round_number, outcome="development_gate_failed", uploaded=False,
                    audit_path=audit_path, audit_sha256=sha(audit_path), source_audited_files=len(audit["audited_file_sha256"]))
    require(selected["eligible"] and selected["delta"] >= 1e-5 and min(selected["block_deltas"]) > 0,
            "Saved development choice did not pass its fixed gate")
    require(selected["candidate"] in ("raw", "raw13", "raw30"), "An unplanned carrier was selected")
    final = load(held_path)
    require(final["selected_development_candidate"] == selected and
            final["development_selection_sha256"] == sha(dev_path) and
            final["locked_prediction_sha256"] == dev["locked_prediction_sha256"] and
            final["source_fingerprints"] == dev["source_fingerprints"] and
            final["deployment_stack"] == dev["deployment_stack"], "Holdout altered the locked development choice")
    require(final["counts"]["active_holdout_name"] == active and
            final["counts"]["active_holdout"] == len(split[active + "_idx"]), "Wrong holdout counts were evaluated")
    held_ids = pd.read_csv(analysis / "holdout-predictions.csv", usecols=["id"]).id.to_numpy()
    same(held_ids, train_id[split[active + "_idx"]], "only predeclared active holdout scored")
    dev_ids = pd.read_csv(analysis / "development-oof.csv", usecols=["id"]).id.to_numpy()
    same(dev_ids, train_id[split["dev_idx"]], "development OOF row coverage")
    require(final["heldout"]["min_delta"] == 1e-5, "Holdout acceptance threshold changed")
    held_passed = bool(final["heldout"]["delta"] >= 1e-5)
    require(final["heldout"]["passed"] == held_passed and
            final["eligible_for_production_transfer"] == held_passed, "Saved holdout gate flags disagree")
    if not held_passed:
        require(not production_path.exists(), "Production ran after a failed holdout")
        for name in ("submission.csv", "oof.csv", "metrics.json", "validation.json"):
            require(not (output / name).exists(), "Failed holdout produced formal output")
        require(not archive.exists(), "Failed holdout has an upload record")
        return dict(round=round_number, outcome="active_holdout_gate_failed", uploaded=False,
                    audit_path=audit_path, audit_sha256=sha(audit_path),
                    development_candidate=selected["candidate"], active_holdout=active,
                    source_audited_files=len(audit["audited_file_sha256"]))
    production = load(production_path)
    require(production["round"] == round_number and production["carrier_mode"] == selected["candidate"] and
            production["frozen_stack"] == dev["deployment_stack"][selected["candidate"]],
            "Production changed the selected carrier or frozen coefficients")
    verify_hashes(production["source_fingerprints"], allow_authorized_changes=True)
    verify_hashes(production["feature_code_sha256"])
    require(production["source_status"] == {"carrier": "completed_five_folds"} and
            production["native_folds"] == 5 and production["historical_native_folds"] == [3, 5, 10],
            "Production coverage/native composition changed")
    gate = production["compatibility_gate"]
    require(gate["min_delta"] == 1e-5 and gate["passed"] == production["eligible_for_submission"],
            "Production compatibility flags disagree")
    require(production["production_candidate_sha256"] == sha(production["production_candidate_path"]) and
            production["production_oof_sha256"] == sha(production["production_oof_path"]),
            "Saved fixed production predictions changed")
    if not gate["passed"]:
        require(production["selected_blend"] is None, "Failed production must not have a formal selected blend")
        for name in ("submission.csv", "oof.csv", "metrics.json", "validation.json"):
            require(not (output / name).exists(), "Failed production acquired formal output")
        require(not archive.exists(), "Failed production has an upload receipt")
        return dict(round=round_number, outcome="historical_compatibility_gate_failed", uploaded=False,
                    audit_path=audit_path, audit_sha256=sha(audit_path),
                    development_candidate=selected["candidate"], active_holdout=active,
                    source_audited_files=len(audit["audited_file_sha256"]))
    selected_full = production["selected_blend"]
    require(selected_full and selected_full["delta"] >= 1e-5 and min(selected_full["fold_deltas"]) > 0 and
            selected_full["all_blocks_positive"] and gate["all_five_blocks_positive"], "Historical production gate did not pass")
    metrics, validation, receipt = load(output / "metrics.json"), load(output / "validation.json"), load(archive)
    require(validation["passed"] and validation["uploaded"] and validation["round"] == round_number and
            validation["exactly_one_new_official_submission"] and validation["official_status"] == "COMPLETE",
            "Formal candidate has not completed its one-time upload reconciliation")
    require(metrics["uploaded"] and metrics["official_status"] == "COMPLETE", "Production metric still says unuploaded")
    require(validation["record_script_sha256"] == sha("record_round16.py"), "Record implementation changed after validation")
    require(validation["metrics_pre_official_sha256"] == sha(production_path),
            "Initial formally validated metrics differ from the frozen production comparison")
    for key, value in production.items():
        if key != "uploaded":
            require(metrics[key] == value, "Official reconciliation altered a frozen production metric: " + key)
    digest = sha(output / "submission.csv")
    require(digest == metrics["submission_sha256"] == validation["submission_sha256"] == receipt["sha256"] ==
            production["production_candidate_sha256"], "Uploaded/final CSV fingerprint mismatch")
    require(sha(output / "oof.csv") == metrics["oof_sha256"] == validation["oof_sha256"] ==
            production["production_oof_sha256"], "Formal OOF changed after validation")
    ref = str(validation["submission_ref"])
    require(ref in official_rows and str(metrics["official_submission_ref"]) == ref and str(receipt["submission_ref"]) == ref,
            "Formal upload reference differs from official account history")
    official = official_rows[ref]
    require(Decimal(str(official["publicScore"])) == Decimal(str(metrics["official_public_score"])) ==
            Decimal(str(validation["public_score"])) == Decimal(str(receipt["public_score"])), "Final public score differs from official snapshot")
    require(validation["new_official_best"] == (Decimal(str(official["publicScore"])) > BASE_SCORE),
            "Strict public-improvement flag differs from the saved official score")
    require(receipt["official_status"] == "COMPLETE", "Receipt is not finalized Complete")
    before_path = f"evidence/round{round_number}-before-upload-official-results.json"
    after_path = f"evidence/round{round_number}-after-submit-official-results.json"
    before, after = complete_snapshot(before_path), complete_snapshot(after_path)
    before_refs = {str(row["ref"]) for row in before["submissions"]}
    after_refs = {str(row["ref"]) for row in after["submissions"]}
    require(after_refs - before_refs == {ref} and before_refs <= after_refs and after_refs <= set(official_rows),
            "Round upload did not add exactly its recorded unique official ref")
    require(validation["before_snapshot_sha256"] == sha(before_path) and validation["after_snapshot_sha256"] == sha(after_path),
            "Archived official before/after snapshot changed")
    return dict(round=round_number, outcome="uploaded_complete", uploaded=True, submission_ref=official["ref"],
                public_score=official["publicScore"], path=rel(output / "submission.csv"), sha256=digest,
                active_holdout=active, development_candidate=selected["candidate"],
                audit_path=audit_path, audit_sha256=sha(audit_path),
                source_audited_files=len(audit["audited_file_sha256"]))


def cycle_audit(value, rounds, uploaded):
    candidates = [path(value)]
    sibling = path("evidence/round17-cycle-state.json")
    if 17 in rounds and sibling.exists() and sibling not in candidates:
        candidates.append(sibling)
    result = []
    represented = set()
    for source in candidates:
        state = load(source)
        require(state["completed"] is True and state["status"] not in
                ("running", "error", "upload_receipt_detected"), "Cycle is incomplete or awaits receipt reconciliation")
        require(state["api_uploads_performed"] == 0, "The cycle itself unexpectedly uploaded")
        require(state["script_sha256"] == sha("round16_cycle.py"), "Cycle source changed")
        verify_hashes(state["child_script_sha256"])
        require(sha(state["cycle_configuration_path"]) == state["cycle_configuration_sha256"],
                "Cycle configuration snapshot changed")
        configuration = load(state["cycle_configuration_path"])
        require(configuration["child_script_sha256"] == state["child_script_sha256"] and
                configuration["script_sha256"] == state["script_sha256"] and configuration["no_api_upload"],
                "Cycle configuration/source state differs")
        reconciled_mutable = {f"outputs-round{r['round']}/{name}"
                              for r in uploaded for name in ("metrics.json", "validation.json")}
        # A preparation-only snapshot can precede root's official reconciliation;
        # those two metadata files are checked against receipts above instead.
        verify_hashes({name: digest for name, digest in state.get("source_artifact_sha256", {}).items()
                       if rel(name) not in reconciled_mutable}, allow_authorized_changes=True)
        for item in state.get("rounds", []):
            represented.add(item["round"])
        if state.get("prepared_upload_round"):
            represented.add(state["prepared_upload_round"])
        if state.get("active_round"):
            represented.add(state["active_round"])
        result.append(dict(path=rel(source), sha256=sha(source), status=state["status"],
                           completed=True, prepared_upload_round=state.get("prepared_upload_round")))
    require(set(rounds) <= represented, "Completed cycles do not account for every audited round")
    return result


def documentation_audit(official_path, official, best):
    doc_path = "evidence/round16-documentation-state.json"
    state = load(doc_path)
    comp = official["competition"][0]
    require(rel(state["official_path"]) == rel(official_path) and state["official_sha256"] == sha(official_path) and
            state["checked_at_utc"] == official["checked_at_utc"], "Documentation used another official snapshot")
    require(state["submissions_count"] == len(official["submissions"]) and
            state["user_rank"] == comp["userRank"] and state["team_count"] == comp["teamCount"],
            "Documentation account count/rank differs from final official snapshot")
    wanted = {key: best[key] for key in ("path", "sha256", "public_score", "submission_ref")}
    require(state["best"] == wanted, "Documentation best pointer differs from protected current best")
    required_docs = {"README.md", "RESULTS.md", "LOCKED_OPTIMIZATION.md"}
    require(set(state["docs"]) == required_docs, "Required latest-result documents were not all updated")
    for name, meta in state["docs"].items():
        require(sha(name) == meta["sha256"], "Document changed after recorded final update: " + name)
        body = path(name).read_text(encoding="utf-8")
        excerpt = meta["status_excerpt"]
        require(excerpt and excerpt in body, "Latest status excerpt not present in document: " + name)
        require(str(best["public_score"]) in excerpt, "Document excerpt omits actual best score: " + name)
        require(re.search(rf"(?<!\d){comp['userRank']}\s*/\s*{comp['teamCount']}(?!\d)", excerpt),
                "Document excerpt omits exact official rank/team count: " + name)
        count = len(official["submissions"])
        require(re.search(rf"(?:累计(?:提交)?|提交(?:总数|次数|数)|共|总提交数)[^\n\d]{{0,12}}{count}(?!\d)", excerpt),
                "Document excerpt omits exact cumulative submission count: " + name)
    return dict(path=doc_path, sha256=sha(doc_path), docs=state["docs"], submissions_count=len(official["submissions"]),
                user_rank=comp["userRank"], team_count=comp["teamCount"])


def main(rounds, official_path, cycle_state):
    require(rounds in ([16], [16, 17]), "Final rounds must be 16 or 16 followed by 17")
    if 17 not in rounds:
        for unreported in ("analyses/round17-locked/development-selection.json",
                           "analyses/round17-locked/final-comparison.json",
                           "analyses/round17-production-comparison.json",
                           "outputs-round17/metrics.json",
                           "evidence/round17-submission-receipt.json"):
            require(not path(unreported).exists(), "Round17 work exists but was omitted from the final audit scope")
    official = complete_snapshot(official_path)
    before = complete_snapshot("evidence/round16-before-training-official-results.json")
    before_refs = {str(row["ref"]) for row in before["submissions"]}
    official_rows = {str(row["ref"]): row for row in official["submissions"]}
    final_refs = set(official_rows)
    require(before_refs <= final_refs and len(final_refs - before_refs) <= 2, "Unexpected new or missing official submissions")
    require(sha(BASE) == BASE_SHA, "Original best CSV was overwritten")
    train_id = pd.read_csv(ROOT / "data" / "train.csv", usecols=["id"]).id.to_numpy()
    split_path = "experiments/round16-locked/split.npz"
    with np.load(path(split_path), allow_pickle=False) as z:
        split = {key: z[key].copy() for key in ("dev_idx", "hold1_idx", "hold2_idx", "dev_fold_id")}
        same(z["train_id"], train_id, "frozen split training identity")
    all_rows = np.concatenate([split["dev_idx"], split["hold1_idx"], split["hold2_idx"]])
    same(np.sort(all_rows), np.arange(len(train_id)), "80/10/10 partition without overlap")
    results = [round_artifacts(number, official_rows, train_id, split) for number in rounds]
    latest_audit = max(datetime.fromisoformat(load(r["audit_path"])["checked_at_utc"]) for r in results)
    require(datetime.fromisoformat(official["checked_at_utc"]) >= latest_audit,
            "Final official query predates the completed training/checkpoint audits")
    uploaded = [r for r in results if r["uploaded"]]
    require({str(r["submission_ref"]) for r in uploaded} == final_refs - before_refs,
            "New official submissions are not exactly the reconciled round uploads")
    if 17 in rounds:
        first = results[0]
        require(first["outcome"] != "uploaded_complete" or Decimal(str(first["public_score"])) <= BASE_SCORE,
                "Fallback ran despite round16 already improving the protected public best")
    best = load("best-submission.json")
    strict_improvements = [r for r in uploaded if Decimal(str(r["public_score"])) > BASE_SCORE]
    chosen = max(strict_improvements, key=lambda r: Decimal(str(r["public_score"])), default=None)
    if chosen is None:
        require(best["path"] == BASE and best["sha256"] == BASE_SHA and Decimal(str(best["public_score"])) == BASE_SCORE,
                "Non-improving candidate replaced the protected official best")
    else:
        for key in ("path", "sha256", "submission_ref", "public_score"):
            require(str(best[key]) == str(chosen[key]), "Best pointer does not identify the genuinely higher scored CSV")
        require(best["previous_best"]["path"] == BASE and best["previous_best"]["sha256"] == BASE_SHA,
                "Improved best lost the protected prior best")
    require(sha(best["path"]) == best["sha256"] and str(best["submission_ref"]) in official_rows and
            Decimal(str(best["public_score"])) == Decimal(str(official_rows[str(best["submission_ref"])]["publicScore"])),
            "Current best pointer/file differs from official history")
    receipt = load("last-submission.json")
    if uploaded:
        last = uploaded[-1]
        require(receipt["sha256"] == last["sha256"] and str(receipt["submission_ref"]) == str(last["submission_ref"]),
                "Live last receipt does not identify the final authorized upload")
    else:
        initial_protocol = load("experiments/round16-locked/protocol.json")
        require(sha("last-submission.json") == initial_protocol["protected_file_sha256"]["last-submission.json"],
                "Live upload receipt changed even though this task made no submission")
    cycles = cycle_audit(cycle_state, rounds, uploaded)
    documents = documentation_audit(official_path, official, best)
    result = dict(passed=True, checked_at_utc=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)), competition=COMPETITION, rounds=results,
        independent_baseline_estimator_fits=33, baseline_native_and_primitive_sha_preserved=True,
        authorized_mutable_files=sorted(AUTHORIZED_MUTABLE),
        holdout_metrics_recomputed=False, coefficients_reselected=False,
        h1_only_round16_scored=True, h2_only_prospective_round17_fallback_scored=True,
        actual_new_submissions=len(uploaded), new_submission_refs=[r["submission_ref"] for r in uploaded],
        official_all_complete=True, official_path=rel(official_path), official_sha256=sha(official_path),
        official_checked_at_utc=official["checked_at_utc"], official_submission_count=len(official["submissions"]),
        official_competition=official["competition"], current_best=best, cycles=cycles, documentation=documents,
        note="Saved source/exclusion audits and decisions reconciled without retraining, label scoring, or reselection. "
             "Locked comparisons validate a freshly retrained structure; production transfer and historical OOF remain adaptive.")
    destination = path("evidence/round16-final-state-validation.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("passed", "actual_new_submissions", "official_submission_count",
                                                  "official_checked_at_utc", "holdout_metrics_recomputed")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", nargs="+", type=int, choices=(16, 17), required=True)
    parser.add_argument("--official", required=True)
    parser.add_argument("--cycle-state", default="evidence/round16-cycle-state.json")
    args = parser.parse_args()
    main(args.rounds, args.official, args.cycle_state)
