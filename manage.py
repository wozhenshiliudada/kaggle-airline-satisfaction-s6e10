"""Use the official Kaggle CLI for this competition; never embeds credentials."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPETITION = "playground-series-s6e10"
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"
STATE = ROOT / "run-state.json"
FILES = ("train.csv", "test.csv", "sample_submission.csv")


def cli(*args, capture=False):
    executable = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    if not executable.exists():
        raise RuntimeError("Kaggle CLI is missing from the current Python environment.")
    command = [str(executable), *map(str, args)]
    if capture:
        # The official CLI may print Unicode upload progress. Force the child
        # output and parent decoder to agree on Windows instead of locale GBK.
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                 encoding='utf-8', errors='replace',
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        text = process.stdout + process.stderr
        text = re.sub(r"KG(?:AT|RT)_[A-Za-z0-9_-]+", "[REDACTED]", text)
        print(text, end="", flush=True)
        if process.returncode:
            raise RuntimeError("Kaggle request failed. Check authentication and accepted competition rules.")
        return text
    subprocess.run(command, cwd=ROOT, check=True)


def doctor():
    print("Project:", ROOT)
    for package in ("kaggle", "pandas", "numpy", "scikit-learn", "catboost", "lightgbm"):
        print(package, importlib.metadata.version(package))
    config = Path(os.environ.get("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle")))
    # Report credential existence, never credential contents.
    print("Environment token:", bool(os.environ.get("KAGGLE_API_TOKEN")))
    print("Token file:", (config / "access_token").is_file())
    print("Legacy credential file:", (config / "kaggle.json").is_file())
    print("OAuth login may be stored separately by the official CLI.")
    cli("--version")


def download():
    DATA.mkdir(exist_ok=True)
    for filename in FILES:
        if not (DATA / filename).is_file():
            cli("competitions", "download", COMPETITION, "-f", filename,
                "-p", DATA)
            if not (DATA / filename).is_file():
                for archive in DATA.glob("*.zip"):
                    with zipfile.ZipFile(archive) as bundle:
                        # Extract only the exact expected CSV, never arbitrary archive paths.
                        candidates = [name for name in bundle.namelist()
                                      if Path(name).name == filename]
                        if len(candidates) == 1:
                            (DATA / filename).write_bytes(bundle.read(candidates[0]))
    missing = [filename for filename in FILES if not (DATA / filename).is_file()]
    if missing:
        raise RuntimeError(f"Missing downloaded files: {missing}")
    manifest = {"competition": COMPETITION,
                "files": {filename: hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
                          for filename in FILES}}
    (DATA / "download-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def state():
    if not STATE.exists():
        raise RuntimeError("No cloud run has been launched from this project.")
    return json.loads(STATE.read_text(encoding="utf-8"))


def cloud(username):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", username):
        raise ValueError("Invalid Kaggle username.")
    kernel = f"{username}/airline-satisfaction-s6e10-cv"
    folder = ROOT / "cloud"
    folder.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "train.py", folder / "train.py")
    metadata = {"id": kernel, "title": "Airline Satisfaction S6E10 CV",
                "code_file": "train.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": False, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [COMPETITION],
                "kernel_sources": []}
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cli("kernels", "push", "-p", folder, capture=True)
    STATE.write_text(json.dumps({"competition": COMPETITION, "kernel": kernel,
                                 "code_sha256": hashlib.sha256((folder / "train.py").read_bytes()).hexdigest()},
                                indent=2), encoding="utf-8")


def collect():
    kernel = state()["kernel"]
    status = cli("kernels", "status", kernel, capture=True)
    if not re.search(r"\bcomplete\b", status, re.I):
        raise RuntimeError("Cloud execution has not completed successfully; do not submit old outputs.")
    OUTPUT.mkdir(exist_ok=True)
    cli("kernels", "output", kernel, "-p", OUTPUT, "--force",
        "--file-pattern", r"(^|/)(submission\.csv|metrics\.json|feature_importance\.csv|oof\.csv|test_predictions\.csv)$")


def submit(output_dir=OUTPUT):
    import pandas as pd
    from train import validate_submission
    if not all((DATA / filename).is_file() for filename in FILES):
        download()
    path = output_dir / "submission.csv"
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.is_file():
        raise ValueError("Validated model metrics are required before submission.")
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("competition") != COMPETITION:
        raise ValueError("Model output belongs to a different competition.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["submission_sha256"]:
        raise ValueError("Submission changed after model validation.")
    expected_hashes = {filename: hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
                       for filename in FILES}
    if report.get("input_sha256") != expected_hashes:
        raise ValueError("Model output was not trained on these competition files.")
    if report.get("external_sha256"):
        original = ROOT / "data-original" / "data.csv"
        if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest() != report["external_sha256"]:
            raise ValueError("External data differs from validated model output.")
    validate_submission(pd.read_csv(path), pd.read_csv(DATA / "sample_submission.csv"))
    digest = report["submission_sha256"]
    receipt_path = ROOT / "last-submission.json"
    if receipt_path.exists() and json.loads(receipt_path.read_text(encoding='utf-8'))["sha256"] == digest:
        print("This exact file was already submitted; skipping duplicate upload.")
        return
    team_request_path = ROOT / 'team-request.json'
    team_request = None
    if team_request_path.exists():
        team_request = json.loads(team_request_path.read_text(encoding='utf-8'))
        if team_request.get('competition') != COMPETITION:
            raise ValueError('Pending team request belongs to a different competition.')
        exception = team_request.get('before_team_submission', {})
        exact_file_authorized = (exception.get('status') == 'authorized'
                                 and exception.get('sha256') == digest)
        if (team_request.get('require_membership_before_submission', True)
                and team_request.get('membership_status') not in ('joined', 'cancelled')
                and not exact_file_authorized):
            raise ValueError('Next submission is waiting for teammate '
                             + team_request['requested_username']
                             + ' to join the competition team, as requested by the user.')
    auc = report["selected_blend"]["auc"]
    label = report.get("model_description", "CatBoost LightGBM")
    receipt = cli("competitions", "submit", COMPETITION, "-f", path,
                  "-m", f"{label} {report['folds']}-fold CV AUC {auc:.6f}", capture=True)
    receipt_path.write_text(json.dumps({"sha256": digest, "receipt": receipt}, indent=2), encoding="utf-8")
    if team_request is not None:
        exception = team_request.get('before_team_submission', {})
        if exception.get('status') == 'authorized' and exception.get('sha256') == digest:
            exception['status'] = 'uploaded'
            team_request_path.write_text(json.dumps(team_request, indent=2, ensure_ascii=False), encoding='utf-8')
    print("Submission uploaded. Run 'manage.py scores' for the official scoring result.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "login", "download", "status", "collect", "scores", "rank"):
        commands.add_parser(name)
    submission = commands.add_parser("submit")
    submission.add_argument("--output", type=Path, default=OUTPUT)
    launcher = commands.add_parser("cloud")
    launcher.add_argument("--username", required=True)
    local = commands.add_parser("local")
    local.add_argument("--folds", type=int, default=3)
    local.add_argument("--iterations", type=int, default=1800)
    local.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 4))
    local.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.command == "doctor":
        doctor()
    elif args.command == "login":
        cli("auth", "login")
    elif args.command == "download":
        download()
    elif args.command == "cloud":
        cloud(args.username)
    elif args.command == "status":
        cli("kernels", "status", state()["kernel"])
    elif args.command == "collect":
        collect()
    elif args.command == "submit":
        submit(args.output)
    elif args.command == "scores":
        cli("competitions", "submissions", COMPETITION)
    elif args.command == "rank":
        cli("competitions", "list", "--group", "entered", "--search", COMPETITION)
    elif args.command == "local":
        from train import run
        download()
        run(DATA, OUTPUT, folds=args.folds, iterations=args.iterations, threads=args.threads)
        if args.submit:
            submit()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        sys.exit(1)
