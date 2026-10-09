"""Save actual Kaggle scoring/ranking responses, without credentials."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
cli = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
competition = "playground-series-s6e10"
def request(*args):
    result = subprocess.run([str(cli), *args, "--format", "json"],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)
report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
          "submissions": request("competitions", "submissions", competition),
          "competition": request("competitions", "list", "--group", "entered", "--search", competition)}
(root / "kaggle-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
