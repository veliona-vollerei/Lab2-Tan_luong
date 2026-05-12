from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import json
import pandas as pd
from src.data_utils import load_model_bundle

REQUIRED_FILES = [
    "data/telemetry_clean.csv",
    "data/feature_dataset.csv",
    "models/occupancy_baseline.joblib",
    "outputs/metrics.json",
    "outputs/decision_log.csv",
    "outputs/figures/01_co2_time_series.png",
    "outputs/figures/02_confusion_matrix.png",
    "outputs/figures/03_occupancy_probability.png",
]

if __name__ == "__main__":
    ok = True
    for f in REQUIRED_FILES:
        path = Path(f)
        exists = path.exists() and path.stat().st_size > 0
        print(("OK  " if exists else "MISS") + f)
        ok = ok and exists

    if not ok:
        raise SystemExit("Some required files are missing. Run the notebook or python src/run_training_pipeline.py first.")

    metrics = json.loads(Path("outputs/metrics.json").read_text(encoding="utf-8"))
    decision_log = pd.read_csv("outputs/decision_log.csv")
    bundle = load_model_bundle()

    required_decision_cols = [
        "timestamp", "occupancy_probability", "predicted_occupancy",
        "anomaly_score", "is_anomaly", "decision", "command_hint", "safety_note",
        "safety_reason"   # <-- cột mở rộng cho mức Giỏi
    ]
    missing_cols = [c for c in required_decision_cols if c not in decision_log.columns]
    print("\nMetrics:", json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Decision log rows:", len(decision_log))
    print("Model version:", bundle.get("model_version"))

    if missing_cols:
        raise SystemExit(f"Decision log missing columns: {missing_cols}")
    if len(decision_log) < 50:
        raise SystemExit("Decision log should have at least 50 rows.")
    if metrics.get("f1", 0) < 0.75:
        raise SystemExit("F1 is lower than expected for the baseline demo. Check preprocessing/model.")

    print("\nPROJECT CHECK PASSED: Notebook outputs and model artifacts are complete.")