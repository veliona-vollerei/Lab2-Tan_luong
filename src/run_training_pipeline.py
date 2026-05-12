from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from src.data_utils import (
    API_FEATURES,
    OUTPUTS_DIR,
    FIGURES_DIR,
    check_schema,
    clean_iot_data,
    create_features,
    ensure_dataset,
    evaluate_model,
    make_decision_log,
    save_artifacts,
    time_train_test_split,
    train_baseline_model,
    compute_train_stats,
)

if __name__ == "__main__":
    raw_df, dataset_status = ensure_dataset(prefer_public=True)
    schema_report = check_schema(raw_df)
    clean_df, cleaning_report = clean_iot_data(raw_df)
    feature_df = create_features(clean_df)
    train_df, test_df = time_train_test_split(feature_df, test_ratio=0.25)

    model = train_baseline_model(train_df)
    metrics = evaluate_model(model, test_df)
    train_stats = compute_train_stats(train_df)

    decision_log = make_decision_log(model, test_df, train_stats, n_rows=200)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv("data/telemetry_clean.csv", index=False)
    feature_df.to_csv("data/feature_dataset.csv", index=False)
    decision_log.to_csv(OUTPUTS_DIR / "decision_log.csv", index=False)

    save_artifacts(model, API_FEATURES, train_stats, metrics, dataset_status)

    # Plots
    plt.figure(figsize=(10, 4))
    feature_df.tail(600).plot(x="timestamp", y="CO2", legend=False, ax=plt.gca())
    plt.title("CO2 theo thời gian")
    plt.xlabel("timestamp")
    plt.ylabel("CO2 ppm")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_co2_time_series.png", dpi=160)
    plt.close()

    disp = ConfusionMatrixDisplay.from_predictions(test_df["Occupancy"], model.predict(test_df[API_FEATURES]))
    disp.ax_.set_title("Confusion matrix - Occupancy baseline")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_confusion_matrix.png", dpi=160)
    plt.close()

    prob = model.predict_proba(test_df[API_FEATURES])[:, 1]
    plot_df = test_df.copy().tail(300)
    plot_df["occupancy_probability"] = prob[-300:]
    plt.figure(figsize=(10, 4))
    plot_df.plot(x="timestamp", y="occupancy_probability", legend=False, ax=plt.gca())
    plt.title("Occupancy probability trên tập test")
    plt.xlabel("timestamp")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_occupancy_probability.png", dpi=160)
    plt.close()

    print("DONE: training pipeline completed.")
    print("Schema report:", json.dumps(schema_report, ensure_ascii=False))
    print("Cleaning report:", json.dumps(cleaning_report, ensure_ascii=False))
    print("Metrics:", json.dumps(metrics, ensure_ascii=False))
    print("Generated:")
    print("- data/telemetry_clean.csv")
    print("- data/feature_dataset.csv")
    print("- models/occupancy_baseline.joblib")
    print("- outputs/metrics.json")
    print("- outputs/decision_log.csv")
    print("- outputs/figures/*.png")
