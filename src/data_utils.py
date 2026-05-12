from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

UCI_REPO_RAW = "https://raw.githubusercontent.com/LuisM78/Occupancy-detection-data/refs/heads/master"
PUBLIC_FILES = {
    "datatraining.txt": f"{UCI_REPO_RAW}/datatraining.txt",
    "datatest.txt": f"{UCI_REPO_RAW}/datatest.txt",
    "datatest2.txt": f"{UCI_REPO_RAW}/datatest2.txt",
}

RAW_FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
API_FEATURES = RAW_FEATURES + ["hour", "dayofweek"]
TARGET_COL = "Occupancy"


def make_dirs() -> None:
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def generate_offline_fallback_dataset(n_rows: int = 2400, seed: int = 42) -> Path:
    """Generate a deterministic fallback dataset with the same schema as UCI Occupancy Detection.
    This is not a replacement for the public dataset. It is only for classrooms without internet.
    """
    make_dirs()
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2015-02-04 17:51:00", periods=n_rows, freq="min")

    hours = timestamps.hour + timestamps.minute / 60.0
    is_work_time = ((timestamps.hour >= 8) & (timestamps.hour <= 18)).astype(int)
    # Occupancy pattern: more likely during working hours, with short meetings.
    prob_occ = 0.08 + 0.72 * is_work_time
    meeting_wave = 0.12 * (np.sin(np.arange(n_rows) / 70.0) > 0.65)
    prob_occ = np.clip(prob_occ + meeting_wave, 0.02, 0.95)
    occupancy = rng.binomial(1, prob_occ)

    temperature = 20.2 + 2.2 * occupancy + 0.7 * np.sin(np.arange(n_rows) / 180.0) + rng.normal(0, 0.35, n_rows)
    humidity = 26.5 + 2.0 * occupancy + 0.9 * np.cos(np.arange(n_rows) / 250.0) + rng.normal(0, 0.55, n_rows)
    light = 25 + 430 * occupancy + 70 * is_work_time + rng.normal(0, 35, n_rows)
    light = np.clip(light, 0, None)
    co2 = 470 + 360 * occupancy + 45 * is_work_time + rng.normal(0, 35, n_rows)
    co2 = np.clip(co2, 380, None)

    # Simplified humidity ratio approximation for the lab demo.
    humidity_ratio = (humidity / 100.0) * 0.006 + (temperature - 20) * 0.00005

    df = pd.DataFrame({
        "date": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
        "Temperature": np.round(temperature, 2),
        "Humidity": np.round(humidity, 3),
        "Light": np.round(light, 2),
        "CO2": np.round(co2, 2),
        "HumidityRatio": np.round(humidity_ratio, 6),
        "Occupancy": occupancy.astype(int),
    })

    # Inject a few data quality problems so the notebook has something to fix.
    bad_idx = rng.choice(df.index[100:-100], size=18, replace=False)
    df.loc[bad_idx[:6], "CO2"] = np.nan
    df.loc[bad_idx[6:10], "Light"] = 6000          # outlier
    df.loc[bad_idx[10:14], "Temperature"] = 80     # outlier
    df.loc[bad_idx[14:], "Humidity"] = 130         # outlier

    # Add duplicates to demonstrate duplicate removal.
    dup = df.iloc[[30, 31, 500, 501]].copy()
    df = pd.concat([df, dup], ignore_index=True)

    out = DATA_DIR / "occupancy_fallback_same_schema.csv"
    df.to_csv(out, index=False)
    return out


def download_public_dataset(timeout: int = 20) -> Tuple[bool, List[str]]:
    """Download the real public Occupancy Detection files from the authors' GitHub mirror.
    Returns (success, messages). If internet is unavailable, return False and do not crash.
    """
    make_dirs()
    raw_dir = DATA_DIR / "public_occupancy_detection"
    raw_dir.mkdir(parents=True, exist_ok=True)
    messages = []

    success_count = 0
    for filename, url in PUBLIC_FILES.items():
        dest = raw_dir / filename
        if dest.exists() and dest.stat().st_size > 1000:
            messages.append(f"OK: {filename} already exists.")
            success_count += 1
            continue
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            text = response.text
            if "date" not in text.splitlines()[0] or len(text) < 1000:
                raise ValueError("Downloaded content does not look like the expected CSV text file.")
            dest.write_text(text, encoding="utf-8")
            messages.append(f"Downloaded: {filename} -> {dest}")
            success_count += 1
        except Exception as exc:
            messages.append(f"WARNING: Could not download {filename}: {exc}")

    return success_count == len(PUBLIC_FILES), messages


def ensure_dataset(prefer_public: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """Load public dataset if available. If not, fallback to deterministic offline sample.
    Output dataframe always has the same core columns as UCI Occupancy Detection.
    """
    make_dirs()
    status = {
        "dataset_source": None,
        "messages": [],
        "public_dataset_links": PUBLIC_FILES,
    }

    if prefer_public and os.getenv("LAB2_OFFLINE", "0") != "1":
        ok, messages = download_public_dataset(timeout=5)
        status["messages"].extend(messages)
        raw_dir = DATA_DIR / "public_occupancy_detection"
        paths = [raw_dir / name for name in PUBLIC_FILES]
        if ok and all(p.exists() for p in paths):
            frames = []
            for split_name, path in zip(["train", "test1", "test2"], paths):
                part = pd.read_csv(path)
                part["source_split"] = split_name
                frames.append(part)
            df = pd.concat(frames, ignore_index=True)
            status["dataset_source"] = "UCI Occupancy Detection public dataset via LuisM78 GitHub mirror"
            return df, status

    if prefer_public and os.getenv("LAB2_OFFLINE", "0") == "1":
        status["messages"].append("LAB2_OFFLINE=1, skip public download for fast local test/classroom offline mode.")

    fallback_path = DATA_DIR / "occupancy_fallback_same_schema.csv"
    if not fallback_path.exists():
        fallback_path = generate_offline_fallback_dataset()
    df = pd.read_csv(fallback_path)
    df["source_split"] = "offline_fallback"
    status["dataset_source"] = "offline fallback sample with the same schema as UCI Occupancy Detection"
    status["messages"].append(
        "Using offline fallback sample. When students have internet, rerun download_data.py to use the public dataset."
    )
    return df, status


def check_schema(df: pd.DataFrame) -> Dict:
    required_cols = ["date"] + RAW_FEATURES + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    duplicated_rows = int(df.duplicated().sum())
    result = {
        "required_columns": required_cols,
        "missing_columns": missing,
        "duplicated_rows": duplicated_rows,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
    }
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return result


def clean_iot_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    before_rows = len(df)

    # Parse timestamp.
    df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    bad_timestamp = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    # Remove duplicate rows and duplicate timestamps.
    duplicate_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    duplicate_timestamps = int(df.duplicated(subset=["timestamp"]).sum())
    df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # Convert numeric columns.
    for col in RAW_FEATURES + [TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Outlier handling: set impossible sensor values to NaN first.
    rules = {
        "Temperature": (-10, 60),
        "Humidity": (0, 100),
        "Light": (0, 3000),
        "CO2": (250, 2500),
        "HumidityRatio": (0, 0.05),
    }
    outlier_counts = {}
    for col, (lo, hi) in rules.items():
        mask = (df[col] < lo) | (df[col] > hi)
        outlier_counts[col] = int(mask.sum())
        df.loc[mask, col] = np.nan

    missing_before_fill = {col: int(df[col].isna().sum()) for col in RAW_FEATURES}

    # Sort by time and fill missing sensor values with time-aware interpolation.
    df = df.sort_values("timestamp").reset_index(drop=True)
    df[RAW_FEATURES] = df[RAW_FEATURES].interpolate(method="linear", limit_direction="both")
    df[RAW_FEATURES] = df[RAW_FEATURES].ffill().bfill()

    # Target must be binary.
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)
    df[TARGET_COL] = df[TARGET_COL].clip(0, 1)

    after_rows = len(df)
    report = {
        "before_rows": int(before_rows),
        "after_rows": int(after_rows),
        "removed_rows": int(before_rows - after_rows),
        "bad_timestamp_rows": bad_timestamp,
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamps": duplicate_timestamps,
        "outlier_counts": outlier_counts,
        "missing_before_fill": missing_before_fill,
    }
    return df, report


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(int)

    # Features for explanation/decision log only, not required by API model.
    df["co2_rolling_mean_10"] = df["CO2"].rolling(window=10, min_periods=1).mean()
    df["light_rolling_mean_10"] = df["Light"].rolling(window=10, min_periods=1).mean()
    df["co2_delta"] = df["CO2"].diff().fillna(0)
    return df


def time_train_test_split(df: pd.DataFrame, test_ratio: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def compute_train_stats(train_df: pd.DataFrame) -> Dict:
    stats = {}
    for col in RAW_FEATURES:
        std = float(train_df[col].std())
        if std == 0 or math.isnan(std):
            std = 1.0
        stats[col] = {"mean": float(train_df[col].mean()), "std": std}
    return stats


def compute_anomaly_score(row_or_df, stats: Dict) -> np.ndarray:
    df = pd.DataFrame(row_or_df)
    scores = []
    for col in RAW_FEATURES:
        mean = stats[col]["mean"]
        std = stats[col]["std"] or 1.0
        scores.append(((df[col].astype(float) - mean).abs() / std).to_numpy())
    if not scores:
        return np.zeros(len(df))
    return np.nanmax(np.vstack(scores), axis=0)


def train_baseline_model(train_df: pd.DataFrame) -> Pipeline:
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(train_df[API_FEATURES], train_df[TARGET_COL])
    return model


def evaluate_model(model: Pipeline, test_df: pd.DataFrame) -> Dict:
    y_true = test_df[TARGET_COL].astype(int)
    y_pred = model.predict(test_df[API_FEATURES])
    y_prob = model.predict_proba(test_df[API_FEATURES])[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = None
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics


# ==================== THAY THẾ HÀM decision_from_outputs ====================

def decision_from_outputs(occupancy_probability: float, anomaly_score: float, co2: float, light: float) -> Dict:
    is_anomaly = bool(anomaly_score >= 3.0)
    predicted_occupancy = int(occupancy_probability >= 0.5)

    safety_reason = ""
    # Safety rule: chặn tự động nếu anomaly hoặc confidence không đủ cao
    if is_anomaly or (0.4 < occupancy_probability < 0.6):
        decision = "SAFETY_HOLD"
        command_hint = "NO_AUTO_CONTROL"
        safety_note = "Dữ liệu có dấu hiệu bất thường hoặc độ tin cậy thấp; không gửi lệnh điều khiển tự động."
        if is_anomaly:
            safety_reason = f"Anomaly score cao ({anomaly_score:.2f})"
        else:
            safety_reason = f"Occupancy probability không chắc chắn ({occupancy_probability:.2f})"
    elif predicted_occupancy == 1 and co2 >= 1000:
        decision = "ALERT_VENTILATION_AND_TURN_FAN_ON"
        command_hint = "fan_state=ON"
        safety_note = "Cho phép khuyến nghị bật quạt/thông gió; cần giới hạn theo rule an toàn."
    elif predicted_occupancy == 1 and light < 100:
        decision = "ROOM_OCCUPIED_LIGHTING_NEEDED"
        command_hint = "lighting_state=ON"
        safety_note = "Có người và ánh sáng thấp; đề xuất bật đèn nếu lịch học phù hợp."
    elif predicted_occupancy == 1:
        decision = "ROOM_OCCUPIED_KEEP_COMFORT_MODE"
        command_hint = "ac_state=COMFORT"
        safety_note = "Duy trì tiện nghi, không cần điều khiển mạnh."
    else:
        decision = "ROOM_EMPTY_SAVE_ENERGY"
        command_hint = "ac_state=ECO; fan_state=OFF"
        safety_note = "Phòng có khả năng trống; chỉ chuyển chế độ tiết kiệm nếu không có lịch học."

    return {
        "predicted_occupancy": predicted_occupancy,
        "is_anomaly": is_anomaly,
        "decision": decision,
        "command_hint": command_hint,
        "safety_note": safety_note,
        "safety_reason": safety_reason,
    }


# ==================== THAY THẾ HÀM make_decision_log ====================

def make_decision_log(model: Pipeline, test_df: pd.DataFrame, train_stats: Dict, n_rows: int = 200) -> pd.DataFrame:
    sample = test_df.copy().tail(n_rows).reset_index(drop=True)
    prob = model.predict_proba(sample[API_FEATURES])[:, 1]
    anomaly_score = compute_anomaly_score(sample[RAW_FEATURES], train_stats)

    rows = []
    for i, r in sample.iterrows():
        d = decision_from_outputs(
            occupancy_probability=float(prob[i]),
            anomaly_score=float(anomaly_score[i]),
            co2=float(r["CO2"]),
            light=float(r["Light"]),
        )
        rows.append({
            "timestamp": r["timestamp"],
            "Temperature": r["Temperature"],
            "Humidity": r["Humidity"],
            "Light": r["Light"],
            "CO2": r["CO2"],
            "HumidityRatio": r["HumidityRatio"],
            "occupancy_probability": round(float(prob[i]), 4),
            "actual_occupancy": int(r[TARGET_COL]),
            "predicted_occupancy": d["predicted_occupancy"],
            "anomaly_score": round(float(anomaly_score[i]), 4),
            "is_anomaly": d["is_anomaly"],
            "decision": d["decision"],
            "command_hint": d["command_hint"],
            "safety_note": d["safety_note"],
            "safety_reason": d["safety_reason"],   # <--- THÊM DÒNG NÀY
        })
    return pd.DataFrame(rows)


def save_artifacts(model: Pipeline, feature_cols: List[str], train_stats: Dict, metrics: Dict, dataset_status: Dict) -> None:
    make_dirs()
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "raw_features": RAW_FEATURES,
        "train_stats": train_stats,
        "metrics": metrics,
        "dataset_status": dataset_status,
        "model_name": "occupancy_logistic_regression_baseline",
        "model_version": "lab2-v1",
    }
    joblib.dump(bundle, MODELS_DIR / "occupancy_baseline.joblib")
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUTS_DIR / "dataset_status.json").write_text(json.dumps(dataset_status, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_bundle(model_path: Path | None = None) -> Dict:
    if model_path is None:
        model_path = MODELS_DIR / "occupancy_baseline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run the notebook first or run: python src/run_training_pipeline.py"
        )
    return joblib.load(model_path)
