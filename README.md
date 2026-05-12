# LAB 2 - AIoT Data Preparation + Baseline Model + Deploy Demo

## 1. Project này dùng để làm gì?

Project này là bài mẫu cho sinh viên chạy trước khi phát triển theo nhóm.

Luồng bài mẫu:

```text
Public dataset / fallback sample
→ kiểm tra schema
→ làm sạch dữ liệu IoT
→ tạo feature dataset
→ chia train/test theo thời gian
→ train Logistic Regression baseline
→ tính anomaly_score bằng Z-score
→ sinh decision_log.csv
→ lưu model .joblib
→ deploy model bằng FastAPI
→ test API /predict
```

Dataset chính: **UCI Occupancy Detection**.  
Khi máy có Internet, script sẽ tải dữ liệu từ GitHub mirror của tác giả. Nếu lớp học không có Internet, project tự dùng file fallback cùng schema để sinh viên vẫn chạy được end-to-end.

## 2. Cấu trúc thư mục

```text
lab2_aiot_public_dataset_deploy/
├── data/
│   ├── DATA_SOURCES.md
│   └── occupancy_fallback_same_schema.csv      # được tạo nếu không tải được public dataset
├── notebooks/
│   └── 01_data_prep_baseline_deploy_ready.ipynb
├── src/
│   ├── data_utils.py                           # hàm tải data, clean, train, decision
│   ├── download_data.py                        # tải public dataset hoặc dùng fallback
│   ├── run_training_pipeline.py                # chạy pipeline không cần notebook
│   ├── app.py                                  # FastAPI deploy model
│   ├── test_api.py                             # test /health, /model-info, /predict
│   └── check_outputs.py                        # kiểm tra đã hoàn thành chưa
├── models/
│   └── occupancy_baseline.joblib               # model sinh ra sau khi chạy notebook
├── outputs/
│   ├── metrics.json
│   ├── decision_log.csv
│   └── figures/
└── requirements.txt
```

## 3. Cài môi trường

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Chạy bài mẫu bằng Jupyter Notebook

```bash
jupyter lab
```

Mở file:

```text
notebooks/01_data_prep_baseline_deploy_ready.ipynb
```

Chọn **Run → Run All Cells**.

Sau khi chạy xong, phải có các file:

```text
data/telemetry_clean.csv
data/feature_dataset.csv
models/occupancy_baseline.joblib
outputs/metrics.json
outputs/decision_log.csv
outputs/figures/01_co2_time_series.png
outputs/figures/02_confusion_matrix.png
outputs/figures/03_occupancy_probability.png
```

## 5. Chạy nhanh không cần notebook

```bash
python src/run_training_pipeline.py
python src/check_outputs.py
```

## 6. Deploy model bằng FastAPI

Mở terminal ở thư mục project, chạy:

```bash
uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
```

Mở trình duyệt:

```text
http://127.0.0.1:8000/docs
```

Test bằng terminal thứ hai:

```bash
python src/test_api.py
```

Kết quả đúng sẽ có dòng:

```text
API TEST PASSED: FastAPI model deployment is working.
```

## 7. Kiểm tra thế nào là hoàn thành?

Sinh viên hoàn thành bài mẫu khi:

1. Notebook chạy hết không lỗi.
2. Có model `models/occupancy_baseline.joblib`.
3. Có `outputs/metrics.json` với accuracy, precision, recall, f1.
4. Có `outputs/decision_log.csv` gồm `occupancy_probability`, `anomaly_score`, `is_anomaly`, `decision`, `command_hint`.
5. Chạy được FastAPI và truy cập được `/docs`.
6. Chạy `python src/test_api.py` thành công.
7. Giải thích được luồng: telemetry → features → model → decision → command hint.

## 8. Lưu ý

- Lab này chỉ deploy local để sinh viên hiểu model deployment cơ bản.
- Lab 5 sẽ phát triển inference service đầy đủ hơn: versioning, logging, validation, monitoring, API contract.
- Không cần lập trình ESP/MQTT ở Lab 2. Telemetry được lấy từ dataset công khai hoặc fallback sample.
