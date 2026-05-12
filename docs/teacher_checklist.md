# Checklist - Lab 2

## Kiểm thử nhanh trước khi lên lớp

1. Giải nén project.
2. Mở terminal tại thư mục project.
3. Cài môi trường:
   ```bash
   python -m venv .venv
   # Windows: .\.venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Chạy notebook hoặc chạy nhanh:
   ```bash
   python src/run_training_pipeline.py
   python src/check_outputs.py
   ```
5. Deploy API:
   ```bash
   uvicorn src.app:app --reload --host 127.0.0.1 --port 8000
   ```
6. Terminal khác:
   ```bash
   python src/test_api.py
   ```
7. Hoàn thành khi thấy:
   ```text
   PROJECT CHECK PASSED
   API TEST PASSED
   ```

## Lưu ý
- Nếu lớp học không có Internet, project vẫn chạy bằng fallback sample cùng schema.
- Nếu muốn ép chạy offline để test nhanh:
  ```bash
  LAB2_OFFLINE=1 python src/run_training_pipeline.py
  ```
- Trên máy sinh viên bình thường, không cần set `LAB2_OFFLINE`.
