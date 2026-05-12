# Tested Outputs

Project đã được kiểm thử end-to-end trong môi trường hiện tại.

Do môi trường kiểm thử không truy cập Internet được, lệnh train được chạy với:
LAB2_OFFLINE=1 python src/run_training_pipeline.py

Kết quả kiểm thử:
- python src/run_training_pipeline.py: PASS
- python src/check_outputs.py: PASS
- uvicorn src.app:app --host 127.0.0.1 --port 8000: PASS
- python src/test_api.py: PASS
- Notebook 01_data_prep_baseline_deploy_ready.ipynb: PASS khi execute bằng nbclient với LAB2_OFFLINE=1

Dấu hiệu thành công:
- PROJECT CHECK PASSED
- API TEST PASSED
- /health trả model_loaded=true
- /predict trả model_output và decision

Lưu ý:
Khi sinh viên có Internet, script sẽ ưu tiên tải dataset công khai UCI Occupancy Detection từ GitHub mirror của tác giả.
Nếu không có Internet, fallback sample cùng schema được dùng để lớp học vẫn chạy được trọn pipeline.
