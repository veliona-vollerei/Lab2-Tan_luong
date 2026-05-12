import json
import requests

BASE_URL = "http://127.0.0.1:8000"

sample_payload = {
    "room_id": "room_101",
    "device_id": "env_node_01",
    "timestamp": "2015-02-06 06:32:00",
    "Temperature": 21.93,
    "Humidity": 28.786,
    "Light": 469.53,
    "CO2": 834.85,
    "HumidityRatio": 0.001824
}

batch_payload = {
    "payloads": [
        sample_payload,
        {
            "room_id": "room_102",
            "device_id": "env_node_02",
            "timestamp": "2015-02-06 06:33:00",
            "Temperature": 30.5,
            "Humidity": 55.0,
            "Light": 80.0,
            "CO2": 2500.0,
            "HumidityRatio": 0.006
        }
    ]
}

if __name__ == "__main__":
    print("Checking /health ...")
    h = requests.get(f"{BASE_URL}/health", timeout=10)
    print(h.status_code, h.json())
    h.raise_for_status()

    print("\nChecking /model-info ...")
    info = requests.get(f"{BASE_URL}/model-info", timeout=10)
    print(info.status_code)
    print(json.dumps(info.json(), ensure_ascii=False, indent=2)[:1000])
    info.raise_for_status()

    print("\nChecking /predict ...")
    r = requests.post(f"{BASE_URL}/predict", json=sample_payload, timeout=10)
    print(r.status_code)
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    r.raise_for_status()
    data = r.json()
    assert "model_output" in data
    assert "decision" in data
    assert "occupancy_probability" in data["model_output"]
    assert "decision" in data["decision"]
    # Kiểm tra safety_reason có mặt
    assert "safety_reason" in data["decision"], "Missing safety_reason in /predict response"
    print("Single predict OK.")

    print("\nChecking /batch-predict ...")
    br = requests.post(f"{BASE_URL}/batch-predict", json=batch_payload, timeout=10)
    print(br.status_code)
    print(json.dumps(br.json(), ensure_ascii=False, indent=2))
    br.raise_for_status()
    batch_data = br.json()
    assert len(batch_data) == 2
    for item in batch_data:
        assert "model_output" in item
        assert "decision" in item
        assert "safety_reason" in item["decision"], "Missing safety_reason in batch response"
    print("Batch predict OK.")

    print("\nAPI TEST PASSED: FastAPI model deployment (single + batch) is working.")   