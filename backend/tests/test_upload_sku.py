# backend/tests/test_upload_sku.py
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
DATA_DIR = Path(__file__).parent / "data"

def test_upload_sku_csv():
    with (DATA_DIR / "sku_sample.csv").open("rb") as f:
        resp = client.post("/v1/upload/sku", files={"file": ("sku_sample.csv", f, "/Users/kounoyousuke/warehouse‑relocator_testdata")})
    assert resp.status_code == 200
    j = resp.json()
    assert j["success_rows"] > 0
    assert j["errors"] == []