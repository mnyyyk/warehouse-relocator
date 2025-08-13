import pandas as pd
from sqlmodel import Session
from app.services.ingestion import ingest
from app.models import Sku
from app.core.database import engine

def test_ingest_sku(tmp_path):
    df = pd.DataFrame(
        {
            "SKU": ["X1", "X2"],
            "入数": [10, 1],
            "商品予備項目００６": [0.123, 0.456],
        }
    )
    with Session(engine) as ses:
        summary = ingest("sku", df, ses)
        assert summary["success_rows"] == 2
        # DB 確認
        rows = ses.exec(Sku.select()).all()
        assert {r.sku_id for r in rows} == {"X1", "X2"}