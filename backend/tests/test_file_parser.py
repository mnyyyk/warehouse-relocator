import pandas as pd
from app.utils.file_parser import read_dataframe
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

def test_utf8_csv():
    df = read_dataframe(DATA_DIR / "sku_utf8.csv")
    assert not df.empty
    assert "SKU" in df.columns

def test_cp932_csv():
    df = read_dataframe(DATA_DIR / "sku_cp932.csv")
    assert df.shape[0] >= 1
    assert set(df.columns) == {"SKU", "入数", "商品予備項目００６"}