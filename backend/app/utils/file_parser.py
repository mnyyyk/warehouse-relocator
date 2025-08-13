"""
file_parser.py
==============

アップロードされた **CSV / Excel** ファイルを堅牢に `pandas.DataFrame`
へ変換するユーティリティ。

* MIME & 拡張子でファイル種別を判定
* CSV は **chardet** でエンコーディング推定＋フォールバックを順次試行
* ヘッダー行から BOM / 全角スペースを除去して統一
* 値はすべて **string 型** で読み取り (`dtype=str`, `keep_default_na=False`)
* 空ファイル・非対応形式は ``ValueError`` を送出

本モジュールは **FastAPI UploadFile / Path / str / bytes** の
いずれでも受け取れるように実装してあるので、  
API・pytest の両方でそのまま呼び出せる。
"""

from __future__ import annotations

import io
import mimetypes
from pathlib import Path
from typing import Final, Iterable

import pandas as pd
import chardet
import unicodedata

try:
    # UploadFile はテスト環境では import 不要なので optional
    from fastapi import UploadFile
except ModuleNotFoundError:  # pragma: no cover
    UploadFile = object   # type: ignore


ENCODINGS: Final[list[str]] = [
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp932",
    "iso8859-1",
]


# --------------------------------------------------------------------------- #
# public API                                                                  #
# --------------------------------------------------------------------------- #
def read_dataframe(file: UploadFile | str | Path | bytes | bytearray) -> pd.DataFrame:
    """
    Parameters
    ----------
    file :
        * **FastAPI UploadFile** – 実運用でのアップロード
        * **str / Path** – 単体テストやローカル実行で Path を直接渡す
        * **bytes / bytearray** – メモリ上のバイト列

    Returns
    -------
    pandas.DataFrame
        先頭行をヘッダーと見なし、すべて **文字列型** で読み込んだ表データ

    Raises
    ------
    ValueError
        - 空ファイル
        - 非対応ファイル形式
        - エンコーディング判定失敗
    """
    raw, filename = _get_raw_and_name(file)

    if not raw:
        raise ValueError("File is empty")

    mime, _ = mimetypes.guess_type(filename)
    lower_name = filename.lower()

    # ----------------------------- CSV ------------------------------------
    if mime in ("text/csv", None) or lower_name.endswith(".csv"):
        mime, _ = mimetypes.guess_type(filename)
        # If the first KB contains many NUL bytes the file is likely UTF‑16
        might_be_utf16 = b"\x00" in raw[:1024]
        enc_guess: str = (chardet.detect(raw[:4096]).get("encoding") or "").lower()

        # Build trial order: UTF‑16 first when it looks likely, then the guess, then fallbacks
        enc_try_order = (
            ["utf-16", "utf-16-le", "utf-16-be"] if might_be_utf16 else []
        ) + [enc_guess] + ENCODINGS

        for enc in _unique(enc_try_order):
            try:
                # csv.Sniffer does not work well on UTF‑16 so we force comma there.
                sep_param = None if enc.startswith("utf-8") or enc == "cp932" else ","

                df_try = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=enc,
                    dtype=str,
                    keep_default_na=False,
                    sep=sep_param,
                    engine="python",
                )
                # Validate parse result for delimiter detection failure
                if df_try.shape[1] == 1 and b"," in raw[:1024]:
                    try:
                        df_try2 = pd.read_csv(
                            io.BytesIO(raw),
                            encoding=enc,
                            dtype=str,
                            keep_default_na=False,
                            sep=",",        # explicit comma delimiter
                        )
                        if df_try2.shape[1] > 1:
                            df_try = df_try2
                    except Exception:
                        pass
                # If we still only have a single column, try a few common
                # alternative delimiters (tab / semicolon / pipe).  Some
                # Japanese ERP exports in CP932 use tab‑separated format.
                if df_try.shape[1] == 1:
                    for _sep in ("\t", ";", "|"):
                        try:
                            df_alt = pd.read_csv(
                                io.BytesIO(raw),
                                encoding=enc,
                                dtype=str,
                                keep_default_na=False,
                                sep=_sep,
                            )
                            if df_alt.shape[1] > 1:
                                df_try = df_alt
                                break
                        except Exception:
                            # Try next delimiter
                            continue
                df = df_try
                break
            except UnicodeDecodeError:
                continue
        else:  # すべて失敗
            raise ValueError("Cannot decode CSV – unknown encoding")

    # ----------------------------- Excel ----------------------------------
    elif lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw), dtype=str, keep_default_na=False)

    else:
        raise ValueError("Unsupported file type (only .csv/.xlsx/.xls accepted)")

    # ---------------------- column normalisation --------------------------
    # 1. Unicode‑normalise to NFKC so that full‑width ASCII (e.g. “ＳＫＵ”)
    #    collapses into regular ASCII (“SKU”), and weird whitespace
    #    variants are removed consistently.
    # 2. Strip BOM, full‑width spaces, leading/trailing spaces.
    df.columns = (
        df.columns.astype(str)
        .map(lambda s: unicodedata.normalize("NFKC", s))
        .str.replace("\ufeff", "", regex=False)   # strip BOM
        .str.replace("　", "", regex=False)       # full‑width space
        .str.strip()
    )

    # ---------------------- header alias folding --------------------------
    # Map synonymous header names that appear in exports from various WMS /
    # accounting systems to the canonical labels expected by the upload
    # mappers.  This prevents “header not found” errors without forcing
    # users to pre‑clean their files.
    _ALIASES = {
        # --- inventory ----------------------------------------------------
        "ロケーションコード": "ロケーション",
        "location": "ロケーション",
        "Location": "ロケーション",
        "ロケーションID": "ロケーション",
        "ロケーションId": "ロケーション",
        "在庫数(引当数を含む)": "在庫数",
        "数量": "在庫数",
        "数": "在庫数",
        # --- SKU -----------------------------------------------------------
        "SKU": "商品ID",
        "sku": "商品ID",
        "item_internalid": "商品ID",
        # additional common variations can be added here
    }

    # Rename in‑place where an alias is present
    df.rename(
        columns={c: _ALIASES[c] for c in df.columns if c in _ALIASES},
        inplace=True,
    )

    # ------------------------------------------------------------------
    # Heuristic fallback: handle vendor‑specific header variants that
    # don’t match the alias table above but still *contain* a canonical
    # keyword (e.g. "ロケーション（棚番）", "location code" …).
    # We only rename the *first* matching column for each canonical key
    # so we won’t accidentally rename multiple, similarly‑named columns.
    # ------------------------------------------------------------------
    def _maybe_rename(keyword: str, canonical: str) -> None:
        if canonical in df.columns:
            return
        for col in df.columns:
            if keyword.lower() in col.lower():
                df.rename(columns={col: canonical}, inplace=True)
                break

    # inventory‑specific common variants
    _maybe_rename("ロケーション", "ロケーション")
    _maybe_rename("location", "ロケーション")
    _maybe_rename("location code", "ロケーション")
    _maybe_rename("在庫", "在庫数")
    _maybe_rename("数量", "在庫数")

    if df.empty:
        raise ValueError("File has no data rows")

    return df


__all__ = ["read_dataframe"]


# --------------------------------------------------------------------------- #
# helpers (private)                                                           #
# --------------------------------------------------------------------------- #
def _get_raw_and_name(
    file: UploadFile | str | Path | bytes | bytearray,
) -> tuple[bytes, str]:
    """
    Convert various *file‑like* inputs into raw bytes + filename.

    Accepts:

    * FastAPI / Starlette ``UploadFile`` (objects with ``.file`` & ``.filename``)
    * ``str`` / ``pathlib.Path`` pointing to a file on disk
    * ``bytes`` / ``bytearray`` already in memory
    """
    # ----------------------- in‑memory bytes ------------------------------
    if isinstance(file, (bytes, bytearray)):
        return bytes(file), ""

    # ----------------------- filesystem path ------------------------------
    if isinstance(file, (str, Path)):
        p = Path(file)
        return p.read_bytes(), p.name

    # ------------------- FastAPI / Starlette UploadFile -------------------
    # NOTE: `fastapi.UploadFile` *is* `starlette.datastructures.UploadFile`,
    #       but when FastAPI is missing (pytest for utils) we fall back to
    #       a dummy ``object``.  Guard against that case explicitly.
    try:
        from starlette.datastructures import UploadFile as StarletteUploadFile  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover – extremely unlikely
        StarletteUploadFile = None  # type: ignore

    # FastAPI import succeeded -> UploadFile is the real class
    if "UploadFile" in globals() and UploadFile is not object and isinstance(file, UploadFile):  # type: ignore
        return file.file.read(), file.filename or ""

    # Fallback: Starlette class available but different import path
    if StarletteUploadFile is not None and isinstance(file, StarletteUploadFile):
        return file.file.read(), file.filename or ""

    # ---------------------------- duck typing -----------------------------
    if hasattr(file, "file") and hasattr(file, "filename"):
        return file.file.read(), getattr(file, "filename", "") or ""

    # ----------------------------- failure --------------------------------
    raise TypeError(
        "file must be UploadFile | str | Path | bytes | bytearray; "
        f"got {type(file)}"
    )


def _unique(seq: Iterable[str]) -> list[str]:
    """順序を保ったまま重複削除"""
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out