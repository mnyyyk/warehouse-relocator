from __future__ import annotations

from typing import Optional, Sequence

from sqlmodel import Session
from sqlalchemy import text as _text
def recompute_all_sku_metrics(
    session: Session,
    *,
    turnover_window_days: int = 30,
    block_filter: Optional[Sequence[str]] = None,
) -> int:
    """
    回転率/速度メトリクスを SKU 単位で再計算して、`sku_metrics` テーブルへUPSERTします。

    出荷系:
      - shipped_cases_all: ship_tx から Σ(qty / pack_qty)   （期間は window_days で制限）
      - cases_per_day     : shipped_cases_all / window_days
      - hits_per_day      : 出荷レコード数 / window_days
      - cube_per_day      : Σ((qty/pack_qty)*volume_m3) / window_days

    在庫:
      - current_cases     : inventory から Σ(cases)          （block_filter があればブロック限定）
      - turnover_rate     : shipped_cases_all / current_cases (在庫0のときは0)

    入荷系（新規追加）:
      - recv_cases_per_day: Σ(qty/pack_qty) / window_days
      - recv_hits_per_day : 入荷レコード数 / window_days
      - recv_cube_per_day : Σ((qty/pack_qty)*volume_m3) / window_days

    永続化スキーマ（なければ自動作成: dev用途）:
      sku_id text, window_days int, shipped_cases_all numeric,
      current_cases numeric, turnover_rate numeric,
      cases_per_day numeric, hits_per_day numeric, cube_per_day numeric,
      recv_cases_per_day numeric, recv_hits_per_day numeric, recv_cube_per_day numeric,
      updated_at timestamptz
      PK: (sku_id, window_days)
    """
    # 接続ダイアレクト（例外時も常に文字列にフォールバック）
    dialect = ""
    try:
        bind = session.get_bind()
        if bind is not None and getattr(bind, "dialect", None) is not None:
            dialect = str(getattr(bind.dialect, "name", "") or "")
    except Exception:
        # ログは出さずに後段の方言分岐を安全にスキップ（dev用途）
        dialect = ""

    # 1) dev向け: テーブル自動作成（本番はAlembic推奨）
    session.exec(_text(
        """
        CREATE TABLE IF NOT EXISTS sku_metrics (
            sku_id              text        NOT NULL,
            window_days         integer     NOT NULL,
            shipped_cases_all   numeric     NOT NULL,
            current_cases       numeric     NOT NULL,
            turnover_rate       numeric     NOT NULL,
            cases_per_day       numeric     NOT NULL DEFAULT 0,
            hits_per_day        numeric     NOT NULL DEFAULT 0,
            cube_per_day        numeric     NOT NULL DEFAULT 0,
            recv_cases_per_day  numeric     NOT NULL DEFAULT 0,
            recv_hits_per_day   numeric     NOT NULL DEFAULT 0,
            recv_cube_per_day   numeric     NOT NULL DEFAULT 0,
            updated_at          timestamptz NOT NULL,
            PRIMARY KEY (sku_id, window_days)
        );
        """
    ))
    # 既存テーブルに不足していれば列を追加（後方互換）
    if dialect == "sqlite":
        # SQLite は古いバージョンで "ADD COLUMN IF NOT EXISTS" 非対応のため PRAGMA 経由で存在確認
        cols = session.exec(_text("PRAGMA table_info(sku_metrics)"))
        existing = {str(r[1]) for r in cols}  # r[1] = name
        def add_col(name: str, sql_type: str = "numeric", default: str = "0"):
            if name not in existing:
                session.exec(_text(f"ALTER TABLE sku_metrics ADD COLUMN {name} {sql_type} NOT NULL DEFAULT {default}"))
        add_col("cases_per_day")
        add_col("hits_per_day")
        add_col("cube_per_day")
        add_col("recv_cases_per_day")
        add_col("recv_hits_per_day")
        add_col("recv_cube_per_day")
    else:
        session.exec(_text("ALTER TABLE sku_metrics ADD COLUMN IF NOT EXISTS cases_per_day      numeric NOT NULL DEFAULT 0"))
        session.exec(_text("ALTER TABLE sku_metrics ADD COLUMN IF NOT EXISTS hits_per_day       numeric NOT NULL DEFAULT 0"))
        session.exec(_text("ALTER TABLE sku_metrics ADD COLUMN IF NOT EXISTS cube_per_day       numeric NOT NULL DEFAULT 0"))
        session.exec(_text("ALTER TABLE sku_metrics ADD COLUMN IF NOT EXISTS recv_cases_per_day numeric NOT NULL DEFAULT 0"))
        session.exec(_text("ALTER TABLE sku_metrics ADD COLUMN IF NOT EXISTS recv_hits_per_day  numeric NOT NULL DEFAULT 0"))
        session.exec(_text("ALTER TABLE sku_metrics ADD COLUMN IF NOT EXISTS recv_cube_per_day  numeric NOT NULL DEFAULT 0"))
    session.commit()

    # 2) 最新の取引日付を取得（入荷・出荷の最新日付）
    # SQLiteにはGREATEST関数がないので、UNIONで最大値を取得
    if dialect == "sqlite":
        max_date_sql = """
            SELECT MAX(trandate) AS latest_trandate FROM (
                SELECT MAX(trandate) AS trandate FROM ship_tx
                UNION ALL
                SELECT MAX(trandate) AS trandate FROM recv_tx
            )
        """
    else:
        max_date_sql = """
            SELECT GREATEST(
                COALESCE((SELECT MAX(trandate) FROM ship_tx), '1900-01-01'),
                COALESCE((SELECT MAX(trandate) FROM recv_tx), '1900-01-01')
            ) AS latest_trandate
        """
    
    result = session.exec(_text(max_date_sql)).first()
    latest_date = result[0] if result else None
    
    print(f"[metrics] Latest transaction date detected: {latest_date}")
    
    # 3) 期間（window）の条件式（方言対応）
    # 起点を最新取引日から遡る方式に変更
    days = int(turnover_window_days or 0)
    if 0 < days < 99999:
        if latest_date:
            if dialect == "sqlite":
                # SQLite: 最新日から指定日数遡る
                window_cond = "WHERE date(trandate) >= date('%s', '-%s days')" % (str(latest_date)[:10], days)
            else:
                # PostgreSQL: 最新日から指定日数遡る
                window_cond = "WHERE trandate >= (DATE '%s' - INTERVAL '%s days')" % (str(latest_date)[:10], days)
        else:
            # 取引データが無い場合はフォールバック（現在日）
            if dialect == "sqlite":
                window_cond = "WHERE date(trandate) >= date('now', '-%s days')" % days
            else:
                window_cond = "WHERE trandate >= (CURRENT_DATE - INTERVAL '%s days')" % days
    else:
        days = 99999  # 全期間扱い
        window_cond = ""

    # 4) 在庫ブロック制限（exec にキーワード引数を渡さないため、VALUES で結合する方式）
    if block_filter:
        # SQLインジェクション対策: シングルクォートを二重化
        _sanitized = [str(b).replace("'", "''") for b in block_filter]
        _vals = ", ".join(f"('{x}')" for x in _sanitized)
        stock_source = f"FROM inventory i JOIN (VALUES {_vals}) AS b(block_code) ON b.block_code = i.block_code"
    else:
        stock_source = "FROM inventory i"

    # 5) 集計 → UPSERT
    # updated_at 式（UTC）
    updated_expr = "CURRENT_TIMESTAMP" if dialect == "sqlite" else "NOW() AT TIME ZONE 'UTC'"

    # 分母（window_days）のゼロ割回避は Python 側で固定値化
    denom = max(days, 1)
    denom_literal = f"{float(denom)}"  # 両DBで少数にしておく

    # 型キャスト（両DBで通る書き方）
    # PostgreSQL: DOUBLE PRECISION は float8, SQLite: 浮動小数扱い
    cast_qty = "CAST(t.qty AS DOUBLE PRECISION)"
    cast_pack_t = "CAST(NULLIF(t.pack_qty,0) AS DOUBLE PRECISION)"
    cast_pack_s = "CAST(NULLIF(s.pack_qty,0) AS DOUBLE PRECISION)"
    cast_count = "CAST(COUNT(*) AS DOUBLE PRECISION)"
    cast_sum_cases = "CAST(SUM(i.cases) AS DOUBLE PRECISION)"

    # SQLite と PostgreSQL で UPSERT 構文を分岐
    common_cte = f"""
    WITH ship AS (
        SELECT t.sku_id,
               SUM({cast_qty} / {cast_pack_t})                                              AS shipped_cases_all,
               {cast_count}                                                                 AS ship_hits,
               SUM(({cast_qty} / {cast_pack_t}) * COALESCE(s.volume_m3, 0))                 AS ship_cube
        FROM ship_tx t
        JOIN sku s ON s.sku_id = t.sku_id
        {window_cond}
        GROUP BY t.sku_id
    ),
    recv AS (
        SELECT t.sku_id,
               SUM({cast_qty} / {cast_pack_s})                                              AS recv_cases_all,
               {cast_count}                                                                 AS recv_hits,
               SUM(({cast_qty} / {cast_pack_s}) * COALESCE(s.volume_m3, 0))                 AS recv_cube
        FROM recv_tx t
        JOIN sku s ON s.sku_id = t.sku_id
        {window_cond}
        GROUP BY t.sku_id
    ),
    stock AS (
        SELECT i.sku_id,
               {cast_sum_cases} AS stock_cases
        {stock_source}
        GROUP BY i.sku_id
    ),
    all_sku AS (
        SELECT sku_id FROM sku
    ),
    joined AS (
        SELECT s.sku_id,
               COALESCE(ship.shipped_cases_all, 0) AS shipped_cases_all,
               COALESCE(ship.ship_hits,        0) AS ship_hits,
               COALESCE(ship.ship_cube,        0) AS ship_cube,
               COALESCE(recv.recv_cases_all,   0) AS recv_cases_all,
               COALESCE(recv.recv_hits,        0) AS recv_hits,
               COALESCE(recv.recv_cube,        0) AS recv_cube,
               COALESCE(st.stock_cases,        0) AS current_cases
        FROM all_sku s
        LEFT JOIN ship  ON ship.sku_id  = s.sku_id
        LEFT JOIN recv  ON recv.sku_id  = s.sku_id
        LEFT JOIN stock st ON st.sku_id = s.sku_id
    )
    """

    if dialect == "sqlite":
        sql_insert = common_cte + f"""
    INSERT OR REPLACE INTO sku_metrics (sku_id, window_days,
                             shipped_cases_all, current_cases, turnover_rate,
                             cases_per_day, hits_per_day, cube_per_day,
                             recv_cases_per_day, recv_hits_per_day, recv_cube_per_day,
                             updated_at)
    SELECT j.sku_id,
           {days} AS window_days,
           j.shipped_cases_all,
           j.current_cases,
           CASE WHEN j.current_cases > 0
                THEN ROUND(CAST(j.shipped_cases_all / NULLIF(j.current_cases,0) AS NUMERIC), 3)
                ELSE 0 END AS turnover_rate,
           ROUND(CAST(j.shipped_cases_all / {denom_literal} AS NUMERIC), 6) AS cases_per_day,
           ROUND(CAST(j.ship_hits        / {denom_literal} AS NUMERIC), 6) AS hits_per_day,
           ROUND(CAST(j.ship_cube        / {denom_literal} AS NUMERIC), 6) AS cube_per_day,
           ROUND(CAST(j.recv_cases_all   / {denom_literal} AS NUMERIC), 6) AS recv_cases_per_day,
           ROUND(CAST(j.recv_hits        / {denom_literal} AS NUMERIC), 6) AS recv_hits_per_day,
           ROUND(CAST(j.recv_cube        / {denom_literal} AS NUMERIC), 6) AS recv_cube_per_day,
           {updated_expr} AS updated_at
    FROM joined j
    """
    else:
        sql_insert = common_cte + f"""
    INSERT INTO sku_metrics (sku_id, window_days,
                             shipped_cases_all, current_cases, turnover_rate,
                             cases_per_day, hits_per_day, cube_per_day,
                             recv_cases_per_day, recv_hits_per_day, recv_cube_per_day,
                             updated_at)
    SELECT j.sku_id,
           {days} AS window_days,
           j.shipped_cases_all,
           j.current_cases,
           CASE WHEN j.current_cases > 0
                THEN ROUND(CAST(j.shipped_cases_all / NULLIF(j.current_cases,0) AS NUMERIC), 3)
                ELSE 0 END AS turnover_rate,
           ROUND(CAST(j.shipped_cases_all / {denom_literal} AS NUMERIC), 6) AS cases_per_day,
           ROUND(CAST(j.ship_hits        / {denom_literal} AS NUMERIC), 6) AS hits_per_day,
           ROUND(CAST(j.ship_cube        / {denom_literal} AS NUMERIC), 6) AS cube_per_day,
           ROUND(CAST(j.recv_cases_all   / {denom_literal} AS NUMERIC), 6) AS recv_cases_per_day,
           ROUND(CAST(j.recv_hits        / {denom_literal} AS NUMERIC), 6) AS recv_hits_per_day,
           ROUND(CAST(j.recv_cube        / {denom_literal} AS NUMERIC), 6) AS recv_cube_per_day,
           {updated_expr} AS updated_at
    FROM joined j
    ON CONFLICT (sku_id, window_days) DO UPDATE
    SET shipped_cases_all  = EXCLUDED.shipped_cases_all,
        current_cases      = EXCLUDED.current_cases,
        turnover_rate      = EXCLUDED.turnover_rate,
        cases_per_day      = EXCLUDED.cases_per_day,
        hits_per_day       = EXCLUDED.hits_per_day,
        cube_per_day       = EXCLUDED.cube_per_day,
        recv_cases_per_day = EXCLUDED.recv_cases_per_day,
        recv_hits_per_day  = EXCLUDED.recv_hits_per_day,
        recv_cube_per_day  = EXCLUDED.recv_cube_per_day,
        updated_at         = EXCLUDED.updated_at;
    """

    session.exec(_text(sql_insert))
    session.commit()

    # 5) 行数を返す（window一致）+ ログ出力
    res = session.exec(_text(f"SELECT COUNT(*) FROM sku_metrics WHERE window_days = {days}")).one()
    count = int(res[0]) if res and len(res) > 0 else 0
    
    # デバッグ用: 最新日付と起点日付をログ出力
    if latest_date and 0 < days < 99999:
        from datetime import datetime, timedelta
        try:
            latest_dt = datetime.fromisoformat(str(latest_date)[:10])
            start_dt = latest_dt - timedelta(days=days)
            print(f"[metrics] Latest transaction date: {latest_dt.date()}, Analysis period: {start_dt.date()} to {latest_dt.date()} ({days} days)")
        except Exception:
            pass
    
    return count