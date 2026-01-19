# Copilot Instructions for Warehouse Optimizer

AI を用いた倉庫ロケーション最適化システム。

## Architecture Overview

| Layer | Tech Stack | Location |
|-------|-----------|----------|
| Backend API | FastAPI + SQLModel + pandas | [backend/app](backend/app) |
| Task Queue | Celery + Redis | [backend/app/core/celery_app.py](backend/app/core/celery_app.py) |
| Frontend | Next.js (TypeScript) | [frontend](frontend) |
| Infra IaC | Terraform (App Runner/ECR) | [infra](infra) |

**Data Flow**: CSV/Excel upload → `file_parser.py` (encoding normalization) → `upload.py` (upsert) → DB → `optimizer.py` (relocation planning) → AI/Heuristic → SSE stream to frontend

## Quick Start (Dev)

```bash
# 1. Start Postgres & Redis
make up  # docker-compose: Postgres:5432, Redis:6380

# 2. Backend
cd backend && poetry install --no-root && poetry run uvicorn app.main:app --reload --port 8000

# 3. Frontend
cd frontend && npm install && npm run dev  # http://localhost:3000

# Or run both: make dev
```

## Key Conventions

### Environment & Config
- **Priority**: `backend/.env.local` → `backend/.env` → `repo/.env.local` → `repo/.env` (no override)
- **DB default**: SQLite `dev.db` if `DATABASE_URL` unset; SQLite PRAGMAs auto-tuned for WAL/throughput
- **AI fallback**: Missing `OPENAI_API_KEY` logs warning → uses heuristic planner (`OPENAI_MODEL` defaults to `gpt-4o-mini`)

### File Parsing Pattern
Always use [backend/app/utils/file_parser.py](backend/app/utils/file_parser.py) for CSV/Excel:
```python
from app.utils.file_parser import read_dataframe
df = read_dataframe(uploaded_file)  # handles UTF-8/CP932/UTF-16, strips BOM, returns all strings
```

### SQLModel Compatibility Shim
[backend/app/models/__init__.py](backend/app/models/__init__.py) patches `Session.exec()` to return model instances (not RowMappings):
```python
# Works in this project (legacy compatibility):
rows = session.exec(select(Sku)).all()  # -> [Sku, Sku, ...]
```

### SKU ID Normalization
All SKU codes are canonicalized via `_normalize_sku_id()` in [upload.py](backend/app/routers/upload.py): NFKC normalize → strip whitespace → remove non-alphanumeric (except `-`) → uppercase.

### Location Format
Locations use 8-digit `LLLCCCDD` format (Level-Column-Depth). Parsed via `_parse_loc8()` in [optimizer.py](backend/app/services/optimizer.py).

## API Endpoints Summary

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/upload/sku\|inventory\|ship_tx\|recv_tx` | Bulk upload (replaces table) |
| `POST /v1/upload/location_master` | Valid/invalid/highness CSVs |
| `POST /v1/upload/analysis/start` | Recompute SKU metrics |
| `POST /v1/upload/relocation/start` | Plan moves (returns trace_id) |
| `GET /v1/upload/relocation/stream?trace_id=X` | SSE progress stream |
| `GET /v1/debug/*` | Debug views (SKU/inventory/txn) |
| `GET /files/{name}.csv` | Download error CSVs |

## Testing

```bash
cd backend && poetry run pytest  # or VS Code test task
```
Tests in [backend/tests](backend/tests) use in-memory SQLite; fixtures in [conftest.py](backend/tests/conftest.py).

## Celery Workers

```bash
# Ensure Redis is running (make up), then:
celery -A app.core.celery_app worker --loglevel=info
celery -A app.core.celery_app beat --loglevel=info  # for scheduled tasks
```

Broker: `redis://localhost:6379/0` (or `CELERY_BROKER_URL`). Tasks: `analysis_tasks`, `relocation_tasks`.

## Deployment

See [README.md](README.md) and [DEPLOY.md](DEPLOY.md). Key points:
- ECR + App Runner via Terraform (`infra/apprunner/`)
- Set `DATABASE_URL`, `OPENAI_API_KEY`, `FRONTEND_ORIGINS` in prod
- Production: disable `AUTO_CREATE_TABLES`, use Alembic migrations

## Relocation Workflow (optimizer.py)

The relocation planner in [optimizer.py](backend/app/services/optimizer.py) (~4300 lines) orchestrates warehouse slot optimization:

**Processing Phases:**
1. **Pass-0 (Eviction)**: Free up target slots by temporarily moving occupants
2. **Pass-1 (FIFO)**: Move older lots to pick locations (level 1-2)
3. **Pass-2 (Consolidation)**: Cluster same-SKU inventory into fewer columns
4. **Pass-3 (Pack-band)**: Group SKUs by similar case pack (±10%)

**Key Functions:**
- `plan_relocation()` - Main entry point; returns `List[Move]`
- `_parse_loc8(loc_str)` - Parse `LLLCCCDD` → `(level, column, depth)`
- `enforce_constraints()` - Validate moves against capacity/FIFO rules
- `_generate_relocation_summary()` - Build stats for frontend display

**Rejection Tracking:**
```python
from app.services.optimizer import get_last_rejection_debug
debug = get_last_rejection_debug()  # {"oversize": 5, "forbidden": 2, ...}
```

**SSE Progress Stream:**
```python
# Subscribe to real-time progress
GET /v1/upload/relocation/stream?trace_id=abc123
# Events: {"type": "progress", "phase": "Pass-1", "percent": 45}
```

## AI Planner (ai_planner.py)

[ai_planner.py](backend/app/services/ai_planner.py) provides OpenAI-based column recommendations:

**Configuration:**
- `OPENAI_API_KEY` - Required for AI mode; falls back to heuristic if missing
- `OPENAI_MODEL` - Default: `gpt-4o-mini`; supports `gpt-5` (no temperature param)

**Key Function:**
```python
from app.services.ai_planner import draft_relocation_with_ai
col_prefs = draft_relocation_with_ai(sku_df, inv_df, cfg)  # {"SKU123": [5, 12, 8], ...}
```

**Logging:**
All AI calls logged to `backend/logs/ai_planner.jsonl`:
```json
{"ts": "2026-01-19T10:30:00Z", "app": "warehouse-optimizer", "model": "gpt-4o-mini", "tokens": 1523}
```

## Test Data

Sample files in [warehouse-optimizer_testdata/](warehouse-optimizer_testdata/):

| File | Purpose |
|------|---------|
| `sku1.csv` | SKU master (商品ID, 入数, 容積) |
| `在庫データ20261223.csv` | Inventory snapshot |
| `出荷実績_20251223.csv` | Shipment transactions |
| `入荷実績_20251222.csv` | Receipt transactions |
| `*Bネス有効ロケ.csv` | Valid location master |
| `*無効化したロケ一覧.csv` | Disabled locations |
| `*ハイネスロケ.csv` | High-rack (highness) locations |

**Upload order for testing:**
1. SKU master → 2. Location master (valid/invalid/highness) → 3. Inventory → 4. Ship/Recv TX

## Alembic Migrations

Production deployments should use Alembic instead of `AUTO_CREATE_TABLES`:

```bash
cd backend

# Create new migration
alembic revision --autogenerate -m "add_new_column"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

**Config:** [alembic.ini](backend/alembic.ini) + [alembic/env.py](backend/alembic/env.py)

The env.py imports `app.models` to register all SQLModel tables in metadata. Existing migrations in `alembic/versions/`.

## Frontend Integration

Next.js frontend in [frontend/](frontend/) with pages:

| Page | Purpose |
|------|---------|
| `/upload` | File upload UI (SKU/Inventory/TX/Locations) |
| `/analyze` | Trigger SKU metrics recomputation |
| `/optimize` | Start relocation planning + SSE progress |
| `/debug` | View DB contents |

**API Client:** [frontend/lib/api.ts](frontend/lib/api.ts)
```typescript
import { postCSV } from '@/lib/api';
await postCSV('/v1/upload/sku', file);  // FormData upload
```

**Environment:**
- `NEXT_PUBLIC_API_BASE` - Backend URL (default: `http://localhost:8000`)

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| AI planner falls back to Greedy | Set `OPENAI_API_KEY` |
| CORS errors | Add origin to `FRONTEND_ORIGINS` |
| SQLite lot_date issues | Postgres-only feature; ignored on SQLite |
| Upload errors | Check `/files/*.csv` for error details |
| Celery tasks not running | Ensure Redis is up (`make up`) and worker started |
| Import errors in migrations | `app.models` must be imported in `alembic/env.py` |
| SSE stream disconnects | Check `trace_id` matches; buffer limit is 200 events |