from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from dotenv import load_dotenv


# ---- load .env files (backend/.env then repo .env) ----------------------
CURRENT_FILE = Path(__file__).resolve()
BACKEND_DIR = CURRENT_FILE.parents[1]
REPO_ROOT = CURRENT_FILE.parents[2]

# Collect candidate .env files in priority order
_env_candidates = [
    BACKEND_DIR / ".env.local",
    BACKEND_DIR / ".env",
    REPO_ROOT / ".env.local",
    REPO_ROOT / ".env",
]
_loaded = []
for env_path in _env_candidates:
    if env_path.exists():
        # Do not override already-set env vars; load in priority order
        load_dotenv(env_path, override=False)
        _loaded.append(str(env_path))

if _loaded:
    print(f"[main] Loaded env files: {', '.join(_loaded)}")
else:
    print("[main] No .env file found next to backend/ or repo root.")

# Warn once if AI env vars are missing (helps detect fallback to Greedy)
if not os.getenv("OPENAI_API_KEY"):
    print("[main] Warning: OPENAI_API_KEY not found; AI main planner may fall back to Greedy.")
if not os.getenv("OPENAI_MODEL"):
    print("[main] Info: OPENAI_MODEL not set; defaulting to gpt-4o-mini.")

# ルーター: ファイルアップロード / デバッグ API（.env ロード後にインポート）
from app.routers import upload
from app.routers import debug


app = FastAPI(title="Warehouse Optimizer API")


# ---- CORS (dev-friendly) ----------------------------------------------
_default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://0.0.0.0:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
]
_env = os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN") or ""
_env_list = [o.strip() for o in _env.split(",") if o and o.strip()]
origins = sorted(set(_default_origins + _env_list))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|[^/]+\.vercel\.app|[^/]+\.warehouse-optimizer\.net)(:\d+)?$",
    allow_credentials=False,   # Cookie 認証が必要なら True に
    allow_methods=["*"],      # OPTIONS を含む全メソッド
    allow_headers=["*"],      # Content-Type など全許可
    max_age=86400,
)

# ---- serve generated error CSVs ------------------------------------------
ERROR_DIR = Path("/tmp/upload_errors")
ERROR_DIR.mkdir(parents=True, exist_ok=True)  # ensure it exists at startup
app.mount("/files", StaticFiles(directory=str(ERROR_DIR)), name="files")

# ---- register routers ------------------------------------------------------
app.include_router(upload.router)
app.include_router(debug.router)


# ---- simple health check ---------------------------------------------------
@app.get("/health")
async def health() -> dict[str, str]:
    """コンテナ／プロセス生存確認用エンドポイント"""
    return {"status": "ok"}
