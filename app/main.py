"""
main.py
~~~~~~~
ASGI entrypoint for *warehouse‑relocator*.

Run locally with:
    uvicorn app.main:app --reload
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pathlib
from fastapi.responses import RedirectResponse

from .routers.uploads import router as uploads_router
from .routers.analysis import router as analysis_router
from .db import Base, engine

# ----------------------------------------------------------------------
# FastAPI instance
# ----------------------------------------------------------------------
app = FastAPI(
    title="warehouse‑relocator API",
    description="AI‑driven slotting optimisation service",
    version="0.1.0",
)

# Optional: permissive CORS for local PoC
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------
app.include_router(uploads_router)
app.include_router(analysis_router)

# ----------------------------------------------------------------------
# Database initialisation
# ----------------------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    """Create database tables on startup."""
    Base.metadata.create_all(bind=engine)

# ----------------------------------------------------------------------
# Static file mounting  (serves webui/index.html at /webui/)
# ----------------------------------------------------------------------
static_dir = pathlib.Path(__file__).resolve().parent.parent / "webui"
app.mount("/webui", StaticFiles(directory=str(static_dir), html=True), name="webui")


# ----------------------------------------------------------------------
# Root → Web UI and health check
# ----------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_webui_root() -> RedirectResponse:
    """Redirect the API root to the Web UI index page."""
    return RedirectResponse(url="/webui/")

@app.get("/api/health", tags=["meta"])
async def health_check() -> dict[str, str]:
    """Simple health‑check endpoint suitable for uptime monitors."""
    return {"status": "ok"}