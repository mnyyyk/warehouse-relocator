"""Routes for uploading data files that seed or update the local DB.

Files are stored temporarily under the `uploads/` directory with a UUID
filename, then passed to the `app.services.ingest` pipeline which parses the
file and writes the relevant records via SQLAlchemy.

Expected file types (determined heuristically by original filename):
* SKU master (e.g. `SKU.xlsx`)
* Receipt history (e.g. `入荷実績*.xlsx`)
* Shipment history (e.g. `出荷実績*.xlsx`)

If the filename cannot be mapped to a known ingest type the request is rejected
with HTTP 400.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.services import ingest

# --------------------------------------------------------------------------- #
# FastAPI boilerplate
# --------------------------------------------------------------------------- #
router = APIRouter(prefix="/uploads", tags=["uploads"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def get_db() -> Session:
    """FastAPI dependency that yields a DB session and ensures it is closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@router.post(
    "/masters",
    summary="Upload a master/history data file (SKU/入荷実績/出荷実績) and trigger ingest.",
)
async def upload_master_file(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(
        None, alias="files"
    ),  # HTML forms often send “files”
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """
    Accept a single Excel file from the client, save it locally, and feed it to
    the ingest pipeline.

    Returns a small JSON payload with the stored filename and the inferred
    ingest type.
    """
    # ------------------------------------------------------------------ #
    # Determine list of uploads
    # ------------------------------------------------------------------ #
    if file is not None:
        upload_list: list[UploadFile] = [file]
    elif files is not None:
        if len(files) == 0:
            raise HTTPException(
                status_code=422,
                detail="No file received in form field 'files'.",
            )
        upload_list = list(files)  # keep original order
    else:
        raise HTTPException(
            status_code=422,
            detail="No file received; expected form field 'file' or 'files'.",
        )

    results: list[dict[str, str]] = []

    for upload in upload_list:
        # Persist to temp file
        file_id = uuid.uuid4().hex
        saved_path = UPLOAD_DIR / f"{file_id}{Path(upload.filename).suffix}"
        try:
            with saved_path.open("wb") as buf:
                shutil.copyfileobj(upload.file, buf)
        finally:
            await upload.close()

        # Detect ingest type
        try:
            ingest_type = ingest.detect_file_type(upload.filename)
        except ValueError as exc:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # Run ingest – pass an (filename, file‑obj) tuple expected by ingest.ingest_file
        try:
            with saved_path.open("rb") as fp:
                ingest.ingest_file((upload.filename, fp), db=db)
        except Exception as exc:  # noqa: BLE001
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500, detail=f"Ingest failed for {upload.filename}: {exc}"
            ) from exc

        results.append(
            {
                "original_filename": upload.filename,
                "stored_as": saved_path.name,
                "ingest_type": ingest_type,
            }
        )

    return {"status": "ok", "files_processed": len(results), "results": results}


# --------------------------------------------------------------------------- #
# Back‑compat: allow front‑end to POST to `/uploads/master` (with or w/o slash)
# --------------------------------------------------------------------------- #
for alt in ("/master", "/master/"):
    router.add_api_route(
        alt,
        upload_master_file,
        methods=["POST"],
        summary="Upload a master/history data file (alias of /uploads/masters)",
        tags=["uploads"],
    )



@router.post(
    "/inventory",
    summary="Upload the current inventory snapshot file and trigger ingest.",
)
async def upload_inventory_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    """
    Accept a single Excel file representing the *current* inventory snapshot,
    save it locally, and feed it to the ingest pipeline (ingest_type='inventory').

    Front‑end `index.html` should call this endpoint.
    """
    # ------------------------------------------------------------------ #
    # Persist file to a temporary location
    # ------------------------------------------------------------------ #
    file_id = uuid.uuid4().hex
    saved_path = UPLOAD_DIR / f"{file_id}{Path(file.filename).suffix}"
    try:
        with saved_path.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
    finally:
        # Explicitly close the SpooledTemporaryFile created by FastAPI
        await file.close()

    # ------------------------------------------------------------------ #
    # Kick off ingest with explicit ingest_type='inventory'
    # ------------------------------------------------------------------ #
    try:
        with saved_path.open("rb") as fp:
            ingest.ingest_file((file.filename, fp), db=db)
    except Exception as exc:  # noqa: BLE001
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Ingest failed: {exc}"
        ) from exc

    return {
        "status": "ok",
        "stored_as": saved_path.name,
        "ingest_type": "inventory",
    }
