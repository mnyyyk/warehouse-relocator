"""
Central Celery application object for the Warehouse‑Optimizer backend.

Usage
-----
* **Worker**: ``celery -A app.core.celery_app worker --loglevel=info``
* **Beat (scheduled jobs)**: ``celery -A app.core.celery_app beat --loglevel=info``

The broker/result backend URLs can be overridden via environment variables:

    CELERY_BROKER_URL   (default: redis://localhost:6379/0)
    CELERY_RESULT_BACKEND (default: same as broker)
"""

from __future__ import annotations

import os
from datetime import timedelta

from celery import Celery
from kombu import Exchange, Queue

# --------------------------------------------------------------------------- #
# Configuration via environment variables                                     #
# --------------------------------------------------------------------------- #

BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL)
TIMEZONE: str = os.getenv("APP_TIMEZONE", "Asia/Tokyo")

# --------------------------------------------------------------------------- #
# Celery application                                                          #
# --------------------------------------------------------------------------- #

celery_app = Celery(
    "warehouse_optimizer",
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        "app.services.analysis_tasks",
        "app.services.relocation_tasks",
    ],
)

# --------------------------------------------------------------------------- #
# Default settings                                                            #
# --------------------------------------------------------------------------- #

celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Reliability
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Time
    timezone=TIMEZONE,
    enable_utc=True,
    # Queues / routing
    task_default_queue="default",
    task_default_exchange="default",
    task_default_routing_key="default",
    task_queues=(
        Queue("default", Exchange("default"), routing_key="default"),
        Queue("analysis", Exchange("analysis"), routing_key="analysis"),
        Queue("relocation", Exchange("relocation"), routing_key="relocation"),
    ),
    # Result expiry
    result_expires=timedelta(days=1),
)

# --------------------------------------------------------------------------- #
# Helper for FastAPI integration                                              #
# --------------------------------------------------------------------------- #


def init_celery() -> None:  # called from FastAPI.startup
    """
    Import all celery tasks to ensure they are registered when the web API
    process (uvicorn) starts.  This avoids a common gotcha where delaying
    task definition causes ``.delay`` calls to fail with *NotRegistered*.
    """
    # The `include=` list above is usually good enough, but an explicit
    # import here makes local development & hot‑reload more robust.
    from importlib import import_module

    for module in celery_app.conf.include:
        import_module(module)
