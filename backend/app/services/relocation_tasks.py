"""
Celery background-tasks for the *relocation* workflow.

Supports both synchronous (direct call) and asynchronous (Celery worker) execution.
Results are stored in Redis via Celery's result backend.
"""

from __future__ import annotations

import celery
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Mapping
from uuid import uuid4

import pandas as pd
from celery import states
from celery.exceptions import Ignore, SoftTimeLimitExceeded
from celery.result import AsyncResult

from app.core.celery_app import celery_app, RESULT_BACKEND
from app.core.database import engine
from sqlmodel import Session

# Conditional imports for type safety
try:
    from app.models import RelocationMove  # type: ignore
except ImportError:
    RelocationMove = None  # type: ignore

try:
    from app.services.optimizer import (
        plan_relocation,
        OptimizerConfig,
        get_last_rejection_debug,
        get_current_trace_id,
        cache_moves,
        start_drop_trace,
        bind_trace_id,
        _publish_progress,
        get_last_summary_report,
    )
except ImportError as e:
    logging.getLogger(__name__).warning(f"Could not import optimizer: {e}")
    plan_relocation = None  # type: ignore
    OptimizerConfig = None  # type: ignore

logger = logging.getLogger(__name__)

# Redis client for result storage (separate from Celery result backend for custom TTL)
_redis_client = None


def _get_redis():
    """Lazy-load Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
            _redis_client = redis.from_url(redis_url, decode_responses=True)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return None
    return _redis_client


def store_relocation_result(trace_id: str, result: dict, ttl: int = 3600) -> bool:
    """Store relocation result in Redis with TTL (default 1 hour)."""
    r = _get_redis()
    if r is None:
        return False
    try:
        key = f"relocation:result:{trace_id}"
        r.setex(key, ttl, json.dumps(result, ensure_ascii=False, default=str))
        return True
    except Exception as e:
        logger.warning(f"Failed to store result in Redis: {e}")
        return False


def get_relocation_result(trace_id: str) -> Optional[dict]:
    """Get relocation result from Redis."""
    r = _get_redis()
    if r is None:
        return None
    try:
        key = f"relocation:result:{trace_id}"
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.warning(f"Failed to get result from Redis: {e}")
        return None


def store_relocation_status(trace_id: str, status: str, progress: dict = None, ttl: int = 3600) -> bool:
    """Store relocation status in Redis."""
    r = _get_redis()
    if r is None:
        return False
    try:
        key = f"relocation:status:{trace_id}"
        data = {"status": status, "updated_at": datetime.utcnow().isoformat()}
        if progress:
            data["progress"] = progress
        r.setex(key, ttl, json.dumps(data, ensure_ascii=False, default=str))
        return True
    except Exception as e:
        logger.warning(f"Failed to store status in Redis: {e}")
        return False


def get_relocation_status(trace_id: str) -> Optional[dict]:
    """Get relocation status from Redis."""
    r = _get_redis()
    if r is None:
        return None
    try:
        key = f"relocation:status:{trace_id}"
        data = r.get(key)
        if data:
            return json.loads(data)
        return None
    except Exception as e:
        logger.warning(f"Failed to get status from Redis: {e}")
        return None


def _parse_loc8(loc: str):
    """Parse 8-digit location code into (level, column, depth)."""
    s = str(loc or "")
    if len(s) == 8 and s.isdigit():
        try:
            return int(s[0:3]), int(s[3:6]), int(s[6:8])
        except Exception:
            return 0, 0, 0
    return 0, 0, 0


def _ease_key(lv: int, col: int, dep: int) -> int:
    """Calculate ease key for move prioritization."""
    try:
        col_rev = 42 - int(col)
        return int(lv) * 10000 + int(col_rev) * 100 + int(dep)
    except Exception:
        return 99_999_999


def _build_summary(inv_df: pd.DataFrame, moves_data: list[dict]) -> dict:
    """Build efficiency summary for UI."""
    if not isinstance(inv_df, pd.DataFrame) or not isinstance(moves_data, list) or not moves_data:
        return {"moves": 0}
    
    total = len(moves_data)
    touched = set()
    uniq_skus = set()
    qty_total = 0
    lvl_delta_sum = 0
    col_dist_sum = 0
    dep_dist_sum = 0
    ease_gain_sum = 0
    to_pick = 0
    examples = []
    
    for m in moves_data:
        sku = str(m.get("sku_id"))
        lot = str(m.get("lot") or m.get("lot_date_key") or "")
        fr = str(m.get("from_loc") or m.get("from") or "")
        to = str(m.get("to_loc") or m.get("to") or "")
        qty = int(m.get("qty") or m.get("qty_cases") or m.get("ケース") or 0)
        
        uniq_skus.add(sku)
        touched.add(fr)
        touched.add(to)
        qty_total += max(0, qty)
        
        flv, fcol, fdep = _parse_loc8(fr)
        tlv, tcol, tdep = _parse_loc8(to)
        
        if tlv in (1, 2):
            to_pick += 1
        lvl_delta_sum += (flv - tlv)
        col_dist_sum += abs(tcol - fcol)
        dep_dist_sum += abs(tdep - fdep)
        ease_gain_sum += max(0, _ease_key(flv, fcol, fdep) - _ease_key(tlv, tcol, tdep))
        
        if len(examples) < 5:
            examples.append({
                "sku_id": sku,
                "lot": lot,
                "from_loc": fr,
                "to_loc": to,
                "qty": qty,
                "level_change": (flv - tlv),
                "column_change": (tcol - fcol),
                "depth_change": (tdep - fdep),
                "ease_gain": max(0, _ease_key(flv, fcol, fdep) - _ease_key(tlv, tcol, tdep)),
            })
    
    return {
        "moves": total,
        "unique_skus": len(uniq_skus),
        "locations_touched": len([x for x in touched if x and x not in ("", "00000000")]),
        "qty_cases_total": qty_total,
        "avg_level_improvement": (lvl_delta_sum / total) if total else 0.0,
        "avg_column_distance": (col_dist_sum / total) if total else 0.0,
        "avg_depth_distance": (dep_dist_sum / total) if total else 0.0,
        "moves_to_pick_levels": to_pick,
        "ease_gain_sum": ease_gain_sum,
        "ease_gain_avg": (ease_gain_sum / total) if total else 0.0,
        "highlights": examples,
    }


# Hard timeout: 10 minutes (600s), soft timeout: 9 minutes (540s)
# This prevents tasks from hanging indefinitely
@celery_app.task(
    bind=True,
    name="relocation.run_async",
    queue="relocation",
    time_limit=600,      # Hard kill after 10 minutes
    soft_time_limit=540,  # Raise SoftTimeLimitExceeded after 9 minutes
    acks_late=True,       # Acknowledge after task completion (retry on worker crash)
)
def run_relocation_async(
    self,
    *,
    trace_id: str,
    config_dict: dict,
    sku_data: list[dict],
    inv_data: list[dict],
    location_master_data: list[dict],
    block_codes: list[str] | None = None,
    quality_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Async Celery task for relocation optimization.
    
    This task runs the full optimization logic in a worker process,
    bypassing App Runner's 120s timeout.
    
    Parameters
    ----------
    trace_id : str
        Unique identifier for this optimization run
    config_dict : dict
        Optimizer configuration as a dictionary
    sku_data : list[dict]
        SKU master data as list of dicts
    inv_data : list[dict]
        Inventory data as list of dicts
    location_master_data : list[dict]
        Location master data as list of dicts
    block_codes : list[str], optional
        Block codes to filter
    quality_names : list[str], optional
        Quality names to filter
    
    Returns
    -------
    dict
        Optimization result with moves, summary, and rejection info
    """
    started_at = datetime.utcnow()
    logger.info(f"[relocation.run_async] Starting task for trace_id={trace_id}")
    
    # Update status to running
    store_relocation_status(trace_id, "running", {"phase": "starting"})
    
    try:
        # Convert data back to DataFrames
        sku_df = pd.DataFrame(sku_data) if sku_data else pd.DataFrame()
        inv_df = pd.DataFrame(inv_data) if inv_data else pd.DataFrame()
        location_master_df = pd.DataFrame(location_master_data) if location_master_data else pd.DataFrame()
        
        logger.info(f"[relocation.run_async] DataFrames: sku={len(sku_df)}, inv={len(inv_df)}, loc={len(location_master_df)}")
        
        # Build OptimizerConfig from dict
        if OptimizerConfig is not None:
            try:
                cfg = OptimizerConfig()
            except TypeError:
                from types import SimpleNamespace
                cfg = SimpleNamespace()
        else:
            from types import SimpleNamespace
            cfg = SimpleNamespace()
        
        # Apply config values
        for key, value in config_dict.items():
            setattr(cfg, key, value)
        
        # Bind trace ID
        setattr(cfg, "trace_id", trace_id)
        try:
            start_drop_trace(trace_id)
        except Exception:
            try:
                bind_trace_id(trace_id)
            except Exception:
                pass
        
        # Update status
        store_relocation_status(trace_id, "running", {"phase": "optimizing"})
        
        # Run optimization
        logger.info(f"[relocation.run_async] Calling plan_relocation...")
        if plan_relocation is None:
            raise RuntimeError("plan_relocation not available")
        
        moves = plan_relocation(
            sku_master=sku_df,
            inventory=inv_df,
            cfg=cfg,
            block_filter=block_codes,
            quality_filter=quality_names,
            ai_col_hints={},
            loc_master=location_master_df,
        )
        
        logger.info(f"[relocation.run_async] plan_relocation returned {len(moves) if moves else 0} moves")
        
        # Get rejection debug info
        try:
            _rej = get_last_rejection_debug() or {}
            rej = {
                "planned": _rej.get("planned"),
                "accepted": _rej.get("accepted"),
                "rejections": _rej.get("rejections", {}),
                "examples": _rej.get("examples", {}),
            }
        except Exception:
            rej = {}
        
        # Normalize moves to list[dict]
        moves_dicts = []
        for m in (moves or []):
            if isinstance(m, dict):
                dict_copy = m.copy()
                if "from_loc" in dict_copy and dict_copy["from_loc"]:
                    dict_copy["from_loc"] = str(dict_copy["from_loc"]).split('.')[0].zfill(8)
                if "to_loc" in dict_copy and dict_copy["to_loc"]:
                    dict_copy["to_loc"] = str(dict_copy["to_loc"]).split('.')[0].zfill(8)
                moves_dicts.append(dict_copy)
            else:
                try:
                    from_loc_val = getattr(m, "from_loc", getattr(m, "from", None))
                    to_loc_val = getattr(m, "to_loc", getattr(m, "to", None))
                    if from_loc_val:
                        from_loc_val = str(from_loc_val).split('.')[0].zfill(8)
                    if to_loc_val:
                        to_loc_val = str(to_loc_val).split('.')[0].zfill(8)
                    
                    moves_dicts.append({
                        "sku_id": getattr(m, "sku_id", None),
                        "lot": getattr(m, "lot", None),
                        "qty": getattr(m, "qty", getattr(m, "qty_cases", None)),
                        "from_loc": from_loc_val,
                        "to_loc": to_loc_val,
                        "lot_date": getattr(m, "lot_date", None),
                        "distance": getattr(m, "distance", None),
                        "reason": getattr(m, "reason", None),
                    })
                except Exception:
                    pass
        
        # Build summary
        summary = _build_summary(inv_df, moves_dicts)
        
        # Cache moves in optimizer's cache too
        try:
            if moves_dicts:
                cache_moves(trace_id, moves_dicts)
        except Exception as e:
            logger.warning(f"Failed to cache moves: {e}")
        
        # Get summary report if available
        try:
            summary_report = get_last_summary_report()
        except Exception:
            summary_report = None
        
        finished_at = datetime.utcnow()
        duration_sec = (finished_at - started_at).total_seconds()
        
        result = {
            "status": "completed",
            "trace_id": trace_id,
            "moves": moves_dicts,
            "rejection_summary": rej,
            "summary": summary,
            "summary_report": summary_report,
            "count": len(moves_dicts),
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_sec": duration_sec,
            # Echo back config
            "max_moves": config_dict.get("max_moves"),
            "fill_rate": config_dict.get("fill_rate"),
            "block_codes": block_codes,
            "quality_names": quality_names,
        }
        
        # Store result in Redis
        store_relocation_result(trace_id, result, ttl=7200)  # 2 hours TTL
        store_relocation_status(trace_id, "completed", {"count": len(moves_dicts)})
        
        logger.info(f"[relocation.run_async] Completed: {len(moves_dicts)} moves in {duration_sec:.1f}s")
        
        return result
        
    except Exception as exc:
        logger.exception(f"[relocation.run_async] Failed: {exc}")
        
        error_result = {
            "status": "failed",
            "trace_id": trace_id,
            "error": str(exc),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
        }
        
        store_relocation_result(trace_id, error_result, ttl=3600)
        store_relocation_status(trace_id, "failed", {"error": str(exc)})
        
        # Don't re-raise to avoid Celery retry; the error is stored in result
        return error_result

    except SoftTimeLimitExceeded:
        # Handle soft time limit (9 minutes)
        logger.warning(f"[relocation.run_async] Task timed out for trace_id={trace_id}")
        
        error_result = {
            "status": "failed",
            "trace_id": trace_id,
            "error": "最適化がタイムアウトしました（9分）。データ量を減らすか、max_movesを小さくしてお試しください。",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
        }
        
        store_relocation_result(trace_id, error_result, ttl=3600)
        store_relocation_status(trace_id, "timeout", {"error": error_result["error"]})
        
        return error_result


def get_task_status(trace_id: str) -> dict:
    """
    Get the status of a relocation task.
    
    Returns status from Redis or Celery result backend.
    """
    # First check Redis status
    status = get_relocation_status(trace_id)
    if status:
        return status
    
    # Check if there's a completed result
    result = get_relocation_result(trace_id)
    if result:
        return {
            "status": result.get("status", "completed"),
            "updated_at": result.get("finished_at"),
        }
    
    # Check Celery task state
    try:
        task_result = AsyncResult(trace_id, app=celery_app)
        if task_result.state == states.PENDING:
            return {"status": "pending"}
        elif task_result.state == states.STARTED:
            return {"status": "running"}
        elif task_result.state == states.SUCCESS:
            return {"status": "completed"}
        elif task_result.state == states.FAILURE:
            return {"status": "failed", "error": str(task_result.result)}
        else:
            return {"status": task_result.state.lower()}
    except Exception:
        pass
    
    return {"status": "unknown"}


# Legacy task for backward compatibility
@celery_app.task(bind=True, name="relocation.start")
def run_relocation_task(
    self,
    *,
    job_id: str | None = None,
    block_codes: list[str] | None = None,
    max_moves: int = 500,
    fill_rate: float = 0.95,
    weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """
    Legacy task for backward compatibility.
    Deprecated: Use run_relocation_async instead.
    """
    logger.warning("Using deprecated run_relocation_task - migrate to run_relocation_async")
    return {
        "job_id": job_id,
        "moves_generated": 0,
        "message": "Deprecated: Use run_relocation_async",
    }
