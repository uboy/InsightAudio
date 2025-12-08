import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app import config_manager
from app.db import Base, engine, session_scope
from app.db_models import Job, User


def ensure_tables() -> None:
    Base.metadata.create_all(bind=engine)


def get_or_create_user(session: Session, session_id: str) -> User:
    user = session.execute(select(User).where(User.session_id == session_id)).scalar_one_or_none()
    if user:
        user.last_seen_at = datetime.utcnow()
        session.commit()
        return user
    user = User(session_id=session_id)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def create_job(
    session: Session,
    user_id: str,
    job_type: str,
    stage: str,
    params: Dict,
    status: str = "queued",
    job_id: Optional[str] = None,
) -> Job:
    job = Job(
        id=job_id or str(uuid.uuid4()),
        user_id=user_id,
        job_type=job_type,
        stage=stage,
        status=status,
        progress=0,
        params_json=params,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def update_job(
    session: Session,
    job_id: str,
    *,
    status: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    eta_seconds: Optional[int] = None,
    error_message: Optional[str] = None,
    error_traceback: Optional[str] = None,
    result_manifest: Optional[List[Dict]] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
) -> None:
    values = {"updated_at": datetime.utcnow()}
    if status is not None:
        values["status"] = status
    if stage is not None:
        values["stage"] = stage
    if progress is not None:
        values["progress"] = max(0, min(100, int(progress)))
    if eta_seconds is not None:
        values["eta_seconds"] = eta_seconds
    if error_message is not None:
        values["error_message"] = error_message
    if error_traceback is not None:
        values["error_traceback"] = error_traceback
    if result_manifest is not None:
        values["result_manifest_json"] = result_manifest
    if started_at is not None:
        values["started_at"] = started_at
    if finished_at is not None:
        values["finished_at"] = finished_at
    session.execute(update(Job).where(Job.id == job_id).values(**values))
    session.commit()


def build_user_job_dir(user_id: str, job_id: str) -> str:
    cfg = config_manager.get_config()
    results_dir = os.path.abspath(cfg.get("RESULTS_DIR", "/tmp"))
    path = os.path.join(results_dir, user_id, job_id)
    os.makedirs(path, exist_ok=True)
    return path


def prune_expired_jobs(ttl_days: int = 14) -> int:
    """
    Удаляет jobs и файлы старше TTL.
    Также удаляет связанные кэши при необходимости.
    Returns: количество удаленных jobs.
    """
    import logging
    logger = logging.getLogger("insightaudio.job_service")
    
    cfg = config_manager.get_config()
    results_dir = os.path.abspath(cfg.get("RESULTS_DIR", "/tmp"))
    cutoff = datetime.utcnow() - timedelta(days=ttl_days)
    removed = 0
    total_size_freed = 0
    
    with session_scope() as session:
        old_jobs = session.execute(select(Job).where(Job.created_at < cutoff)).scalars().all()
        logger.info("Найдено %d устаревших jobs для удаления (старше %d дней)", len(old_jobs), ttl_days)
        
        for job in old_jobs:
            job_dir = os.path.join(results_dir, job.user_id, job.id)
            job_size = 0
            
            if os.path.exists(job_dir):
                try:
                    # Вычисляем размер перед удалением
                    for root, _, files in os.walk(job_dir):
                        for f in files:
                            try:
                                file_path = os.path.join(root, f)
                                job_size += os.path.getsize(file_path)
                            except OSError:
                                pass
                    
                    # Удаляем файлы и директории
                    import shutil
                    shutil.rmtree(job_dir, ignore_errors=True)
                    total_size_freed += job_size
                except Exception as e:
                    logger.warning("Ошибка при удалении job_dir %s: %s", job_dir, e)
            
            session.delete(job)
            removed += 1
        
        session.commit()
    
    if removed > 0:
        logger.info(
            "Удалено %d устаревших jobs, освобождено ~%.2f МБ",
            removed,
            total_size_freed / (1024 * 1024),
        )
    
    return removed

