import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app import config_manager


def _build_engine():
    cfg = config_manager.get_config()
    db_url = cfg.get("DB_URL") or "sqlite:///../tmp/insightaudio.db"
    connect_args = {}
    if db_url.startswith("sqlite"):
        # SQLite needs check_same_thread disabled for Celery/FastAPI workers
        connect_args["check_same_thread"] = False
        # Ensure directory exists
        if db_url.startswith("sqlite:///"):
            path = db_url.replace("sqlite:///", "", 1)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return create_engine(db_url, future=True, echo=False, connect_args=connect_args)


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_session() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

