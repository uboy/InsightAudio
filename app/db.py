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
        # Ensure directory exists and path is absolute
        if db_url.startswith("sqlite:///"):
            path = db_url.replace("sqlite:///", "", 1)
            abs_path = os.path.abspath(path)
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            # Use absolute path for stability
            db_url = f"sqlite:///{abs_path}"
        
    # Set pool timeout for SQLite
    pool_args = {}
    if db_url.startswith("sqlite"):
        pool_args["pool_timeout"] = 30  # 30 seconds timeout
    
    engine = create_engine(db_url, future=True, echo=False, connect_args=connect_args, **pool_args)
    
    # Apply WAL mode and timeout settings for SQLite after engine creation
    if db_url.startswith("sqlite"):
        def set_sqlite_pragmas(dbapi_conn, connection_record):
            """Set SQLite PRAGMAs for better concurrency and reliability"""
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds timeout
            cursor.close()
        
        from sqlalchemy import event
        event.listen(engine, "connect", set_sqlite_pragmas)
        
        # Also set on initial connection
        try:
            with engine.connect() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.commit()
        except Exception:
            pass  # Ignore if already set
    
    return engine


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

