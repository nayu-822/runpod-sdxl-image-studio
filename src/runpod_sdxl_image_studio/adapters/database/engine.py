"""SQLAlchemy engine and session construction without import-time side effects."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from runpod_sdxl_image_studio.config import Settings


def resolved_database_url(settings: Settings) -> str:
    if settings.database_url and settings.database_url.strip():
        return settings.database_url
    return f"sqlite:///{(settings.data_dir / 'database' / 'image_studio.sqlite3').as_posix()}"


def create_image_studio_engine(settings: Settings) -> Engine:
    url = resolved_database_url(settings)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(url, future=True, connect_args=connect_args)
    if url.startswith("sqlite"):
        _enable_sqlite_foreign_keys(engine)
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def ensure_database_directory(settings: Settings) -> Path:
    if settings.database_url and settings.database_url.strip():
        parsed = make_url(settings.database_url)
        if (
            parsed.drivername.startswith("sqlite")
            and parsed.database is not None
            and parsed.database != ":memory:"
        ):
            database_path = Path(parsed.database)
            database_path.parent.mkdir(parents=True, exist_ok=True)
            return database_path.parent
        return settings.data_dir / "database"
    database_dir = settings.data_dir / "database"
    database_dir.mkdir(parents=True, exist_ok=True)
    return database_dir


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def set_foreign_keys(dbapi_connection: object, connection_record: object) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
