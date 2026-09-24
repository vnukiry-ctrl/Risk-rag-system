import os

from sqlmodel import Session, SQLModel, create_engine

# DECISION (UNIVERSAL): DATABASE_URL is a standard Postgres connection string
# (works unchanged against Supabase, RDS, a local instance, etc.) -- same
# swappable-provider pattern already used for the LLM/embeddings config.
DATABASE_URL = os.getenv("DATABASE_URL", "")

engine = create_engine(DATABASE_URL, echo=False) if DATABASE_URL else None


def init_db() -> None:
    """Create any tables that don't exist yet. Safe to call on every
    startup -- a no-op once the schema is already there. Alembic (see
    backend/alembic/) is for actual schema *changes* after this first
    version; this is just enough to get a fresh environment running."""
    if engine is None:
        raise RuntimeError("DATABASE_URL is not set -- see backend/.env.example")
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    return Session(engine)
