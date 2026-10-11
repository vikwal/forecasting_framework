"""
Optuna databases on the PostgreSQL server of OPTUNA_STORAGE, for the database picker of
optuna_dashboard.py. Kept outside utils/ so the dashboard does not import torch via utils/__init__.

A database counts as an Optuna database when its current schema holds the Optuna tables
(studies, trials, version_info). The server, user and password always come from the given URL;
only the database name changes.
"""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import NullPool

OPTUNA_TABLES = {'studies', 'trials', 'version_info'}


def database_name(url: str) -> str:
    """Database name of a storage URL."""
    return make_url(url).database


def with_database(url: str, db: str) -> str:
    """Same server and credentials, other database."""
    return make_url(url).set(database=db).render_as_string(hide_password=False)


def _query(url: str, sql: str) -> list:
    engine = create_engine(url, poolclass=NullPool, connect_args={'connect_timeout': 5})
    try:
        with engine.connect() as conn:
            return [row[0] for row in conn.execute(text(sql))]
    finally:
        engine.dispose()


def is_optuna_db(url: str) -> bool:
    """True if the database holds the Optuna schema (False if unreachable or not permitted)."""
    try:
        tables = _query(url, "SELECT table_name FROM information_schema.tables "
                             "WHERE table_schema = current_schema()")
    except SQLAlchemyError:
        return False
    return OPTUNA_TABLES <= set(tables)


def list_optuna_databases(url: str) -> list:
    """Sorted names of the Optuna databases on the server of a PostgreSQL URL; always includes the
    database of the URL itself. Non-PostgreSQL URLs (SQLite) and unlistable servers -> just that one."""
    own = database_name(url)
    if not url.startswith('postgresql'):
        return [own]
    try:
        names = _query(with_database(url, 'postgres'), "SELECT datname FROM pg_database "
                                                       "WHERE datallowconn AND NOT datistemplate")
    except SQLAlchemyError:
        return [own]
    found = {db for db in names if db != 'postgres' and is_optuna_db(with_database(url, db))}
    return sorted(found | {own})
