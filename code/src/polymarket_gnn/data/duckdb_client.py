"""DuckDB client module for read-only connection to preprocessing.duckdb."""

from pathlib import Path
import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "preprocessing.duckdb"


def get_duckdb_connection(db_path: Path | str = DEFAULT_DB_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB database safely.
    
    Args:
        db_path: Path to DuckDB file.
        read_only: Open connection in read-only mode (default True).
        
    Returns:
        DuckDB connection instance.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB database not found at {path}")
    return duckdb.connect(str(path), read_only=read_only)


def execute_sql_file(con: duckdb.DuckDBPyConnection, sql_path: Path | str, params: dict | None = None) -> duckdb.DuckDBPyRelation:
    """Read and execute a SQL file on the DuckDB connection.
    
    Args:
        con: Active DuckDB connection.
        sql_path: Path to .sql file.
        params: Optional dictionary of query parameters.
        
    Returns:
        DuckDB relation object.
    """
    path = Path(sql_path)
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found at {path}")
    query = path.read_text(encoding="utf-8")
    if params:
        for key, val in params.items():
            query = query.replace(f"${key}", str(val))
    return con.sql(query)
