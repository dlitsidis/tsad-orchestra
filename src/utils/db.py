"""Database utilities for reading time series from TimescaleDB."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()


def get_db_url() -> str:
    """Build PostgreSQL connection URL from environment variables.

    Returns:
        PostgreSQL connection URL string.

    Raises:
        ValueError: If required environment variables are missing.
    """
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")

    if not all([user, password, host, db]):
        raise ValueError(
            "Missing required environment variables: "
            "POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, POSTGRES_DB"
        )

    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def read_time_series(
    table_name: str,
    time_column: str = "time",
    value_column: str = "data",
) -> pd.DataFrame:
    """Read time series data from TimescaleDB.

    Args:
        table_name: Name of the hypertable or regular table in TimescaleDB.
        time_column: Name of the time column (default: 'time').
        value_column: Name of the value column (default: 'data').

    Returns:
        DataFrame with time series data sorted by time.

    Raises:
        ValueError: If database connection fails or query is invalid.
        Exception: If table does not exist or other database errors occur.
    """
    try:
        db_url = get_db_url()
        engine = create_engine(db_url)

        # Build the query with quoted table name (handles table names starting with numbers)
        query = f'SELECT {time_column}, {value_column} FROM "{table_name}"'

        query += f" ORDER BY {time_column} ASC"

        # Execute query and read into DataFrame
        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)

        if df.empty:
            raise ValueError(
                f"No data found in table '{table_name}' with query: {query}"
            )

        return df

    except ValueError as e:
        raise ValueError(f"Database configuration error: {e}") from e
    except Exception as e:
        raise Exception(f"Failed to read time series from TimescaleDB: {e}") from e


def read_time_series_by_id(
    series_id: str,
    time_column: str = "time",
    value_column: str = "data",
) -> list[float]:
    """Read a time series by ID from TimescaleDB.

    Searches for a table matching the pattern '{series_id}_*' and reads from it.

    Args:
        series_id: Series ID (e.g., '1' will match table '1_facility', '1_webservice', etc.).
        time_column: Name of the time column (default: 'time').
        value_column: Name of the value column (default: 'data').

    Returns:
        List of float values sorted by time.

    Raises:
        ValueError: If no matching table found or database error occurs.
        Exception: If query fails.
    """
    try:
        table_name = get_time_series_name(series_id)

        return read_time_series(
            table_name=table_name
        )

    except ValueError as e:
        raise ValueError(f"Failed to read time series by ID '{series_id}': {e}") from e
    except Exception as e:
        raise Exception(f"Failed to read time series by ID '{series_id}': {e}") from e


def get_time_series_name(
    series_id: str
) -> pd.DataFrame:
    """Read a specific time series by name from TimescaleDB.

    Args:
        table_name: Name of the table in TimescaleDB.
        time_column: Name of the time column.
        value_column: Name of the value column (default: 'data').

    Returns:
        DataFrame with time series data for the specified series.

    Raises:
        ValueError: If series not found or database error occurs.
    """
    try:
        # Get all tables
        db_url = get_db_url()
        engine = create_engine(db_url)

        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """

        with engine.connect() as connection:
            result = connection.execute(text(query))
            tables = [row[0] for row in result]

        # Find table matching the ID pattern
        matching_tables = [t for t in tables if t.startswith(f"{series_id}_")]

        if not matching_tables:
            raise ValueError(
                f"No table found matching pattern '{series_id}_*'. "
                # f"Available tables: {', '.join(tables)}"
            )

        # Use the first matching table
        table_name = matching_tables[0]

        return table_name
    except ValueError as e:
        raise ValueError(f"Failed to read time series by ID '{series_id}': {e}") from e
    except Exception as e:
        raise Exception(f"Failed to read time series by ID '{series_id}': {e}") from e




def list_tables() -> list[str]:
    """List all tables in the database.

    Returns:
        List of table names.

    Raises:
        ValueError: If database connection fails.
    """
    try:
        db_url = get_db_url()
        engine = create_engine(db_url)

        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
        """

        with engine.connect() as connection:
            result = connection.execute(text(query))
            tables = [row[0] for row in result]

        return tables

    except ValueError as e:
        raise ValueError(f"Database configuration error: {e}") from e
    except Exception as e:
        raise Exception(f"Failed to list tables: {e}") from e


def get_table_schema(table_name: str) -> pd.DataFrame:
    """Get schema information for a specific table.

    Args:
        table_name: Name of the table.

    Returns:
        DataFrame with column information (column_name, data_type, etc.).

    Raises:
        ValueError: If database connection fails.
    """
    try:
        db_url = get_db_url()
        engine = create_engine(db_url)

        query = f"""
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = '{table_name}' AND table_schema = 'public'
        ORDER BY ordinal_position
        """

        with engine.connect() as connection:
            df = pd.read_sql(text(query), connection)

        if df.empty:
            raise ValueError(f"Table '{table_name}' not found or has no columns")

        return df

    except ValueError as e:
        raise ValueError(f"Database configuration error: {e}") from e
    except Exception as e:
        raise Exception(f"Failed to get table schema: {e}") from e