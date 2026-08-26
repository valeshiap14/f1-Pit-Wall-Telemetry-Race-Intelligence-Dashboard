from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import DATA_DIR


DATABASE_PATH = DATA_DIR / "f1_pitwall.db"


def get_connection(
    database_path: Path = DATABASE_PATH,
) -> sqlite3.Connection:
    """
    Open a SQLite connection.

    Row access by column name makes later debugging and
    inspection easier than relying only on positional indexes.
    """

    connection = sqlite3.connect(
        database_path
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database() -> None:
    """
    Create the tables used by the local processed-data cache.

    The database stores processed session metadata and
    analytical results, not FastF1's raw cache files.
    """

    with closing(
        get_connection()
    ) as connection:

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_key TEXT PRIMARY KEY,
                season INTEGER NOT NULL,
                event_name TEXT NOT NULL,
                session_type TEXT NOT NULL,
                circuit TEXT,
                country TEXT,
                loaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS driver_lap_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                driver TEXT NOT NULL,
                lap_number INTEGER,
                lap_time_seconds REAL,
                compound TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_key, driver),
                FOREIGN KEY(session_key)
                    REFERENCES sessions(session_key)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tyre_stint_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                driver TEXT NOT NULL,
                stint_number INTEGER NOT NULL,
                compound TEXT,
                clean_laps INTEGER,
                best_lap REAL,
                average_lap REAL,
                degradation_seconds_per_lap REAL,
                mae REAL,
                rmse REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_key, driver, stint_number),
                FOREIGN KEY(session_key)
                    REFERENCES sessions(session_key)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                model_name TEXT NOT NULL,
                mae REAL,
                rmse REAL,
                r2 REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_key, model_name),
                FOREIGN KEY(session_key)
                    REFERENCES sessions(session_key)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_key TEXT NOT NULL,
                category TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(session_key)
                    REFERENCES sessions(session_key)
            )
            """
        )

        connection.commit()


def build_session_key(
    season: int,
    event_name: str,
    session_type: str,
) -> str:
    """
    Create a stable identifier for one F1 session.

    Example:
    2025|Monaco Grand Prix|Q
    """

    return (
        f"{season}|"
        f"{event_name.strip()}|"
        f"{session_type.strip().upper()}"
    )


def save_session_metadata(
    season: int,
    event_name: str,
    session_type: str,
    circuit: str | None = None,
    country: str | None = None,
) -> str:
    """
    Store or update metadata for a processed session.
    """

    initialize_database()

    session_key = build_session_key(
        season,
        event_name,
        session_type,
    )

    with closing(
        get_connection()
    ) as connection:

        connection.execute(
            """
            INSERT INTO sessions (
                session_key,
                season,
                event_name,
                session_type,
                circuit,
                country
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_key)
            DO UPDATE SET
                circuit = excluded.circuit,
                country = excluded.country
            """,
            (
                session_key,
                season,
                event_name,
                session_type,
                circuit,
                country,
            ),
        )

        connection.commit()

    return session_key


def save_driver_lap_summary(
    session_key: str,
    driver: str,
    summary: dict[str, Any],
) -> None:
    """
    Cache a processed fastest-lap summary for one driver.
    """

    initialize_database()

    with closing(
        get_connection()
    ) as connection:

        connection.execute(
            """
            INSERT INTO driver_lap_summaries (
                session_key,
                driver,
                lap_number,
                lap_time_seconds,
                compound
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(session_key, driver)
            DO UPDATE SET
                lap_number = excluded.lap_number,
                lap_time_seconds = excluded.lap_time_seconds,
                compound = excluded.compound
            """,
            (
                session_key,
                driver,
                summary.get("lap_number"),
                summary.get("lap_time_seconds"),
                summary.get("compound"),
            ),
        )

        connection.commit()


def load_driver_lap_summary(
    session_key: str,
    driver: str,
) -> dict | None:
    """
    Retrieve a cached fastest-lap summary.
    """

    initialize_database()

    with closing(
        get_connection()
    ) as connection:

        row = connection.execute(
            """
            SELECT
                driver,
                lap_number,
                lap_time_seconds,
                compound
            FROM driver_lap_summaries
            WHERE session_key = ?
              AND driver = ?
            """,
            (
                session_key,
                driver,
            ),
        ).fetchone()

    if row is None:
        return None

    return dict(row)


def save_tyre_stint_result(
    session_key: str,
    driver: str,
    stint_number: int,
    tyre_analysis: dict,
) -> None:
    """
    Store processed tyre-stint metrics.
    """

    initialize_database()

    summary = tyre_analysis.get(
        "summary",
        {}
    )

    model = tyre_analysis.get(
        "model"
    )

    degradation = None
    mae = None
    rmse = None

    if model is not None:
        degradation = model.get(
            "seconds_per_lap"
        )
        mae = model.get("mae")
        rmse = model.get("rmse")

    with closing(
        get_connection()
    ) as connection:

        connection.execute(
            """
            INSERT INTO tyre_stint_results (
                session_key,
                driver,
                stint_number,
                compound,
                clean_laps,
                best_lap,
                average_lap,
                degradation_seconds_per_lap,
                mae,
                rmse
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                session_key,
                driver,
                stint_number
            )
            DO UPDATE SET
                compound = excluded.compound,
                clean_laps = excluded.clean_laps,
                best_lap = excluded.best_lap,
                average_lap = excluded.average_lap,
                degradation_seconds_per_lap =
                    excluded.degradation_seconds_per_lap,
                mae = excluded.mae,
                rmse = excluded.rmse
            """,
            (
                session_key,
                driver,
                stint_number,
                summary.get("compound"),
                summary.get("clean_laps"),
                summary.get("best_lap"),
                summary.get("average_lap"),
                degradation,
                mae,
                rmse,
            ),
        )

        connection.commit()


def save_model_results(
    session_key: str,
    model_results,
) -> None:
    """
    Store evaluation metrics for trained predictive models.
    """

    initialize_database()

    with closing(
        get_connection()
    ) as connection:

        for result in model_results:

            connection.execute(
                """
                INSERT INTO model_results (
                    session_key,
                    model_name,
                    mae,
                    rmse,
                    r2
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_key, model_name)
                DO UPDATE SET
                    mae = excluded.mae,
                    rmse = excluded.rmse,
                    r2 = excluded.r2
                """,
                (
                    session_key,
                    result.name,
                    result.mae,
                    result.rmse,
                    result.r2,
                ),
            )

        connection.commit()


def save_insight_report(
    session_key: str,
    category: str,
    insights: list[str],
) -> None:
    """
    Save generated analytical insights as JSON.

    JSON keeps the database structure simple while preserving
    each insight as a separate item.
    """

    initialize_database()

    payload = json.dumps(
        insights
    )

    with closing(
        get_connection()
    ) as connection:

        connection.execute(
            """
            INSERT INTO insights (
                session_key,
                category,
                payload
            )
            VALUES (?, ?, ?)
            """,
            (
                session_key,
                category,
                payload,
            ),
        )

        connection.commit()


def read_table(
    table_name: str,
) -> pd.DataFrame:
    """
    Read one internal database table into Pandas.

    Only known table names are accepted so arbitrary SQL
    cannot be passed through this helper.
    """

    allowed_tables = {
        "sessions",
        "driver_lap_summaries",
        "tyre_stint_results",
        "model_results",
        "insights",
    }

    if table_name not in allowed_tables:
        raise ValueError(
            f"Unsupported table: {table_name}"
        )

    initialize_database()

    with closing(
        get_connection()
    ) as connection:

        return pd.read_sql_query(
            f"SELECT * FROM {table_name}",
            connection,
        )


def delete_session_cache(
    session_key: str,
) -> None:
    """
    Remove processed results for one session.

    FastF1's separate raw cache is not affected.
    """

    initialize_database()

    with closing(
        get_connection()
    ) as connection:

        tables = [
            "driver_lap_summaries",
            "tyre_stint_results",
            "model_results",
            "insights",
        ]

        for table in tables:

            connection.execute(
                f"""
                DELETE FROM {table}
                WHERE session_key = ?
                """,
                (
                    session_key,
                ),
            )

        connection.execute(
            """
            DELETE FROM sessions
            WHERE session_key = ?
            """,
            (
                session_key,
            ),
        )

        connection.commit()


if __name__ == "__main__":
    # Running this file directly creates the local database
    # and prints its location as a quick smoke test.
    initialize_database()

    print(
        f"Database ready: {DATABASE_PATH}"
    )