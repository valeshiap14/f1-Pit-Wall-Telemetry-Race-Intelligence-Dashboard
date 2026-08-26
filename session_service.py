from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.corners import compare_driver_corners
from src.data_loader import (
    get_driver_codes,
    get_fastest_lap_summary,
    load_session,
)
from src.database import (
    build_session_key,
    save_driver_lap_summary,
    save_model_results,
    save_session_metadata,
)
from src.insights import build_race_engineer_report
from src.predictions import (
    build_model_comparison_table,
    train_lap_time_models,
)
from src.strategy import compare_driver_strategies
from src.telemetry import compare_fastest_laps
from src.track_evolution import analyse_track_evolution
from src.tyres import analyse_driver_stint


@dataclass
class SessionContext:
    """
    Store the essential metadata for one loaded F1 session.
    """

    season: int
    event_name: str
    session_type: str
    session: Any
    session_key: str
    circuit: str
    country: str


def load_session_context(
    season: int,
    event_name: str,
    session_type: str,
) -> SessionContext:
    """
    Load a FastF1 session and register it in the local database.
    """

    session = load_session(
        season,
        event_name,
        session_type,
    )

    circuit = str(
        session.event.get(
            "Location",
            "Unknown",
        )
    )

    country = str(
        session.event.get(
            "Country",
            "Unknown",
        )
    )

    session_key = save_session_metadata(
        season=season,
        event_name=event_name,
        session_type=session_type,
        circuit=circuit,
        country=country,
    )

    return SessionContext(
        season=season,
        event_name=event_name,
        session_type=session_type,
        session=session,
        session_key=session_key,
        circuit=circuit,
        country=country,
    )


def get_session_drivers(
    context: SessionContext,
) -> list[str]:
    """
    Return all available driver abbreviations.
    """

    return get_driver_codes(
        context.session
    )


def get_driver_summary(
    context: SessionContext,
    driver: str,
) -> dict:
    """
    Calculate and persist one driver's fastest-lap summary.
    """

    summary = get_fastest_lap_summary(
        context.session,
        driver,
    )

    save_driver_lap_summary(
        context.session_key,
        driver,
        summary,
    )

    return summary


def build_driver_comparison(
    context: SessionContext,
    driver_a: str,
    driver_b: str,
) -> dict:
    """
    Run the main telemetry comparison for two drivers.
    """

    summary_a = get_driver_summary(
        context,
        driver_a,
    )

    summary_b = get_driver_summary(
        context,
        driver_b,
    )

    telemetry = compare_fastest_laps(
        context.session,
        driver_a,
        driver_b,
    )

    return {
        "summary_a": summary_a,
        "summary_b": summary_b,
        "telemetry": telemetry,
    }


def build_corner_analysis(
    comparison: dict,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Calculate corner-level performance differences.
    """

    telemetry = comparison[
        "telemetry"
    ]

    return compare_driver_corners(
        telemetry["telemetry_a"],
        telemetry["telemetry_b"],
        telemetry["aligned"],
        driver_a,
        driver_b,
    )


def build_tyre_analysis(
    context: SessionContext,
    driver: str,
    stint_number: int,
) -> dict:
    """
    Run tyre degradation analysis for one selected stint.
    """

    return analyse_driver_stint(
        context.session,
        driver,
        stint_number,
    )


def build_track_analysis(
    context: SessionContext,
) -> dict:
    """
    Run the session-wide track evolution analysis.
    """

    return analyse_track_evolution(
        context.session
    )


def build_strategy_analysis(
    context: SessionContext,
    driver_a: str,
    driver_b: str,
) -> dict:
    """
    Compare stint structures and pit-stop effects.
    """

    return compare_driver_strategies(
        context.session,
        driver_a,
        driver_b,
    )


def build_prediction_analysis(
    context: SessionContext,
) -> dict:
    """
    Train and evaluate the lap-time prediction models.

    Evaluation results are also stored in SQLite.
    """

    training_result = train_lap_time_models(
        context.session
    )

    save_model_results(
        context.session_key,
        training_result["results"],
    )

    comparison_table = (
        build_model_comparison_table(
            training_result
        )
    )

    return {
        "training_result": training_result,
        "comparison_table": comparison_table,
    }


def build_full_engineer_report(
    driver_a: str,
    driver_b: str,
    comparison: dict,
    corner_table: pd.DataFrame | None = None,
    tyre_analysis: dict | None = None,
    track_analysis: dict | None = None,
    strategy_analysis: dict | None = None,
    prediction_analysis: dict | None = None,
) -> dict:
    """
    Combine all calculated analytics into the structured
    race-engineer report.
    """

    telemetry = comparison[
        "telemetry"
    ]

    training_result = None

    if prediction_analysis is not None:
        training_result = prediction_analysis.get(
            "training_result"
        )

    return build_race_engineer_report(
        driver_a=driver_a,
        driver_b=driver_b,
        summary_a=comparison[
            "summary_a"
        ],
        summary_b=comparison[
            "summary_b"
        ],
        metrics_a=telemetry[
            "metrics_a"
        ],
        metrics_b=telemetry[
            "metrics_b"
        ],
        corner_table=corner_table,
        tyre_analysis=tyre_analysis,
        track_analysis=track_analysis,
        strategy_analysis=strategy_analysis,
        training_result=training_result,
    )


def build_session_snapshot(
    context: SessionContext,
    driver_a: str,
    driver_b: str,
) -> dict:
    """
    Run the lightweight analytics needed immediately after
    a user selects two drivers.

    More expensive modules such as ML can be requested later.
    """

    comparison = build_driver_comparison(
        context,
        driver_a,
        driver_b,
    )

    corner_table = None
    track_analysis = None

    try:
        corner_table = build_corner_analysis(
            comparison,
            driver_a,
            driver_b,
        )
    except Exception:
        pass

    try:
        track_analysis = build_track_analysis(
            context
        )
    except Exception:
        pass

    report = build_full_engineer_report(
        driver_a=driver_a,
        driver_b=driver_b,
        comparison=comparison,
        corner_table=corner_table,
        track_analysis=track_analysis,
    )

    return {
        "context": context,
        "comparison": comparison,
        "corner_table": corner_table,
        "track_analysis": track_analysis,
        "report": report,
    }


def describe_context(
    context: SessionContext,
) -> dict:
    """
    Return dashboard-friendly metadata.
    """

    session_name = str(
        getattr(
            context.session,
            "name",
            context.session_type,
        )
    )

    return {
        "season": context.season,
        "event": context.event_name,
        "session": session_name,
        "circuit": context.circuit,
        "country": context.country,
        "session_key": context.session_key,
    }