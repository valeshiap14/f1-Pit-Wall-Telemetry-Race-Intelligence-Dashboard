from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_float(value: Any) -> float | None:
    """
    Convert a value to float when possible.

    Returning None instead of raising an exception keeps the
    insight engine tolerant of incomplete historical telemetry.
    """

    try:
        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def _format_difference(
    value: float | None,
    unit: str,
    decimals: int = 1,
) -> str:
    """Format a signed numerical difference."""

    if value is None:
        return "N/A"

    return f"{value:+.{decimals}f} {unit}"


def build_driver_performance_summary(
    driver_a: str,
    driver_b: str,
    summary_a: dict,
    summary_b: dict,
    metrics_a: dict,
    metrics_b: dict,
) -> list[str]:
    """
    Generate defensible high-level driver comparison insights.
    """

    insights: list[str] = []

    lap_a = _safe_float(
        summary_a.get(
            "lap_time_seconds"
        )
    )

    lap_b = _safe_float(
        summary_b.get(
            "lap_time_seconds"
        )
    )

    if (
        lap_a is not None
        and lap_b is not None
    ):

        gap = abs(
            lap_a - lap_b
        )

        faster_driver = (
            driver_a
            if lap_a < lap_b
            else driver_b
        )

        insights.append(
            f"{faster_driver} produced the faster selected lap "
            f"by approximately {gap:.3f} seconds."
        )

    speed_a = _safe_float(
        metrics_a.get(
            "maximum_speed"
        )
    )

    speed_b = _safe_float(
        metrics_b.get(
            "maximum_speed"
        )
    )

    if (
        speed_a is not None
        and speed_b is not None
    ):

        difference = speed_a - speed_b

        if abs(difference) >= 1:

            faster_driver = (
                driver_a
                if difference > 0
                else driver_b
            )

            insights.append(
                f"{faster_driver} recorded approximately "
                f"{abs(difference):.0f} km/h more peak speed."
            )

    throttle_a = _safe_float(
        metrics_a.get(
            "full_throttle_percentage"
        )
    )

    throttle_b = _safe_float(
        metrics_b.get(
            "full_throttle_percentage"
        )
    )

    if (
        throttle_a is not None
        and throttle_b is not None
    ):

        difference = (
            throttle_a - throttle_b
        )

        if abs(difference) >= 0.5:

            greater_driver = (
                driver_a
                if difference > 0
                else driver_b
            )

            insights.append(
                f"{greater_driver} spent approximately "
                f"{abs(difference):.1f} percentage points more "
                f"of the lap at full throttle."
            )

    braking_a = metrics_a.get(
        "braking_events"
    )

    braking_b = metrics_b.get(
        "braking_events"
    )

    if (
        braking_a is not None
        and braking_b is not None
        and braking_a != braking_b
    ):

        insights.append(
            f"{driver_a} registered {braking_a} braking events "
            f"versus {braking_b} for {driver_b}."
        )

    return insights


def find_largest_corner_gain(
    corner_table: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> dict | None:
    """
    Find the corner region with the largest relative time change.
    """

    if (
        corner_table is None
        or corner_table.empty
        or "Corner Delta"
        not in corner_table.columns
    ):
        return None

    usable = corner_table[
        corner_table[
            "Corner Delta"
        ].notna()
    ].copy()

    if usable.empty:
        return None

    usable[
        "AbsoluteDelta"
    ] = (
        usable[
            "Corner Delta"
        ].abs()
    )

    row = usable.loc[
        usable[
            "AbsoluteDelta"
        ].idxmax()
    ]

    delta = float(
        row[
            "Corner Delta"
        ]
    )

    gaining_driver = (
        driver_b
        if delta > 0
        else driver_a
    )

    return {
        "corner": int(
            row["Corner"]
        ),
        "delta": abs(delta),
        "driver": gaining_driver,
        "row": row,
    }


def build_corner_summary(
    corner_table: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> list[str]:
    """
    Summarize the largest cornering differences.
    """

    insights: list[str] = []

    largest_gain = find_largest_corner_gain(
        corner_table,
        driver_a,
        driver_b,
    )

    if largest_gain is None:
        return insights

    corner = largest_gain[
        "corner"
    ]

    delta = largest_gain[
        "delta"
    ]

    gaining_driver = largest_gain[
        "driver"
    ]

    row = largest_gain[
        "row"
    ]

    insights.append(
        f"The largest detected corner-region advantage was "
        f"approximately {delta:.3f} seconds for {gaining_driver} "
        f"in region {corner}."
    )

    minimum_a = _safe_float(
        row.get(
            f"{driver_a} Minimum"
        )
    )

    minimum_b = _safe_float(
        row.get(
            f"{driver_b} Minimum"
        )
    )

    if (
        minimum_a is not None
        and minimum_b is not None
    ):

        difference = (
            minimum_a - minimum_b
        )

        if abs(difference) >= 1:

            faster_driver = (
                driver_a
                if difference > 0
                else driver_b
            )

            insights.append(
                f"In that region, {faster_driver} carried "
                f"approximately {abs(difference):.0f} km/h more "
                f"minimum speed."
            )

    brake_a = _safe_float(
        row.get(
            f"{driver_a} Brake"
        )
    )

    brake_b = _safe_float(
        row.get(
            f"{driver_b} Brake"
        )
    )

    if (
        brake_a is not None
        and brake_b is not None
    ):

        difference = (
            brake_a - brake_b
        )

        if abs(difference) >= 2:

            later_driver = (
                driver_a
                if difference > 0
                else driver_b
            )

            insights.append(
                f"{later_driver} began braking approximately "
                f"{abs(difference):.0f} metres later in that "
                f"analytical region."
            )

    throttle_a = _safe_float(
        row.get(
            f"{driver_a} Full Throttle"
        )
    )

    throttle_b = _safe_float(
        row.get(
            f"{driver_b} Full Throttle"
        )
    )

    if (
        throttle_a is not None
        and throttle_b is not None
    ):

        difference = (
            throttle_a - throttle_b
        )

        if abs(difference) >= 2:

            earlier_driver = (
                driver_a
                if difference < 0
                else driver_b
            )

            insights.append(
                f"{earlier_driver} returned to full throttle "
                f"approximately {abs(difference):.0f} metres earlier."
            )

    return insights


def build_tyre_summary(
    tyre_analysis: dict | None,
) -> list[str]:
    """
    Generate tyre-performance observations from one stint analysis.
    """

    if not tyre_analysis:
        return []

    driver = tyre_analysis.get(
        "driver",
        "Driver",
    )

    summary = tyre_analysis.get(
        "summary",
        {}
    )

    model = tyre_analysis.get(
        "model"
    )

    insights: list[str] = []

    compound = summary.get(
        "compound",
        "Unknown",
    )

    clean_laps = summary.get(
        "clean_laps"
    )

    if clean_laps is not None:

        insights.append(
            f"{driver}'s analysed {compound} stint contains "
            f"{clean_laps} clean representative laps."
        )

    if model is None:

        insights.append(
            "There were not enough clean laps to produce a "
            "reliable linear degradation estimate."
        )

        return insights

    degradation = _safe_float(
        model.get(
            "seconds_per_lap"
        )
    )

    if degradation is not None:

        if degradation > 0:

            insights.append(
                f"The fitted stint trend indicates approximately "
                f"{degradation:.3f} seconds of additional lap time "
                f"per tyre lap."
            )

        elif degradation < 0:

            insights.append(
                f"The fitted trend improved by approximately "
                f"{abs(degradation):.3f} seconds per lap. "
                f"Fuel burn, track evolution or traffic effects may "
                f"therefore be outweighing tyre degradation."
            )

        else:

            insights.append(
                "The fitted stint trend is approximately flat."
            )

    mae = _safe_float(
        model.get(
            "mae"
        )
    )

    if mae is not None:

        insights.append(
            f"The degradation model's in-sample mean absolute "
            f"error is approximately {mae:.3f} seconds."
        )

    return insights


def build_track_evolution_summary(
    track_analysis: dict | None,
) -> list[str]:
    """
    Summarize session-wide pace evolution.
    """

    if not track_analysis:
        return []

    trend = track_analysis.get(
        "trend",
        {}
    )

    correlations = track_analysis.get(
        "correlations",
        {}
    )

    insights: list[str] = []

    change = _safe_float(
        trend.get(
            "seconds_per_10_minutes"
        )
    )

    if change is not None:

        if abs(change) < 0.02:

            insights.append(
                "Representative session pace remained broadly "
                "stable as the session progressed."
            )

        elif change < 0:

            insights.append(
                f"Representative lap pace improved by approximately "
                f"{abs(change):.3f} seconds per 10 minutes."
            )

        else:

            insights.append(
                f"Representative lap pace became approximately "
                f"{change:.3f} seconds slower per 10 minutes."
            )

    track_temp = correlations.get(
        "track_temperature"
    )

    if track_temp is not None:

        correlation = _safe_float(
            track_temp.get(
                "correlation"
            )
        )

        if correlation is not None:

            insights.append(
                f"Track temperature and lap time showed a Pearson "
                f"correlation of {correlation:+.2f} in the cleaned "
                f"sample. This is correlation, not evidence of causation."
            )

    return insights


def build_strategy_summary(
    strategy_analysis: dict | None,
) -> list[str]:
    """
    Summarize the existing strategy-analysis calculations.
    """

    if not strategy_analysis:
        return []

    from src.strategy import generate_strategy_insight

    return generate_strategy_insight(
        strategy_analysis
    )


def build_model_summary(
    training_result: dict | None,
) -> list[str]:
    """
    Summarize lap-time model evaluation results.
    """

    if not training_result:
        return []

    best_model = training_result.get(
        "best_model"
    )

    if best_model is None:
        return []

    insights = [
        (
            f"{best_model.name} produced the lowest held-out "
            f"MAE at approximately {best_model.mae:.3f} seconds."
        ),
        (
            f"Its held-out RMSE was approximately "
            f"{best_model.rmse:.3f} seconds."
        ),
    ]

    if pd.notna(
        best_model.r2
    ):

        insights.append(
            f"The model's held-out R² was "
            f"{best_model.r2:.3f}."
        )

    return insights


def build_race_engineer_report(
    driver_a: str,
    driver_b: str,
    summary_a: dict,
    summary_b: dict,
    metrics_a: dict,
    metrics_b: dict,
    corner_table: pd.DataFrame | None = None,
    tyre_analysis: dict | None = None,
    track_analysis: dict | None = None,
    strategy_analysis: dict | None = None,
    training_result: dict | None = None,
) -> dict:
    """
    Build a structured race-engineer report from calculated data.

    No observation is generated unless the corresponding
    analytical result exists.
    """

    performance = (
        build_driver_performance_summary(
            driver_a,
            driver_b,
            summary_a,
            summary_b,
            metrics_a,
            metrics_b,
        )
    )

    corners = build_corner_summary(
        corner_table,
        driver_a,
        driver_b,
    )

    tyres = build_tyre_summary(
        tyre_analysis
    )

    track = build_track_evolution_summary(
        track_analysis
    )

    strategy = build_strategy_summary(
        strategy_analysis
    )

    models = build_model_summary(
        training_result
    )

    return {
        "performance": performance,
        "corners": corners,
        "tyres": tyres,
        "track_evolution": track,
        "strategy": strategy,
        "predictive_models": models,
    }


def flatten_report(
    report: dict,
) -> list[str]:
    """
    Flatten a structured report into one ordered list of insights.
    """

    ordered_sections = [
        "performance",
        "corners",
        "tyres",
        "track_evolution",
        "strategy",
        "predictive_models",
    ]

    output: list[str] = []

    for section in ordered_sections:

        items = report.get(
            section,
            []
        )

        output.extend(
            items
        )

    return output