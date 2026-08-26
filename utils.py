from __future__ import annotations

from datetime import timedelta

import pandas as pd


def format_lap_time(
    seconds: float | None,
) -> str:
    """
    Convert seconds into Formula 1 timing format.

    Example:
    74.321 -> 1:14.321
    """

    if seconds is None:
        return "N/A"

    try:
        value = float(seconds)

    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(value):
        return "N/A"

    lap_time = timedelta(
        seconds=value
    )

    total_seconds = (
        lap_time.total_seconds()
    )

    minutes = int(
        total_seconds // 60
    )

    remaining_seconds = (
        total_seconds % 60
    )

    return (
        f"{minutes}:"
        f"{remaining_seconds:06.3f}"
    )


def format_temperature(
    value: float | None,
) -> str:
    """
    Format a temperature reading for dashboard output.
    """

    if value is None:
        return "N/A"

    try:
        temperature = float(value)

    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(temperature):
        return "N/A"

    return f"{temperature:.1f} °C"


def format_speed(
    value: float | None,
) -> str:
    """
    Format speed in kilometres per hour.
    """

    if value is None:
        return "N/A"

    try:
        speed = float(value)

    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(speed):
        return "N/A"

    return f"{speed:.0f} km/h"


def format_percentage(
    value: float | None,
    decimals: int = 1,
) -> str:
    """
    Format a percentage value consistently.
    """

    if value is None:
        return "N/A"

    try:
        percentage = float(value)

    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(percentage):
        return "N/A"

    return (
        f"{percentage:.{decimals}f}%"
    )


def format_seconds(
    value: float | None,
    decimals: int = 3,
    signed: bool = False,
) -> str:
    """
    Format a time difference in seconds.

    Signed output is useful for lap delta and degradation.
    """

    if value is None:
        return "N/A"

    try:
        seconds = float(value)

    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(seconds):
        return "N/A"

    sign = "+" if signed else ""

    return (
        f"{seconds:{sign}.{decimals}f} s"
    )


def format_degradation(
    value: float | None,
) -> str:
    """
    Format tyre degradation in seconds per lap.
    """

    if value is None:
        return "N/A"

    try:
        degradation = float(value)

    except (TypeError, ValueError):
        return "N/A"

    if pd.isna(degradation):
        return "N/A"

    return (
        f"{degradation:+.3f} s/lap"
    )


def safe_float(
    value,
) -> float | None:
    """
    Convert a value to float without raising an exception.
    """

    try:
        if pd.isna(value):
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def safe_int(
    value,
) -> int | None:
    """
    Convert a value to integer without raising an exception.
    """

    try:
        if pd.isna(value):
            return None

        return int(
            round(
                float(value)
            )
        )

    except (TypeError, ValueError):
        return None


def column_exists(
    dataframe: pd.DataFrame,
    column: str,
) -> bool:
    """
    Check whether a usable column exists in a dataframe.
    """

    return (
        dataframe is not None
        and not dataframe.empty
        and column in dataframe.columns
    )


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Return a numeric version of a dataframe column.

    Missing columns return an empty series instead of
    causing downstream analytical code to fail.
    """

    if not column_exists(
        dataframe,
        column,
    ):
        return pd.Series(
            dtype=float
        )

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


def describe_advantage(
    value_a: float | None,
    value_b: float | None,
    driver_a: str,
    driver_b: str,
    higher_is_better: bool = True,
) -> str:
    """
    Identify which driver has the advantage for a metric.

    This helper does not decide whether a metric is inherently
    meaningful; the caller supplies whether higher or lower
    values should be considered favourable.
    """

    a = safe_float(
        value_a
    )

    b = safe_float(
        value_b
    )

    if (
        a is None
        or b is None
    ):
        return "N/A"

    if a == b:
        return "Equal"

    if higher_is_better:
        return (
            driver_a
            if a > b
            else driver_b
        )

    return (
        driver_a
        if a < b
        else driver_b
    )


def round_dataframe(
    dataframe: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Round only numeric columns in a dataframe.

    This is useful for dashboard tables while preserving
    text columns such as driver names and tyre compounds.
    """

    result = dataframe.copy()

    numeric_columns = (
        result.select_dtypes(
            include="number"
        ).columns
    )

    result[
        numeric_columns
    ] = (
        result[
            numeric_columns
        ].round(
            decimals
        )
    )

    return result


def clean_display_value(
    value,
    fallback: str = "N/A",
):
    """
    Replace missing dashboard values with a readable fallback.
    """

    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback

    except (TypeError, ValueError):
        pass

    return value