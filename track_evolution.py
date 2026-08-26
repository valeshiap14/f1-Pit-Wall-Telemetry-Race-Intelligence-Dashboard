from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def prepare_track_evolution_data(session) -> pd.DataFrame:
    """
    Build a session-wide dataset for track-evolution analysis.

    The goal is to relate lap pace to session progression
    and environmental conditions without claiming causation.
    """

    laps = session.laps.copy()

    if laps.empty:
        raise ValueError(
            "No lap data is available for this session."
        )

    required_columns = [
        "LapTime",
        "LapStartTime",
        "Driver",
        "Compound",
    ]

    missing = [
        column
        for column in required_columns
        if column not in laps.columns
    ]

    if missing:
        raise ValueError(
            "The session is missing required lap data."
        )

    laps["LapTimeSeconds"] = (
        laps["LapTime"]
        .dt.total_seconds()
    )

    laps["SessionTimeSeconds"] = (
        laps["LapStartTime"]
        .dt.total_seconds()
    )

    # Remove laps that cannot represent meaningful performance.
    clean = laps[
        laps["LapTimeSeconds"].notna()
        & laps["SessionTimeSeconds"].notna()
        & (laps["LapTimeSeconds"] > 0)
    ].copy()

    if "PitInTime" in clean.columns:
        clean = clean[
            clean["PitInTime"].isna()
        ]

    if "PitOutTime" in clean.columns:
        clean = clean[
            clean["PitOutTime"].isna()
        ]

    if "Deleted" in clean.columns:
        deleted = (
            clean["Deleted"]
            .fillna(False)
            .astype(bool)
        )

        clean = clean[
            ~deleted
        ]

    if clean.empty:
        raise ValueError(
            "No clean laps remain for track-evolution analysis."
        )

    return clean.reset_index(drop=True)


def attach_weather_to_laps(
    session,
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the nearest available weather reading to each lap.

    Weather is sampled less frequently than telemetry, so an
    as-of merge is more appropriate than expecting exact timestamps.
    """

    weather = getattr(
        session,
        "weather_data",
        None,
    )

    result = laps.copy()

    if weather is None or weather.empty:
        result["TrackTemp"] = np.nan
        result["AirTemp"] = np.nan
        result["Humidity"] = np.nan
        result["Rainfall"] = np.nan

        return result

    weather = weather.copy()

    if "Time" not in weather.columns:
        return result

    weather["SessionTimeSeconds"] = (
        weather["Time"]
        .dt.total_seconds()
    )

    weather = weather.sort_values(
        "SessionTimeSeconds"
    )

    result = result.sort_values(
        "SessionTimeSeconds"
    )

    weather_columns = [
        column
        for column in [
            "SessionTimeSeconds",
            "TrackTemp",
            "AirTemp",
            "Humidity",
            "Rainfall",
        ]
        if column in weather.columns
    ]

    merged = pd.merge_asof(
        result,
        weather[weather_columns],
        on="SessionTimeSeconds",
        direction="nearest",
    )

    return merged


def remove_session_outliers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove extreme session-wide lap-time anomalies using IQR.

    This is deliberately broad because track-evolution analysis
    should preserve genuine pace variation while excluding obvious
    non-representative laps.
    """

    clean = data.copy()

    lap_times = pd.to_numeric(
        clean["LapTimeSeconds"],
        errors="coerce",
    )

    valid = lap_times.dropna()

    if len(valid) < 8:
        return clean

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)

    iqr = q3 - q1

    if iqr <= 0:
        return clean

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return clean[
        lap_times.between(
            lower,
            upper,
        )
    ].copy()


def calculate_track_evolution_trend(
    data: pd.DataFrame,
) -> dict:
    """
    Estimate the session-wide relationship between lap time
    and session progression.

    A negative slope means the representative lap pace became
    quicker as the session progressed.
    """

    modelling_data = data[
        [
            "SessionTimeSeconds",
            "LapTimeSeconds",
        ]
    ].dropna()

    if len(modelling_data) < 5:
        raise ValueError(
            "Not enough clean laps for track-evolution analysis."
        )

    x = modelling_data[
        "SessionTimeSeconds"
    ].to_numpy(dtype=float)

    y = modelling_data[
        "LapTimeSeconds"
    ].to_numpy(dtype=float)

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    predicted = (
        intercept
        + slope * x
    )

    seconds_per_10_minutes = (
        slope * 600
    )

    return {
        "slope_seconds_per_second": float(slope),
        "seconds_per_10_minutes": float(
            seconds_per_10_minutes
        ),
        "intercept": float(intercept),
        "prediction": predicted,
    }


def safe_correlation(
    data: pd.DataFrame,
    x_column: str,
    y_column: str = "LapTimeSeconds",
) -> dict | None:
    """
    Calculate a Pearson correlation only when sufficient
    usable data exists.

    Correlation is descriptive and should not be interpreted
    as proof that one variable caused another.
    """

    subset = data[
        [
            x_column,
            y_column,
        ]
    ].dropna()

    if len(subset) < 5:
        return None

    if (
        subset[x_column].nunique() < 2
        or subset[y_column].nunique() < 2
    ):
        return None

    correlation, p_value = pearsonr(
        subset[x_column],
        subset[y_column],
    )

    return {
        "correlation": float(correlation),
        "p_value": float(p_value),
        "sample_size": int(len(subset)),
    }


def analyse_track_evolution(
    session,
) -> dict:
    """
    Run the complete track-evolution analysis pipeline.
    """

    data = prepare_track_evolution_data(
        session
    )

    data = attach_weather_to_laps(
        session,
        data,
    )

    data = remove_session_outliers(
        data
    )

    trend = calculate_track_evolution_trend(
        data
    )

    correlations = {
        "track_temperature": safe_correlation(
            data,
            "TrackTemp",
        ),
        "air_temperature": safe_correlation(
            data,
            "AirTemp",
        ),
        "humidity": safe_correlation(
            data,
            "Humidity",
        ),
        "session_time": safe_correlation(
            data,
            "SessionTimeSeconds",
        ),
    }

    return {
        "data": data,
        "trend": trend,
        "correlations": correlations,
    }


def generate_track_evolution_insight(
    analysis: dict,
) -> str:
    """
    Convert the calculated trend into a cautious analytical summary.
    """

    trend = analysis["trend"]

    change = trend[
        "seconds_per_10_minutes"
    ]

    if abs(change) < 0.02:

        return (
            "Representative lap pace remained broadly stable "
            "as the session progressed. No strong session-wide "
            "track-evolution trend is visible in the cleaned data."
        )

    if change < 0:

        return (
            f"Representative lap pace improved by approximately "
            f"{abs(change):.3f} seconds per 10 minutes of session "
            f"time. This is consistent with a faster evolving track, "
            f"but fuel load, tyres, traffic and driver run plans may "
            f"also contribute to the trend."
        )

    return (
        f"Representative lap pace became approximately "
        f"{change:.3f} seconds slower per 10 minutes of session "
        f"time. This may reflect changing conditions, tyre behaviour, "
        f"traffic or run-plan effects rather than track grip alone."
    )