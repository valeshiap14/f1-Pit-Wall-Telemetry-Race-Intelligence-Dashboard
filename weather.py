from __future__ import annotations

import numpy as np
import pandas as pd


def get_weather_data(session) -> pd.DataFrame:
    """
    Return a clean copy of FastF1 weather data.

    Some older sessions may have incomplete weather records,
    so this helper always returns a dataframe.
    """

    weather = getattr(
        session,
        "weather_data",
        None,
    )

    if weather is None:
        return pd.DataFrame()

    return weather.copy()


def _median_numeric(
    weather: pd.DataFrame,
    column: str,
) -> float | None:
    """
    Return the median numeric value of a weather channel.
    """

    if (
        weather.empty
        or column not in weather.columns
    ):
        return None

    values = pd.to_numeric(
        weather[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.median()
    )


def get_weather_summary(
    session,
) -> dict:
    """
    Calculate representative conditions for a session.

    Median values are used because one isolated reading may
    not accurately represent the overall session.
    """

    weather = get_weather_data(
        session
    )

    if weather.empty:
        return {
            "air_temperature": None,
            "track_temperature": None,
            "humidity": None,
            "pressure": None,
            "wind_speed": None,
            "wind_direction": None,
            "rainfall": None,
            "condition": "Unknown",
        }

    rainfall = None

    if "Rainfall" in weather.columns:

        rain = (
            weather["Rainfall"]
            .dropna()
        )

        if not rain.empty:
            rainfall = bool(
                rain.astype(bool).any()
            )

    if rainfall is True:
        condition = "Wet"

    elif rainfall is False:
        condition = "Dry"

    else:
        condition = "Unknown"

    return {
        "air_temperature": _median_numeric(
            weather,
            "AirTemp",
        ),
        "track_temperature": _median_numeric(
            weather,
            "TrackTemp",
        ),
        "humidity": _median_numeric(
            weather,
            "Humidity",
        ),
        "pressure": _median_numeric(
            weather,
            "Pressure",
        ),
        "wind_speed": _median_numeric(
            weather,
            "WindSpeed",
        ),
        "wind_direction": _median_numeric(
            weather,
            "WindDirection",
        ),
        "rainfall": rainfall,
        "condition": condition,
    }


def attach_weather_to_laps(
    session,
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the nearest weather sample to each lap.

    FastF1 weather data is sampled less frequently than lap data,
    so merge_asof is used instead of requiring identical timestamps.
    """

    result = laps.copy()

    weather = get_weather_data(
        session
    )

    if weather.empty:
        return result

    if (
        "LapStartTime"
        not in result.columns
        or "Time"
        not in weather.columns
    ):
        return result

    result["WeatherMergeTime"] = (
        result["LapStartTime"]
        .dt.total_seconds()
    )

    weather["WeatherMergeTime"] = (
        weather["Time"]
        .dt.total_seconds()
    )

    result = result.sort_values(
        "WeatherMergeTime"
    )

    weather = weather.sort_values(
        "WeatherMergeTime"
    )

    useful_weather_columns = [
        column
        for column in [
            "WeatherMergeTime",
            "AirTemp",
            "TrackTemp",
            "Humidity",
            "Pressure",
            "Rainfall",
            "WindSpeed",
            "WindDirection",
        ]
        if column in weather.columns
    ]

    merged = pd.merge_asof(
        result,
        weather[
            useful_weather_columns
        ],
        on="WeatherMergeTime",
        direction="nearest",
    )

    return merged


def calculate_weather_ranges(
    session,
) -> dict:
    """
    Calculate the observed range of important conditions.
    """

    weather = get_weather_data(
        session
    )

    def range_for(
        column: str,
    ) -> tuple[float, float] | None:

        if column not in weather.columns:
            return None

        values = pd.to_numeric(
            weather[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return None

        return (
            float(values.min()),
            float(values.max()),
        )

    return {
        "air_temperature": range_for(
            "AirTemp"
        ),
        "track_temperature": range_for(
            "TrackTemp"
        ),
        "humidity": range_for(
            "Humidity"
        ),
        "wind_speed": range_for(
            "WindSpeed"
        ),
    }


def detect_significant_weather_change(
    session,
    track_temp_threshold: float = 5.0,
    air_temp_threshold: float = 3.0,
) -> dict:
    """
    Detect whether conditions changed substantially.

    The thresholds are analytical flags rather than statements
    that weather caused any observed pace changes.
    """

    ranges = calculate_weather_ranges(
        session
    )

    track_change = None
    air_change = None

    track_range = ranges[
        "track_temperature"
    ]

    if track_range is not None:
        track_change = (
            track_range[1]
            - track_range[0]
        )

    air_range = ranges[
        "air_temperature"
    ]

    if air_range is not None:
        air_change = (
            air_range[1]
            - air_range[0]
        )

    significant_track_change = (
        track_change is not None
        and track_change
        >= track_temp_threshold
    )

    significant_air_change = (
        air_change is not None
        and air_change
        >= air_temp_threshold
    )

    weather = get_weather_data(
        session
    )

    rainfall_change = False

    if (
        not weather.empty
        and "Rainfall" in weather.columns
    ):

        rain = (
            weather["Rainfall"]
            .dropna()
            .astype(bool)
        )

        if not rain.empty:
            rainfall_change = (
                rain.nunique()
                > 1
            )

    return {
        "track_temperature_change": (
            track_change
        ),
        "air_temperature_change": (
            air_change
        ),
        "significant_track_temperature_change": (
            significant_track_change
        ),
        "significant_air_temperature_change": (
            significant_air_change
        ),
        "rainfall_changed": rainfall_change,
        "significant_change": (
            significant_track_change
            or significant_air_change
            or rainfall_change
        ),
    }


def build_weather_insight(
    session,
) -> str:
    """
    Generate a cautious summary of session conditions.
    """

    summary = get_weather_summary(
        session
    )

    change = detect_significant_weather_change(
        session
    )

    parts = []

    if summary[
        "track_temperature"
    ] is not None:

        parts.append(
            f"Median track temperature was "
            f"{summary['track_temperature']:.1f} °C"
        )

    if summary[
        "air_temperature"
    ] is not None:

        parts.append(
            f"median air temperature was "
            f"{summary['air_temperature']:.1f} °C"
        )

    if summary[
        "humidity"
    ] is not None:

        parts.append(
            f"median humidity was "
            f"{summary['humidity']:.0f}%"
        )

    if not parts:

        opening = (
            "Reliable weather data was not available "
            "for this session."
        )

    else:

        opening = (
            ", ".join(parts)
            + "."
        )

    if change[
        "significant_change"
    ]:

        return (
            opening
            + " Conditions changed meaningfully during the session, "
            + "so direct pace comparisons should account for "
            + "possible weather-related differences."
        )

    return (
        opening
        + " No major weather transition was detected using "
        + "the current analytical thresholds."
    )