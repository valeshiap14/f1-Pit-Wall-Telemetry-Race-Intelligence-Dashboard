import numpy as np
import pandas as pd

from src.track_evolution import (
    calculate_track_evolution_trend,
    remove_session_outliers,
    safe_correlation,
)


def build_track_data():
    """
    Create a session where lap time gradually improves.

    Every ten minutes the representative lap becomes
    approximately 0.2 seconds faster.
    """

    session_time = np.arange(
        0,
        6000,
        300,
        dtype=float,
    )

    lap_time = (
        92
        - (
            session_time
            / 600
        )
        * 0.2
    )

    track_temp = np.linspace(
        35,
        40,
        len(session_time),
    )

    return pd.DataFrame(
        {
            "SessionTimeSeconds": (
                session_time
            ),
            "LapTimeSeconds": (
                lap_time
            ),
            "TrackTemp": (
                track_temp
            ),
        }
    )


def test_track_evolution_detects_improvement():
    data = build_track_data()

    trend = (
        calculate_track_evolution_trend(
            data
        )
    )

    assert (
        trend[
            "seconds_per_10_minutes"
        ]
        < 0
    )


def test_track_evolution_matches_expected_rate():
    data = build_track_data()

    trend = (
        calculate_track_evolution_trend(
            data
        )
    )

    assert np.isclose(
        trend[
            "seconds_per_10_minutes"
        ],
        -0.2,
        atol=0.001,
    )


def test_prediction_length_matches_data():
    data = build_track_data()

    trend = (
        calculate_track_evolution_trend(
            data
        )
    )

    assert (
        len(
            trend["prediction"]
        )
        == len(data)
    )


def test_correlation_is_calculated():
    data = build_track_data()

    result = safe_correlation(
        data,
        "TrackTemp",
    )

    assert result is not None

    assert (
        "correlation"
        in result
    )


def test_session_outlier_is_removed():
    data = build_track_data()

    bad_row = pd.DataFrame(
        {
            "SessionTimeSeconds": [
                6500
            ],
            "LapTimeSeconds": [
                180
            ],
            "TrackTemp": [
                40
            ],
        }
    )

    data = pd.concat(
        [
            data,
            bad_row,
        ],
        ignore_index=True,
    )

    cleaned = remove_session_outliers(
        data
    )

    assert (
        cleaned[
            "LapTimeSeconds"
        ].max()
        < 180
    )


def test_positive_temperature_correlation_with_session_time():
    data = build_track_data()

    result = safe_correlation(
        data,
        "TrackTemp",
        "SessionTimeSeconds",
    )

    assert result is not None

    assert (
        result[
            "correlation"
        ]
        > 0
    )