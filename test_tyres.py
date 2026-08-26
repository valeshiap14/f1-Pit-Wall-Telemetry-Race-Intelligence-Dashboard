import numpy as np
import pandas as pd

from src.tyres import (
    fit_linear_degradation,
    remove_lap_time_outliers,
    tyre_stint_summary,
)


def build_stint():
    """
    Create a predictable tyre stint with roughly
    0.1 seconds of degradation per lap.
    """

    tyre_life = np.arange(
        1,
        11,
        dtype=float,
    )

    lap_times = (
        90
        + tyre_life * 0.1
    )

    return pd.DataFrame(
        {
            "TyreLife": tyre_life,
            "LapTimeSeconds": lap_times,
            "Compound": [
                "MEDIUM"
            ] * len(tyre_life),
            "LapNumber": np.arange(
                1,
                11,
            ),
        }
    )


def test_linear_degradation_is_positive():
    stint = build_stint()

    result = fit_linear_degradation(
        stint
    )

    assert (
        result[
            "seconds_per_lap"
        ]
        > 0
    )


def test_linear_degradation_is_close_to_expected_value():
    stint = build_stint()

    result = fit_linear_degradation(
        stint
    )

    assert np.isclose(
        result[
            "seconds_per_lap"
        ],
        0.1,
        atol=0.001,
    )


def test_degradation_model_has_low_error():
    stint = build_stint()

    result = fit_linear_degradation(
        stint
    )

    assert (
        result["mae"]
        < 0.01
    )


def test_stint_summary_detects_compound():
    stint = build_stint()

    summary = tyre_stint_summary(
        stint
    )

    assert (
        summary[
            "compound"
        ]
        == "MEDIUM"
    )


def test_outlier_removal_removes_extreme_lap():
    stint = build_stint()

    bad_lap = pd.DataFrame(
        {
            "TyreLife": [11],
            "LapTimeSeconds": [150],
            "Compound": ["MEDIUM"],
            "LapNumber": [11],
        }
    )

    stint = pd.concat(
        [
            stint,
            bad_lap,
        ],
        ignore_index=True,
    )

    cleaned = remove_lap_time_outliers(
        stint
    )

    assert (
        cleaned[
            "LapTimeSeconds"
        ].max()
        < 150
    )