import numpy as np
import pandas as pd

from src.strategy import (
    analyse_pit_stop_effect,
    average_pace_after_stop,
    average_pace_before_stop,
    build_stint_summary,
    find_pit_stops,
)


def build_strategy_laps():
    """
    Build a miniature race containing two stints and one pit stop.

    The first stint deliberately slows slightly before the stop,
    while the second begins with stronger pace.
    """

    return pd.DataFrame(
        {
            "LapNumber": np.arange(
                1,
                11,
            ),
            "LapTimeSeconds": [
                91.0,
                91.2,
                91.4,
                91.6,
                92.0,
                94.0,
                90.5,
                90.7,
                90.9,
                91.1,
            ],
            "Stint": [
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
            ],
            "Compound": [
                "MEDIUM",
                "MEDIUM",
                "MEDIUM",
                "MEDIUM",
                "MEDIUM",
                "MEDIUM",
                "HARD",
                "HARD",
                "HARD",
                "HARD",
            ],
            "IsPitLap": [
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
            "PitInTime": [
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.Timedelta(
                    seconds=500
                ),
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
            ],
        }
    )


def test_stint_summary_detects_two_stints():
    laps = build_strategy_laps()

    summary = build_stint_summary(
        laps
    )

    assert len(summary) == 2


def test_first_stint_uses_medium_tyres():
    laps = build_strategy_laps()

    summary = build_stint_summary(
        laps
    )

    assert (
        summary.iloc[0][
            "Compound"
        ]
        == "MEDIUM"
    )


def test_second_stint_uses_hard_tyres():
    laps = build_strategy_laps()

    summary = build_stint_summary(
        laps
    )

    assert (
        summary.iloc[1][
            "Compound"
        ]
        == "HARD"
    )


def test_pit_stop_is_detected():
    laps = build_strategy_laps()

    stops = find_pit_stops(
        laps
    )

    assert len(stops) == 1

    assert (
        int(
            stops.iloc[0]["Lap"]
        )
        == 6
    )


def test_average_pace_before_stop():
    laps = build_strategy_laps()

    pace = average_pace_before_stop(
        laps,
        pit_lap=6,
        lookback=3,
    )

    expected = np.mean(
        [
            91.4,
            91.6,
            92.0,
        ]
    )

    assert np.isclose(
        pace,
        expected,
    )


def test_average_pace_after_stop():
    laps = build_strategy_laps()

    pace = average_pace_after_stop(
        laps,
        pit_lap=6,
        lookahead=3,
    )

    expected = np.mean(
        [
            90.5,
            90.7,
            90.9,
        ]
    )

    assert np.isclose(
        pace,
        expected,
    )


def test_pit_stop_effect_is_calculated():
    laps = build_strategy_laps()

    result = analyse_pit_stop_effect(
        laps
    )

    assert len(result) == 1

    assert pd.notna(
        result.iloc[0][
            "PaceChange"
        ]
    )


def test_post_stop_pace_is_faster():
    laps = build_strategy_laps()

    result = analyse_pit_stop_effect(
        laps
    )

    assert (
        result.iloc[0][
            "PaceChange"
        ]
        < 0
    )