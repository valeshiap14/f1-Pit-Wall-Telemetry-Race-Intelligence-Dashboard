import pandas as pd

from src.cleaning import (
    add_lap_quality_flags,
    attach_quality_reason,
    build_clean_lap_mask,
    flag_lap_time_outliers,
)


def build_test_laps():
    """
    Create a small predictable lap dataset for cleaning tests.

    The final lap is deliberately much slower so the outlier
    detector has something obvious to identify.
    """

    lap_times = pd.to_timedelta(
        [
            90,
            91,
            90.5,
            89.8,
            160,
        ],
        unit="s",
    )

    # Explicit datetime dtypes prevent Pandas from needing to
    # guess the column type later when a pit timestamp is added.
    pit_in_times = pd.Series(
        pd.NaT,
        index=range(5),
        dtype="timedelta64[ns]",
    )

    pit_out_times = pd.Series(
        pd.NaT,
        index=range(5),
        dtype="timedelta64[ns]",
    )

    return pd.DataFrame(
        {
            "Driver": [
                "AAA",
                "AAA",
                "AAA",
                "AAA",
                "AAA",
            ],
            "Stint": [
                1,
                1,
                1,
                1,
                1,
            ],
            "LapTime": lap_times,
            "PitInTime": pit_in_times,
            "PitOutTime": pit_out_times,
            "Deleted": [
                False,
                False,
                False,
                False,
                False,
            ],
            "IsAccurate": [
                True,
                True,
                True,
                True,
                True,
            ],
        }
    )


def test_quality_flags_are_created():
    """
    The cleaning pipeline should create all expected
    quality-control columns.
    """

    laps = build_test_laps()

    flagged = add_lap_quality_flags(
        laps
    )

    expected_columns = [
        "FlagMissingLapTime",
        "FlagPitLap",
        "FlagDeleted",
        "FlagInvalidLap",
        "FlagSafetyCar",
        "FlagVSC",
        "FlagOutlier",
    ]

    for column in expected_columns:
        assert column in flagged.columns


def test_outlier_lap_is_detected():
    """
    The deliberately slow 160-second lap should be marked
    as an outlier relative to the rest of the stint.
    """

    laps = build_test_laps()

    flagged = add_lap_quality_flags(
        laps
    )

    flagged = flag_lap_time_outliers(
        flagged
    )

    assert bool(
        flagged[
            "FlagOutlier"
        ].iloc[-1]
    )


def test_pit_lap_is_flagged():
    """
    A lap containing PitInTime should be identified
    as pit-affected.
    """

    laps = build_test_laps()

    laps.loc[
        2,
        "PitInTime",
    ] = pd.Timedelta(
        seconds=100
    )

    flagged = add_lap_quality_flags(
        laps
    )

    assert bool(
        flagged[
            "FlagPitLap"
        ].iloc[2]
    )


def test_clean_mask_removes_flagged_laps():
    """
    A quality flag should cause the lap to be excluded
    from the default clean-lap mask.
    """

    laps = build_test_laps()

    flagged = add_lap_quality_flags(
        laps
    )

    flagged.loc[
        1,
        "FlagPitLap",
    ] = True

    mask = build_clean_lap_mask(
        flagged
    )

    assert not bool(
        mask.iloc[1]
    )


def test_quality_reason_reports_clean_lap():
    """
    Laps without any quality problems should receive
    the human-readable reason 'clean'.
    """

    laps = build_test_laps()

    flagged = add_lap_quality_flags(
        laps
    )

    flagged = attach_quality_reason(
        flagged
    )

    assert (
        flagged[
            "QualityReason"
        ].iloc[0]
        == "clean"
    )


def test_deleted_lap_is_flagged():
    """
    Deleted FastF1 laps should not be treated as clean
    analytical observations.
    """

    laps = build_test_laps()

    laps.loc[
        1,
        "Deleted",
    ] = True

    flagged = add_lap_quality_flags(
        laps
    )

    assert bool(
        flagged[
            "FlagDeleted"
        ].iloc[1]
    )


def test_inaccurate_lap_is_flagged():
    """
    FastF1 laps marked inaccurate should be flagged
    for exclusion from analytical models.
    """

    laps = build_test_laps()

    laps.loc[
        3,
        "IsAccurate",
    ] = False

    flagged = add_lap_quality_flags(
        laps
    )

    assert bool(
        flagged[
            "FlagInvalidLap"
        ].iloc[3]
    )


def test_missing_lap_time_is_flagged():
    """
    Missing timing data should be detected rather than
    silently entering later analytical calculations.
    """

    laps = build_test_laps()

    laps.loc[
        0,
        "LapTime",
    ] = pd.NaT

    flagged = add_lap_quality_flags(
        laps
    )

    assert bool(
        flagged[
            "FlagMissingLapTime"
        ].iloc[0]
    )