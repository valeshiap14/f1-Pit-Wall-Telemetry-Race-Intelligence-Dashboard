from __future__ import annotations

import numpy as np
import pandas as pd


def add_lap_quality_flags(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add transparent quality flags to lap data.

    Each flag records why a lap may be unsuitable for
    pace, tyre, strategy, or predictive analysis.
    """

    clean = laps.copy()

    if "LapTime" in clean.columns:
        clean["LapTimeSeconds"] = (
            clean["LapTime"]
            .dt.total_seconds()
        )
    else:
        clean["LapTimeSeconds"] = np.nan

    clean["FlagMissingLapTime"] = (
        clean["LapTimeSeconds"].isna()
    )

    clean["FlagPitLap"] = False

    if "PitInTime" in clean.columns:
        clean["FlagPitLap"] |= (
            clean["PitInTime"].notna()
        )

    if "PitOutTime" in clean.columns:
        clean["FlagPitLap"] |= (
            clean["PitOutTime"].notna()
        )

    clean["FlagDeleted"] = False

    if "Deleted" in clean.columns:
        clean["FlagDeleted"] = (
            clean["Deleted"]
            .fillna(False)
            .astype(bool)
        )

    clean["FlagInvalidLap"] = False

    if "IsAccurate" in clean.columns:
        clean["FlagInvalidLap"] |= (
            ~clean["IsAccurate"]
            .fillna(False)
            .astype(bool)
        )

    if "LapTimeSeconds" in clean.columns:
        clean["FlagInvalidLap"] |= (
            clean["LapTimeSeconds"]
            .fillna(0)
            <= 0
        )

    clean["FlagSafetyCar"] = False
    clean["FlagVSC"] = False

    if "TrackStatus" in clean.columns:
        status = (
            clean["TrackStatus"]
            .fillna("")
            .astype(str)
        )

        # FastF1 track-status codes may contain multiple states.
        # These checks are deliberately conservative.
        clean["FlagSafetyCar"] = (
            status.str.contains("4")
            | status.str.contains("6")
        )

        clean["FlagVSC"] = (
            status.str.contains("7")
        )

    clean["FlagOutlier"] = False

    return clean


def flag_lap_time_outliers(
    laps: pd.DataFrame,
    group_columns: list[str] | None = None,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Flag lap-time outliers using IQR.

    By default, laps are grouped by driver and stint so that
    different drivers and tyre runs are not judged against
    one global lap-time distribution.
    """

    result = laps.copy()

    if "LapTimeSeconds" not in result.columns:
        raise ValueError(
            "LapTimeSeconds must exist before outlier detection."
        )

    if "FlagOutlier" not in result.columns:
        result["FlagOutlier"] = False

    if group_columns is None:
        group_columns = [
            column
            for column in [
                "Driver",
                "Stint",
            ]
            if column in result.columns
        ]

    if group_columns:
        groups = result.groupby(
            group_columns,
            dropna=False,
        )
    else:
        groups = [(None, result)]

    for _, group in groups:

        values = pd.to_numeric(
            group["LapTimeSeconds"],
            errors="coerce",
        )

        valid = values.dropna()

        if len(valid) < 4:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)

        iqr = q3 - q1

        if iqr <= 0:
            continue

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outlier_mask = (
            values < lower
        ) | (
            values > upper
        )

        result.loc[
            group.index,
            "FlagOutlier"
        ] |= outlier_mask.fillna(False)

    return result


def add_weather_change_flag(
    laps: pd.DataFrame,
    track_temp_threshold: float = 5.0,
) -> pd.DataFrame:
    """
    Flag laps exposed to large track-temperature changes.

    This does not imply that temperature caused the pace change;
    it simply marks potentially non-comparable conditions.
    """

    result = laps.copy()

    result["FlagWeatherChange"] = False

    if "TrackTemp" not in result.columns:
        return result

    track_temp = pd.to_numeric(
        result["TrackTemp"],
        errors="coerce",
    )

    median_temp = track_temp.median()

    if pd.isna(median_temp):
        return result

    result["FlagWeatherChange"] = (
        (track_temp - median_temp).abs()
        >= track_temp_threshold
    )

    return result


def attach_quality_reason(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a human-readable explanation of why each lap was flagged.
    """

    result = laps.copy()

    flag_labels = {
        "FlagMissingLapTime": "missing lap time",
        "FlagPitLap": "pit/in-out lap",
        "FlagDeleted": "deleted lap",
        "FlagInvalidLap": "invalid/inaccurate lap",
        "FlagSafetyCar": "safety car affected",
        "FlagVSC": "virtual safety car affected",
        "FlagOutlier": "lap-time outlier",
        "FlagWeatherChange": "significant weather change",
    }

    def build_reason(row) -> str:

        reasons = []

        for column, label in flag_labels.items():

            if (
                column in row.index
                and bool(row[column])
            ):
                reasons.append(
                    label
                )

        if not reasons:
            return "clean"

        return ", ".join(
            reasons
        )

    result["QualityReason"] = (
        result.apply(
            build_reason,
            axis=1,
        )
    )

    return result


def build_clean_lap_mask(
    laps: pd.DataFrame,
    exclude_safety_car: bool = True,
    exclude_vsc: bool = True,
    exclude_weather_change: bool = False,
) -> pd.Series:
    """
    Return a boolean mask for representative clean laps.
    """

    mask = pd.Series(
        True,
        index=laps.index,
    )

    always_excluded = [
        "FlagMissingLapTime",
        "FlagPitLap",
        "FlagDeleted",
        "FlagInvalidLap",
        "FlagOutlier",
    ]

    for column in always_excluded:

        if column in laps.columns:
            mask &= ~(
                laps[column]
                .fillna(False)
                .astype(bool)
            )

    if (
        exclude_safety_car
        and "FlagSafetyCar" in laps.columns
    ):
        mask &= ~(
            laps["FlagSafetyCar"]
            .fillna(False)
            .astype(bool)
        )

    if (
        exclude_vsc
        and "FlagVSC" in laps.columns
    ):
        mask &= ~(
            laps["FlagVSC"]
            .fillna(False)
            .astype(bool)
        )

    if (
        exclude_weather_change
        and "FlagWeatherChange" in laps.columns
    ):
        mask &= ~(
            laps["FlagWeatherChange"]
            .fillna(False)
            .astype(bool)
        )

    return mask


def clean_laps(
    laps: pd.DataFrame,
    exclude_safety_car: bool = True,
    exclude_vsc: bool = True,
    exclude_weather_change: bool = False,
) -> pd.DataFrame:
    """
    Run the complete lap-cleaning pipeline.

    The original rows are preserved until the final mask is applied,
    making every exclusion traceable through the quality flags.
    """

    result = add_lap_quality_flags(
        laps
    )

    result = flag_lap_time_outliers(
        result
    )

    result = add_weather_change_flag(
        result
    )

    result = attach_quality_reason(
        result
    )

    mask = build_clean_lap_mask(
        result,
        exclude_safety_car=exclude_safety_car,
        exclude_vsc=exclude_vsc,
        exclude_weather_change=exclude_weather_change,
    )

    return (
        result[
            mask
        ]
        .copy()
        .reset_index(drop=True)
    )


def data_quality_summary(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how many laps were affected by each quality issue.
    """

    flag_columns = [
        column
        for column in [
            "FlagMissingLapTime",
            "FlagPitLap",
            "FlagDeleted",
            "FlagInvalidLap",
            "FlagSafetyCar",
            "FlagVSC",
            "FlagOutlier",
            "FlagWeatherChange",
        ]
        if column in laps.columns
    ]

    rows = []

    for column in flag_columns:

        count = int(
            laps[column]
            .fillna(False)
            .astype(bool)
            .sum()
        )

        rows.append(
            {
                "Quality Flag": column,
                "Affected Laps": count,
                "Percentage": (
                    count / len(laps) * 100
                    if len(laps)
                    else 0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )