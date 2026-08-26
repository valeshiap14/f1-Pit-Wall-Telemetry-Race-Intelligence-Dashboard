from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_strategy_laps(
    session,
    driver: str,
) -> pd.DataFrame:
    """
    Prepare race laps for stint and pit-stop strategy analysis.
    """

    laps = session.laps.pick_drivers(
        driver
    ).copy()

    if laps.empty:
        raise ValueError(
            f"No laps were found for {driver}."
        )

    if "LapTime" in laps.columns:
        laps["LapTimeSeconds"] = (
            laps["LapTime"]
            .dt.total_seconds()
        )
    else:
        laps["LapTimeSeconds"] = np.nan

    laps["IsPitLap"] = False

    if "PitInTime" in laps.columns:
        laps["IsPitLap"] = (
            laps["IsPitLap"]
            | laps["PitInTime"].notna()
        )

    if "PitOutTime" in laps.columns:
        laps["IsPitLap"] = (
            laps["IsPitLap"]
            | laps["PitOutTime"].notna()
        )

    return laps.sort_values(
        "LapNumber"
    ).reset_index(
        drop=True
    )


def build_stint_summary(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize each recorded stint for a selected driver.
    """

    if "Stint" not in laps.columns:
        raise ValueError(
            "Stint information is unavailable."
        )

    rows = []

    stint_values = pd.to_numeric(
        laps["Stint"],
        errors="coerce",
    ).dropna()

    for stint_number in sorted(
        stint_values.astype(int).unique()
    ):

        stint = laps[
            laps["Stint"]
            == stint_number
        ].copy()

        if stint.empty:
            continue

        clean = stint[
            stint["LapTimeSeconds"].notna()
            & ~stint["IsPitLap"]
        ].copy()

        compound = "Unknown"

        if "Compound" in stint.columns:
            compounds = (
                stint["Compound"]
                .dropna()
                .astype(str)
            )

            if not compounds.empty:
                compound = compounds.mode().iloc[0]

        rows.append(
            {
                "Stint": stint_number,
                "Compound": compound,
                "StartLap": int(
                    stint["LapNumber"].min()
                ),
                "EndLap": int(
                    stint["LapNumber"].max()
                ),
                "StintLength": int(
                    len(stint)
                ),
                "CleanLaps": int(
                    len(clean)
                ),
                "BestLap": (
                    float(
                        clean["LapTimeSeconds"].min()
                    )
                    if not clean.empty
                    else None
                ),
                "AverageLap": (
                    float(
                        clean["LapTimeSeconds"].mean()
                    )
                    if not clean.empty
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def find_pit_stops(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify pit stops from pit-in and pit-out timestamps.
    """

    pit_rows = []

    for _, lap in laps.iterrows():

        if (
            "PitInTime" in laps.columns
            and pd.notna(
                lap.get("PitInTime")
            )
        ):

            pit_rows.append(
                {
                    "Lap": int(
                        lap["LapNumber"]
                    ),
                    "Type": "Pit In",
                    "Stint": (
                        int(lap["Stint"])
                        if pd.notna(
                            lap.get("Stint")
                        )
                        else None
                    ),
                    "Compound": (
                        str(
                            lap.get(
                                "Compound",
                                "Unknown",
                            )
                        )
                    ),
                }
            )

    return pd.DataFrame(
        pit_rows
    )


def average_pace_before_stop(
    laps: pd.DataFrame,
    pit_lap: int,
    lookback: int = 3,
) -> float | None:
    """
    Calculate average clean pace immediately before a stop.
    """

    candidates = laps[
        (
            laps["LapNumber"]
            < pit_lap
        )
        &
        (
            laps["LapNumber"]
            >= pit_lap - lookback
        )
        &
        laps["LapTimeSeconds"].notna()
        &
        ~laps["IsPitLap"]
    ]

    if candidates.empty:
        return None

    return float(
        candidates[
            "LapTimeSeconds"
        ].mean()
    )


def average_pace_after_stop(
    laps: pd.DataFrame,
    pit_lap: int,
    lookahead: int = 3,
) -> float | None:
    """
    Calculate average clean pace immediately after a stop.
    """

    candidates = laps[
        (
            laps["LapNumber"]
            > pit_lap
        )
        &
        (
            laps["LapNumber"]
            <= pit_lap + lookahead
        )
        &
        laps["LapTimeSeconds"].notna()
        &
        ~laps["IsPitLap"]
    ]

    if candidates.empty:
        return None

    return float(
        candidates[
            "LapTimeSeconds"
        ].mean()
    )


def analyse_pit_stop_effect(
    laps: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare short-run pace before and after each pit stop.
    """

    pit_stops = find_pit_stops(
        laps
    )

    rows = []

    for _, pit in pit_stops.iterrows():

        pit_lap = int(
            pit["Lap"]
        )

        before = average_pace_before_stop(
            laps,
            pit_lap,
        )

        after = average_pace_after_stop(
            laps,
            pit_lap,
        )

        pace_change = None

        if (
            before is not None
            and after is not None
        ):
            pace_change = (
                after - before
            )

        rows.append(
            {
                "PitLap": pit_lap,
                "CompoundBefore": pit[
                    "Compound"
                ],
                "AverageBefore": before,
                "AverageAfter": after,
                "PaceChange": pace_change,
            }
        )

    return pd.DataFrame(
        rows
    )


def compare_driver_strategies(
    session,
    driver_a: str,
    driver_b: str,
) -> dict:
    """
    Build a strategy comparison for two drivers.
    """

    laps_a = prepare_strategy_laps(
        session,
        driver_a,
    )

    laps_b = prepare_strategy_laps(
        session,
        driver_b,
    )

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "laps_a": laps_a,
        "laps_b": laps_b,
        "stints_a": build_stint_summary(
            laps_a
        ),
        "stints_b": build_stint_summary(
            laps_b
        ),
        "pit_effect_a": analyse_pit_stop_effect(
            laps_a
        ),
        "pit_effect_b": analyse_pit_stop_effect(
            laps_b
        ),
    }


def generate_strategy_insight(
    strategy: dict,
) -> list[str]:
    """
    Convert strategy calculations into cautious observations.
    """

    insights = []

    for driver_key, label in [
        ("stints_a", strategy["driver_a"]),
        ("stints_b", strategy["driver_b"]),
    ]:

        stints = strategy[
            driver_key
        ]

        if stints.empty:
            continue

        longest = stints.loc[
            stints[
                "StintLength"
            ].idxmax()
        ]

        insights.append(
            f"{label}'s longest recorded stint lasted "
            f"{int(longest['StintLength'])} laps on "
            f"{longest['Compound']} tyres."
        )

    for pit_key, label in [
        ("pit_effect_a", strategy["driver_a"]),
        ("pit_effect_b", strategy["driver_b"]),
    ]:

        pit_data = strategy[
            pit_key
        ]

        if pit_data.empty:
            continue

        usable = pit_data[
            pit_data[
                "PaceChange"
            ].notna()
        ]

        if usable.empty:
            continue

        best_stop = usable.loc[
            usable[
                "PaceChange"
            ].idxmin()
        ]

        change = float(
            best_stop[
                "PaceChange"
            ]
        )

        if change < 0:
            insights.append(
                f"{label}'s pace improved by approximately "
                f"{abs(change):.3f} s/lap across the short "
                f"window surrounding the stop on lap "
                f"{int(best_stop['PitLap'])}."
            )

    return insights