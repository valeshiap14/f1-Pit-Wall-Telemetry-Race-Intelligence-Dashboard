from __future__ import annotations

import numpy as np
import pandas as pd


def detect_corner_regions(
    telemetry: pd.DataFrame,
    minimum_speed_drop: float = 20.0,
    minimum_corner_gap: float = 80.0,
) -> list[dict]:
    """
    Detect approximate corner regions from the speed trace.

    The algorithm looks for local speed minima and uses them
    as approximate corner centres.

    This is intentionally conservative because telemetry alone
    does not always identify official FIA corner boundaries.
    """

    if "Distance" not in telemetry.columns:
        raise ValueError(
            "Telemetry does not contain Distance."
        )

    if "Speed" not in telemetry.columns:
        raise ValueError(
            "Telemetry does not contain Speed."
        )

    data = telemetry[
        ["Distance", "Speed"]
    ].copy()

    data["Distance"] = pd.to_numeric(
        data["Distance"],
        errors="coerce",
    )

    data["Speed"] = pd.to_numeric(
        data["Speed"],
        errors="coerce",
    )

    data = data.dropna()

    if len(data) < 20:
        raise ValueError(
            "Not enough telemetry samples for corner detection."
        )

    # A small rolling median smooths noisy telemetry without
    # removing the major braking and cornering features.
    data["SmoothedSpeed"] = (
        data["Speed"]
        .rolling(
            window=7,
            center=True,
            min_periods=1,
        )
        .median()
    )

    speeds = data[
        "SmoothedSpeed"
    ].to_numpy()

    distances = data[
        "Distance"
    ].to_numpy()

    candidate_indices = []

    for index in range(
        2,
        len(data) - 2,
    ):
        current_speed = speeds[index]

        is_local_minimum = (
            current_speed
            <= speeds[index - 1]
            and current_speed
            <= speeds[index + 1]
            and current_speed
            <= speeds[index - 2]
            and current_speed
            <= speeds[index + 2]
        )

        if not is_local_minimum:
            continue

        lookback_start = max(
            0,
            index - 30,
        )

        lookahead_end = min(
            len(data),
            index + 31,
        )

        nearby_maximum = max(
            speeds[
                lookback_start:index
            ].max(),
            speeds[
                index + 1:lookahead_end
            ].max(),
        )

        speed_drop = (
            nearby_maximum
            - current_speed
        )

        if speed_drop >= minimum_speed_drop:
            candidate_indices.append(
                index
            )

    corner_centres = []

    for index in candidate_indices:
        distance = distances[index]

        if not corner_centres:
            corner_centres.append(
                index
            )
            continue

        previous_distance = distances[
            corner_centres[-1]
        ]

        if (
            distance - previous_distance
            >= minimum_corner_gap
        ):
            corner_centres.append(
                index
            )
        else:
            previous_index = (
                corner_centres[-1]
            )

            if (
                speeds[index]
                < speeds[previous_index]
            ):
                corner_centres[-1] = (
                    index
                )

    corners = []

    for corner_number, index in enumerate(
        corner_centres,
        start=1,
    ):
        centre_distance = float(
            distances[index]
        )

        # These windows are approximate analytical regions,
        # not claims about official corner boundaries.
        start_distance = max(
            0.0,
            centre_distance - 120.0,
        )

        end_distance = min(
            float(distances[-1]),
            centre_distance + 140.0,
        )

        corners.append(
            {
                "corner_number": corner_number,
                "centre_distance": centre_distance,
                "start_distance": start_distance,
                "end_distance": end_distance,
                "minimum_speed": float(
                    speeds[index]
                ),
            }
        )

    return corners


def _window(
    telemetry: pd.DataFrame,
    start_distance: float,
    end_distance: float,
) -> pd.DataFrame:
    """Return telemetry inside a selected distance region."""

    return telemetry[
        (
            telemetry["Distance"]
            >= start_distance
        )
        &
        (
            telemetry["Distance"]
            <= end_distance
        )
    ].copy()


def _first_brake_distance(
    data: pd.DataFrame,
) -> float | None:
    """
    Find the first meaningful brake application in a region.
    """

    if "Brake" not in data.columns:
        return None

    brake = (
        data["Brake"]
        .fillna(False)
        .astype(bool)
    )

    braking = data[
        brake
    ]

    if braking.empty:
        return None

    return float(
        braking["Distance"].iloc[0]
    )


def _full_throttle_distance(
    data: pd.DataFrame,
    minimum_distance: float,
) -> float | None:
    """
    Find where the driver first returns to approximately
    full throttle after the corner minimum-speed point.
    """

    if "Throttle" not in data.columns:
        return None

    exit_data = data[
        data["Distance"]
        >= minimum_distance
    ].copy()

    if exit_data.empty:
        return None

    throttle = pd.to_numeric(
        exit_data["Throttle"],
        errors="coerce",
    )

    full_throttle = exit_data[
        throttle >= 99
    ]

    if full_throttle.empty:
        return None

    return float(
        full_throttle[
            "Distance"
        ].iloc[0]
    )


def analyse_corner(
    telemetry: pd.DataFrame,
    corner: dict,
) -> dict:
    """
    Calculate engineering metrics for one approximate corner.
    """

    region = _window(
        telemetry,
        corner["start_distance"],
        corner["end_distance"],
    )

    if region.empty:
        raise ValueError(
            "Corner region contains no telemetry."
        )

    speed = pd.to_numeric(
        region["Speed"],
        errors="coerce",
    )

    valid_speed = region[
        speed.notna()
    ].copy()

    if valid_speed.empty:
        raise ValueError(
            "Corner region contains no usable speed data."
        )

    valid_speed["Speed"] = pd.to_numeric(
        valid_speed["Speed"],
        errors="coerce",
    )

    minimum_index = (
        valid_speed["Speed"]
        .idxmin()
    )

    minimum_row = valid_speed.loc[
        minimum_index
    ]

    minimum_speed = float(
        minimum_row["Speed"]
    )

    minimum_speed_distance = float(
        minimum_row["Distance"]
    )

    entry_region = valid_speed[
        valid_speed["Distance"]
        < minimum_speed_distance
    ]

    exit_region = valid_speed[
        valid_speed["Distance"]
        > minimum_speed_distance
    ]

    entry_speed = None
    exit_speed = None

    if not entry_region.empty:
        entry_speed = float(
            entry_region[
                "Speed"
            ].iloc[0]
        )

    if not exit_region.empty:
        exit_speed = float(
            exit_region[
                "Speed"
            ].iloc[-1]
        )

    brake_distance = (
        _first_brake_distance(
            region
        )
    )

    full_throttle_distance = (
        _full_throttle_distance(
            region,
            minimum_speed_distance,
        )
    )

    gear = None

    if "nGear" in region.columns:
        gear_value = pd.to_numeric(
            minimum_row.get(
                "nGear"
            ),
            errors="coerce",
        )

        if pd.notna(gear_value):
            gear = int(
                round(
                    float(gear_value)
                )
            )

    throttle_at_minimum = None

    if "Throttle" in region.columns:
        throttle_value = pd.to_numeric(
            minimum_row.get(
                "Throttle"
            ),
            errors="coerce",
        )

        if pd.notna(
            throttle_value
        ):
            throttle_at_minimum = float(
                throttle_value
            )

    return {
        "corner_number": corner[
            "corner_number"
        ],
        "start_distance": corner[
            "start_distance"
        ],
        "centre_distance": corner[
            "centre_distance"
        ],
        "end_distance": corner[
            "end_distance"
        ],
        "entry_speed": entry_speed,
        "minimum_speed": minimum_speed,
        "minimum_speed_distance": minimum_speed_distance,
        "exit_speed": exit_speed,
        "brake_distance": brake_distance,
        "full_throttle_distance": full_throttle_distance,
        "gear": gear,
        "throttle_at_minimum": throttle_at_minimum,
    }


def analyse_all_corners(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect and analyse every approximate corner on a lap.
    """

    corners = detect_corner_regions(
        telemetry
    )

    results = []

    for corner in corners:
        try:
            result = analyse_corner(
                telemetry,
                corner,
            )

            results.append(
                result
            )

        except ValueError:
            continue

    return pd.DataFrame(
        results
    )


def compare_driver_corners(
    telemetry_a: pd.DataFrame,
    telemetry_b: pd.DataFrame,
    aligned: pd.DataFrame,
    driver_a: str,
    driver_b: str,
) -> pd.DataFrame:
    """
    Compare two drivers corner-by-corner.

    Driver A's detected corner regions are used as common
    analytical windows so both drivers are measured over
    approximately the same track positions.
    """

    corners = detect_corner_regions(
        telemetry_a
    )

    rows = []

    for corner in corners:

        try:

            analysis_a = analyse_corner(
                telemetry_a,
                corner,
            )

            analysis_b = analyse_corner(
                telemetry_b,
                corner,
            )

        except ValueError:
            continue

        start_distance = (
            corner["start_distance"]
        )

        end_distance = (
            corner["end_distance"]
        )

        delta_region = aligned[
            (
                aligned["Distance"]
                >= start_distance
            )
            &
            (
                aligned["Distance"]
                <= end_distance
            )
        ]

        corner_delta = None

        if not delta_region.empty:
            start_delta = float(
                delta_region[
                    "Delta"
                ].iloc[0]
            )

            end_delta = float(
                delta_region[
                    "Delta"
                ].iloc[-1]
            )

            # Change in cumulative delta estimates how much
            # relative lap time changed through this region.
            corner_delta = (
                end_delta - start_delta
            )

        rows.append(
            {
                "Corner": corner[
                    "corner_number"
                ],
                "Distance": corner[
                    "centre_distance"
                ],

                f"{driver_a} Entry": analysis_a[
                    "entry_speed"
                ],

                f"{driver_b} Entry": analysis_b[
                    "entry_speed"
                ],

                f"{driver_a} Minimum": analysis_a[
                    "minimum_speed"
                ],

                f"{driver_b} Minimum": analysis_b[
                    "minimum_speed"
                ],

                f"{driver_a} Exit": analysis_a[
                    "exit_speed"
                ],

                f"{driver_b} Exit": analysis_b[
                    "exit_speed"
                ],

                f"{driver_a} Brake": analysis_a[
                    "brake_distance"
                ],

                f"{driver_b} Brake": analysis_b[
                    "brake_distance"
                ],

                f"{driver_a} Full Throttle": analysis_a[
                    "full_throttle_distance"
                ],

                f"{driver_b} Full Throttle": analysis_b[
                    "full_throttle_distance"
                ],

                f"{driver_a} Gear": analysis_a[
                    "gear"
                ],

                f"{driver_b} Gear": analysis_b[
                    "gear"
                ],

                "Corner Delta": corner_delta,
            }
        )

    return pd.DataFrame(
        rows
    )


def generate_corner_insight(
    row: pd.Series,
    driver_a: str,
    driver_b: str,
) -> str:
    """
    Convert measured corner differences into a short
    race-engineer style explanation.
    """

    delta = row.get(
        "Corner Delta"
    )

    minimum_a = row.get(
        f"{driver_a} Minimum"
    )

    minimum_b = row.get(
        f"{driver_b} Minimum"
    )

    brake_a = row.get(
        f"{driver_a} Brake"
    )

    brake_b = row.get(
        f"{driver_b} Brake"
    )

    throttle_a = row.get(
        f"{driver_a} Full Throttle"
    )

    throttle_b = row.get(
        f"{driver_b} Full Throttle"
    )

    observations = []

    if (
        pd.notna(minimum_a)
        and pd.notna(minimum_b)
    ):
        difference = (
            minimum_a - minimum_b
        )

        if abs(difference) >= 1:
            faster = (
                driver_a
                if difference > 0
                else driver_b
            )

            observations.append(
                f"{faster} carried approximately "
                f"{abs(difference):.0f} km/h more "
                f"minimum speed"
            )

    if (
        pd.notna(brake_a)
        and pd.notna(brake_b)
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

            observations.append(
                f"{later_driver} braked approximately "
                f"{abs(difference):.0f} m later"
            )

    if (
        pd.notna(throttle_a)
        and pd.notna(throttle_b)
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

            observations.append(
                f"{earlier_driver} reached full throttle "
                f"approximately {abs(difference):.0f} m earlier"
            )

    if pd.notna(delta):
        gaining_driver = (
            driver_b
            if delta > 0
            else driver_a
        )

        opening = (
            f"{gaining_driver} gained approximately "
            f"{abs(delta):.3f} s through this region."
        )

    else:
        opening = (
            "A reliable time delta could not be calculated "
            "for this region."
        )

    if not observations:
        return opening

    return (
        opening
        + " "
        + ". ".join(
            observations
        )
        + "."
    )