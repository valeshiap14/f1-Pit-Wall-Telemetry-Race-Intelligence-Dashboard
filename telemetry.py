from __future__ import annotations

import numpy as np
import pandas as pd


def get_lap_telemetry(lap) -> pd.DataFrame:
    """
    Extract and clean telemetry from one FastF1 lap.

    The lap is only accepted when it has a real distance axis and
    enough timing samples to support interpolation.
    """

    telemetry = lap.get_telemetry()

    if (
        telemetry is None
        or telemetry.empty
    ):
        raise ValueError(
            "No telemetry data is available for this lap."
        )

    useful_channels = [
        "Time",
        "SessionTime",
        "Distance",
        "Speed",
        "Throttle",
        "Brake",
        "RPM",
        "nGear",
        "DRS",
        "X",
        "Y",
    ]

    available_channels = [
        column
        for column in useful_channels
        if column in telemetry.columns
    ]

    telemetry = telemetry[
        available_channels
    ].copy()

    if "Distance" not in telemetry.columns:
        raise ValueError(
            "Telemetry does not contain lap distance."
        )

    if "Time" not in telemetry.columns:
        raise ValueError(
            "Telemetry does not contain a Time channel."
        )

    telemetry["Distance"] = pd.to_numeric(
        telemetry["Distance"],
        errors="coerce",
    )

    telemetry = telemetry.dropna(
        subset=["Distance"]
    )

    telemetry = telemetry[
        telemetry["Distance"] >= 0
    ]

    telemetry = telemetry.sort_values(
        "Distance"
    )

    telemetry = telemetry.drop_duplicates(
        subset=["Distance"],
        keep="first",
    )

    telemetry = telemetry.reset_index(
        drop=True
    )

    if len(telemetry) < 2:
        raise ValueError(
            "The lap does not contain enough valid telemetry "
            "samples for comparison."
        )

    maximum_distance = telemetry[
        "Distance"
    ].max()

    if (
        pd.isna(maximum_distance)
        or maximum_distance <= 0
    ):
        raise ValueError(
            "Telemetry does not contain a valid lap distance."
        )

    return telemetry


def get_fastest_lap_telemetry(
    session,
    driver: str,
) -> tuple[object, pd.DataFrame]:
    """
    Retrieve the fastest timed lap that also contains usable telemetry.

    If the absolute fastest lap has incomplete telemetry, continue
    through the remaining timed laps until a usable one is found.
    """

    driver_laps = session.laps.pick_drivers(
        driver
    )

    if driver_laps.empty:
        raise ValueError(
            f"No laps were found for {driver}."
        )

    if "LapTime" not in driver_laps.columns:
        raise ValueError(
            f"Lap timing is unavailable for {driver}."
        )

    valid_laps = driver_laps[
        driver_laps["LapTime"].notna()
    ].copy()

    if valid_laps.empty:
        raise ValueError(
            f"No timed laps were found for {driver}."
        )

    valid_laps = valid_laps.sort_values(
        "LapTime"
    )

    last_error = None

    for _, lap in valid_laps.iterlaps():

        try:
            telemetry = get_lap_telemetry(
                lap
            )

            return (
                lap,
                telemetry,
            )

        except Exception as error:
            last_error = error
            continue

    raise ValueError(
        f"No usable telemetry was found for any timed lap "
        f"completed by {driver}."
    ) from last_error


def _elapsed_seconds(
    telemetry: pd.DataFrame,
) -> np.ndarray:
    """
    Convert telemetry time into elapsed lap seconds.
    """

    if "Time" not in telemetry.columns:
        raise ValueError(
            "Telemetry does not contain a Time channel."
        )

    time_values = pd.to_timedelta(
        telemetry["Time"],
        errors="coerce",
    )

    seconds = (
        time_values.dt.total_seconds()
        .to_numpy(dtype=float)
    )

    if np.isfinite(seconds).sum() < 2:
        raise ValueError(
            "Telemetry does not contain enough valid timing samples."
        )

    return seconds


def _interpolate_numeric_channel(
    telemetry: pd.DataFrame,
    channel: str,
    distance_grid: np.ndarray,
) -> np.ndarray:

    if channel not in telemetry.columns:
        return np.full(
            len(distance_grid),
            np.nan,
        )

    distance = pd.to_numeric(
        telemetry["Distance"],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = pd.to_numeric(
        telemetry[channel],
        errors="coerce",
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(distance)
        & np.isfinite(values)
    )

    if valid.sum() < 2:
        return np.full(
            len(distance_grid),
            np.nan,
        )

    return np.interp(
        distance_grid,
        distance[valid],
        values[valid],
    )


def _interpolate_boolean_channel(
    telemetry: pd.DataFrame,
    channel: str,
    distance_grid: np.ndarray,
) -> np.ndarray:
    """
    Map discrete channels such as Brake onto the shared axis.
    """

    if channel not in telemetry.columns:
        return np.full(
            len(distance_grid),
            np.nan,
        )

    if telemetry.empty:
        return np.full(
            len(distance_grid),
            np.nan,
        )

    source_distance = pd.to_numeric(
        telemetry["Distance"],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = (
        telemetry[channel]
        .fillna(False)
        .astype(bool)
        .astype(float)
        .to_numpy()
    )

    valid = np.isfinite(
        source_distance
    )

    source_distance = source_distance[
        valid
    ]

    values = values[
        valid
    ]

    if len(source_distance) == 0:
        return np.full(
            len(distance_grid),
            np.nan,
        )

    if len(source_distance) == 1:
        return np.full(
            len(distance_grid),
            values[0],
        )

    indices = np.searchsorted(
        source_distance,
        distance_grid,
        side="left",
    )

    indices = np.clip(
        indices,
        0,
        len(source_distance) - 1,
    )

    previous_indices = np.maximum(
        indices - 1,
        0,
    )

    current_distance = np.abs(
        source_distance[indices]
        - distance_grid
    )

    previous_distance = np.abs(
        distance_grid
        - source_distance[previous_indices]
    )

    use_previous = (
        previous_distance
        < current_distance
    )

    indices[
        use_previous
    ] = previous_indices[
        use_previous
    ]

    return values[
        indices
    ]


def align_driver_telemetry(
    telemetry_a: pd.DataFrame,
    telemetry_b: pd.DataFrame,
    samples: int = 3000,
) -> pd.DataFrame:

    if samples < 100:
        raise ValueError(
            "At least 100 alignment samples are required."
        )

    if telemetry_a.empty or telemetry_b.empty:
        raise ValueError(
            "Both drivers require usable telemetry."
        )

    maximum_distance = min(
        telemetry_a["Distance"].max(),
        telemetry_b["Distance"].max(),
    )

    if (
        pd.isna(maximum_distance)
        or maximum_distance <= 0
    ):
        raise ValueError(
            "Unable to determine a valid lap distance."
        )

    distance_grid = np.linspace(
        0.0,
        float(maximum_distance),
        samples,
    )

    aligned = pd.DataFrame(
        {
            "Distance": distance_grid,
        }
    )

    continuous_channels = [
        "Speed",
        "Throttle",
        "RPM",
        "nGear",
        "DRS",
    ]

    for channel in continuous_channels:

        aligned[f"{channel}_A"] = (
            _interpolate_numeric_channel(
                telemetry_a,
                channel,
                distance_grid,
            )
        )

        aligned[f"{channel}_B"] = (
            _interpolate_numeric_channel(
                telemetry_b,
                channel,
                distance_grid,
            )
        )

    aligned["Brake_A"] = (
        _interpolate_boolean_channel(
            telemetry_a,
            "Brake",
            distance_grid,
        )
    )

    aligned["Brake_B"] = (
        _interpolate_boolean_channel(
            telemetry_b,
            "Brake",
            distance_grid,
        )
    )

    time_a = _elapsed_seconds(
        telemetry_a
    )

    time_b = _elapsed_seconds(
        telemetry_b
    )

    distance_a = telemetry_a[
        "Distance"
    ].to_numpy(dtype=float)

    distance_b = telemetry_b[
        "Distance"
    ].to_numpy(dtype=float)

    valid_time_a = (
        np.isfinite(distance_a)
        & np.isfinite(time_a)
    )

    valid_time_b = (
        np.isfinite(distance_b)
        & np.isfinite(time_b)
    )

    if valid_time_a.sum() < 2 or valid_time_b.sum() < 2:
        raise ValueError(
            "Insufficient timing data for lap-delta calculation."
        )

    aligned["Time_A"] = np.interp(
        distance_grid,
        distance_a[valid_time_a],
        time_a[valid_time_a],
    )

    aligned["Time_B"] = np.interp(
        distance_grid,
        distance_b[valid_time_b],
        time_b[valid_time_b],
    )

    aligned["Delta"] = (
        aligned["Time_A"]
        - aligned["Time_B"]
    )

    return aligned


def calculate_driver_metrics(
    telemetry: pd.DataFrame,
) -> dict:

    metrics = {
        "maximum_speed": None,
        "average_speed": None,
        "full_throttle_percentage": None,
        "braking_events": None,
        "average_rpm": None,
    }

    if "Speed" in telemetry.columns:

        speed = pd.to_numeric(
            telemetry["Speed"],
            errors="coerce",
        )

        metrics["maximum_speed"] = (
            float(speed.max())
            if speed.notna().any()
            else None
        )

        metrics["average_speed"] = (
            float(speed.mean())
            if speed.notna().any()
            else None
        )

    if "Throttle" in telemetry.columns:

        throttle = pd.to_numeric(
            telemetry["Throttle"],
            errors="coerce",
        )

        valid_throttle = throttle.dropna()

        if not valid_throttle.empty:

            full_throttle = (
                valid_throttle >= 99
            )

            metrics[
                "full_throttle_percentage"
            ] = float(
                full_throttle.mean() * 100
            )

    if "Brake" in telemetry.columns:

        brake = (
            telemetry["Brake"]
            .fillna(False)
            .astype(bool)
        )

        brake_start = (
            brake.astype(int).diff() == 1
        )

        metrics["braking_events"] = int(
            brake_start.sum()
        )

    if "RPM" in telemetry.columns:

        rpm = pd.to_numeric(
            telemetry["RPM"],
            errors="coerce",
        )

        metrics["average_rpm"] = (
            float(rpm.mean())
            if rpm.notna().any()
            else None
        )

    return metrics


def compare_fastest_laps(
    session,
    driver_a: str,
    driver_b: str,
) -> dict:

    if driver_a == driver_b:
        raise ValueError(
            "Two different drivers are required."
        )

    lap_a, telemetry_a = (
        get_fastest_lap_telemetry(
            session,
            driver_a,
        )
    )

    lap_b, telemetry_b = (
        get_fastest_lap_telemetry(
            session,
            driver_b,
        )
    )

    aligned = align_driver_telemetry(
        telemetry_a,
        telemetry_b,
    )

    metrics_a = calculate_driver_metrics(
        telemetry_a
    )

    metrics_b = calculate_driver_metrics(
        telemetry_b
    )

    return {
        "driver_a": driver_a,
        "driver_b": driver_b,
        "lap_a": lap_a,
        "lap_b": lap_b,
        "telemetry_a": telemetry_a,
        "telemetry_b": telemetry_b,
        "aligned": aligned,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
    }


if __name__ == "__main__":

    from src.data_loader import load_session

    session = load_session(
        2025,
        "Monaco Grand Prix",
        "Q",
    )

    comparison = compare_fastest_laps(
        session,
        "NOR",
        "LEC",
    )

    print("\nNOR metrics:")
    print(comparison["metrics_a"])

    print("\nLEC metrics:")
    print(comparison["metrics_b"])

    print("\nAligned telemetry:")
    print(
        comparison["aligned"][
            [
                "Distance",
                "Speed_A",
                "Speed_B",
                "Throttle_A",
                "Throttle_B",
                "Delta",
            ]
        ].head(10)
    )
