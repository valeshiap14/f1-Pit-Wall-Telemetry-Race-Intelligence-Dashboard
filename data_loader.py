from pathlib import Path

import fastf1
import pandas as pd

# FastF1 cache
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


def load_session(
    year: int,
    grand_prix: str,
    session_type: str,
):
    """
    Download and load an F1 session.
    """

    print(
        f"Loading {year} {grand_prix} "
        f"session: {session_type}..."
    )

    try:
        session = fastf1.get_session(
            year,
            grand_prix,
            session_type,
        )

        session.load()

        return session

    except Exception as error:
        raise RuntimeError(
            f"Could not load {year} {grand_prix} "
            f"{session_type}."
        ) from error


def get_driver_codes(session) -> list[str]:
    """
    Return only drivers with at least one usable timed lap.

    A driver can appear in session metadata without completing a
    valid timed lap. Those drivers should not be offered to modules
    that depend on fastest-lap telemetry.
    """

    drivers = []

    for driver_number in session.drivers:

        try:
            driver = session.get_driver(driver_number)

            abbreviation = driver.get(
                "Abbreviation"
            )

            if not abbreviation:
                continue

            abbreviation = str(abbreviation)

            laps = session.laps.pick_drivers(
                abbreviation
            )

            if laps.empty:
                continue

            if "LapTime" not in laps.columns:
                continue

            timed_laps = laps[
                laps["LapTime"].notna()
            ]

            if timed_laps.empty:
                continue

            fastest_lap = timed_laps.pick_fastest()

            if fastest_lap is None:
                continue

            drivers.append(
                abbreviation
            )

        except Exception:
            # A single imperfect driver record should never stop
            # the remainder of the grid from loading.
            continue

    return sorted(
        set(drivers)
    )


def get_driver_laps(
    session,
    driver: str,
) -> pd.DataFrame:
    """
    Return every recorded lap for a driver.
    """

    laps = session.laps.pick_drivers(
        driver
    )

    if laps.empty:
        raise ValueError(
            f"No laps found for driver {driver}."
        )

    return laps


def get_fastest_lap(
    session,
    driver: str,
):
    """
    Find the driver's fastest usable timed lap.
    """

    laps = get_driver_laps(
        session,
        driver,
    )

    if "LapTime" not in laps.columns:
        raise ValueError(
            f"Lap-time information is unavailable for {driver}."
        )

    timed_laps = laps[
        laps["LapTime"].notna()
    ].copy()

    if timed_laps.empty:
        raise ValueError(
            f"No valid timed laps were found for {driver}."
        )

    fastest_lap = timed_laps.pick_fastest()

    if fastest_lap is None:
        raise ValueError(
            f"No valid fastest lap found for {driver}."
        )

    return fastest_lap


def get_fastest_lap_summary(
    session,
    driver: str,
) -> dict:
    """
    Create a dashboard-friendly fastest-lap summary.
    """

    lap = get_fastest_lap(
        session,
        driver,
    )

    lap_time = lap.get("LapTime")
    lap_number = lap.get("LapNumber")
    compound = lap.get("Compound")

    return {
        "driver": driver,
        "lap_number": (
            int(lap_number)
            if pd.notna(lap_number)
            else None
        ),
        "lap_time_seconds": (
            float(lap_time.total_seconds())
            if pd.notna(lap_time)
            else None
        ),
        "compound": (
            str(compound)
            if pd.notna(compound)
            else "Unknown"
        ),
    }


def get_fastest_lap_telemetry(
    session,
    driver: str,
) -> pd.DataFrame:
    """
    Retrieve telemetry from a driver's fastest lap.
    """

    lap = get_fastest_lap(
        session,
        driver,
    )

    telemetry = lap.get_telemetry()

    if (
        telemetry is None
        or telemetry.empty
    ):
        raise ValueError(
            f"No telemetry available for {driver}."
        )

    return telemetry


if __name__ == "__main__":

    session = load_session(
        year=2025,
        grand_prix="Monaco",
        session_type="Q",
    )

    print("\nSESSION LOADED")
    print("-" * 50)

    print(f"Event: {session.event['EventName']}")
    print(f"Session: {session.name}")

    drivers = get_driver_codes(session)

    print("\nDrivers:")
    print(drivers)

    if not drivers:
        raise RuntimeError(
            "No drivers with usable timed laps were found."
        )

    driver = drivers[0]

    summary = get_fastest_lap_summary(
        session,
        driver,
    )

    print(f"\nFastest lap — {driver}")
    print(summary)

    telemetry = get_fastest_lap_telemetry(
        session,
        driver,
    )

    print("\nTelemetry channels:")
    print(telemetry.columns.tolist())

    print("\nTelemetry preview:")
    print(
        telemetry[
            [
                column
                for column in [
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
                if column in telemetry.columns
            ]
        ].head()
    )
