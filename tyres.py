from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_driver_laps(
    session,
    driver: str,
) -> pd.DataFrame:
    """
    Prepare a driver's lap data for stint and tyre analysis.

    The function keeps the raw lap information but adds
    numerical lap time and quality flags used later.
    """

    laps = session.laps.pick_drivers(
        driver
    ).copy()

    if laps.empty:
        raise ValueError(
            f"No laps were found for {driver}."
        )

    if "LapTime" not in laps.columns:
        raise ValueError(
            "Lap-time data is unavailable."
        )

    laps["LapTimeSeconds"] = (
        laps["LapTime"]
        .dt.total_seconds()
    )

    laps["IsPitAffected"] = False

    if "PitInTime" in laps.columns:
        laps["IsPitAffected"] = (
            laps["IsPitAffected"]
            | laps["PitInTime"].notna()
        )

    if "PitOutTime" in laps.columns:
        laps["IsPitAffected"] = (
            laps["IsPitAffected"]
            | laps["PitOutTime"].notna()
        )

    laps["IsValidLap"] = (
        laps["LapTimeSeconds"].notna()
        & (laps["LapTimeSeconds"] > 0)
    )

    return laps


def remove_lap_time_outliers(
    laps: pd.DataFrame,
    column: str = "LapTimeSeconds",
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Remove obvious lap-time outliers using the IQR method.

    Outlier removal is applied within a stint later so that
    genuinely different tyre compounds are not mixed together.
    """

    clean = laps.copy()

    values = pd.to_numeric(
        clean[column],
        errors="coerce",
    )

    valid = values.dropna()

    if len(valid) < 4:
        return clean

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)

    iqr = q3 - q1

    if iqr <= 0:
        return clean

    lower_bound = (
        q1 - multiplier * iqr
    )

    upper_bound = (
        q3 + multiplier * iqr
    )

    return clean[
        values.between(
            lower_bound,
            upper_bound,
        )
    ].copy()


def prepare_stint(
    laps: pd.DataFrame,
    stint_number: int,
) -> pd.DataFrame:
    """
    Extract and clean one tyre stint.

    Pit laps, invalid laps and obvious timing anomalies
    are excluded from degradation modelling.
    """

    if "Stint" not in laps.columns:
        raise ValueError(
            "Stint information is unavailable."
        )

    stint = laps[
        laps["Stint"]
        == stint_number
    ].copy()

    if stint.empty:
        raise ValueError(
            f"Stint {stint_number} contains no laps."
        )

    stint = stint[
        stint["IsValidLap"]
    ]

    stint = stint[
        ~stint["IsPitAffected"]
    ]

    if "Deleted" in stint.columns:
        deleted = (
            stint["Deleted"]
            .fillna(False)
            .astype(bool)
        )

        stint = stint[
            ~deleted
        ]

    stint = remove_lap_time_outliers(
        stint
    )

    stint = stint.sort_values(
        "LapNumber"
    ).reset_index(
        drop=True
    )

    return stint


def available_stints(
    laps: pd.DataFrame,
) -> list[int]:
    """Return valid stint numbers for a driver."""

    if "Stint" not in laps.columns:
        return []

    stints = pd.to_numeric(
        laps["Stint"],
        errors="coerce",
    ).dropna()

    return sorted(
        stints.astype(int).unique().tolist()
    )


def fit_linear_degradation(
    stint: pd.DataFrame,
) -> dict:
    """
    Fit a simple interpretable tyre-degradation model.

    Lap time is modelled against tyre life. The slope estimates
    the average seconds lost per additional tyre lap.
    """

    required = [
        "TyreLife",
        "LapTimeSeconds",
    ]

    missing = [
        column
        for column in required
        if column not in stint.columns
    ]

    if missing:
        raise ValueError(
            "TyreLife or lap-time data is unavailable."
        )

    modelling_data = stint[
        required
    ].copy()

    modelling_data["TyreLife"] = pd.to_numeric(
        modelling_data["TyreLife"],
        errors="coerce",
    )

    modelling_data["LapTimeSeconds"] = pd.to_numeric(
        modelling_data["LapTimeSeconds"],
        errors="coerce",
    )

    modelling_data = modelling_data.dropna()

    if len(modelling_data) < 3:
        raise ValueError(
            "At least three clean laps are required "
            "for degradation modelling."
        )

    x = modelling_data[
        ["TyreLife"]
    ].to_numpy()

    y = modelling_data[
        "LapTimeSeconds"
    ].to_numpy()

    model = LinearRegression()

    model.fit(
        x,
        y,
    )

    predictions = model.predict(
        x
    )

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            predictions,
        )
    )

    return {
        "model": model,
        "seconds_per_lap": float(
            model.coef_[0]
        ),
        "intercept": float(
            model.intercept_
        ),
        "mae": float(
            mae
        ),
        "rmse": float(
            rmse
        ),
        "predictions": predictions,
        "tyre_life": modelling_data[
            "TyreLife"
        ].to_numpy(),
        "actual_lap_times": y,
    }


def tyre_stint_summary(
    stint: pd.DataFrame,
) -> dict:
    """
    Calculate descriptive statistics for one tyre stint.
    """

    if stint.empty:
        raise ValueError(
            "Cannot summarize an empty stint."
        )

    lap_times = pd.to_numeric(
        stint["LapTimeSeconds"],
        errors="coerce",
    ).dropna()

    if lap_times.empty:
        raise ValueError(
            "No usable lap times exist in this stint."
        )

    compound = "Unknown"

    if "Compound" in stint.columns:
        compounds = (
            stint["Compound"]
            .dropna()
            .astype(str)
        )

        if not compounds.empty:
            compound = (
                compounds.mode().iloc[0]
            )

    tyre_age_start = None
    tyre_age_end = None

    if "TyreLife" in stint.columns:
        tyre_life = pd.to_numeric(
            stint["TyreLife"],
            errors="coerce",
        ).dropna()

        if not tyre_life.empty:
            tyre_age_start = float(
                tyre_life.min()
            )

            tyre_age_end = float(
                tyre_life.max()
            )

    return {
        "compound": compound,
        "clean_laps": int(
            len(lap_times)
        ),
        "best_lap": float(
            lap_times.min()
        ),
        "average_lap": float(
            lap_times.mean()
        ),
        "median_lap": float(
            lap_times.median()
        ),
        "stint_length": int(
            len(stint)
        ),
        "tyre_age_start": tyre_age_start,
        "tyre_age_end": tyre_age_end,
    }


def analyse_driver_stint(
    session,
    driver: str,
    stint_number: int,
) -> dict:
    """
    Run the complete tyre analysis for one driver and stint.
    """

    laps = prepare_driver_laps(
        session,
        driver,
    )

    stint = prepare_stint(
        laps,
        stint_number,
    )

    summary = tyre_stint_summary(
        stint
    )

    model_result = None

    try:
        model_result = (
            fit_linear_degradation(
                stint
            )
        )

    except ValueError:
        model_result = None

    return {
        "driver": driver,
        "stint_number": stint_number,
        "laps": stint,
        "summary": summary,
        "model": model_result,
    }