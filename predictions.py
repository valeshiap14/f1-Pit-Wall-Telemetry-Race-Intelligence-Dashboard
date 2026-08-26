from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelResult:
    """
    Store a trained model and its evaluation metrics.
    """

    name: str
    model: Pipeline
    mae: float
    rmse: float
    r2: float


def prepare_lap_prediction_data(
    session,
) -> pd.DataFrame:
    """
    Build a session-wide feature table for lap-time prediction.

    The first model focuses on features that are available
    consistently from FastF1 across many sessions.
    """

    laps = session.laps.copy()

    if laps.empty:
        raise ValueError(
            "No lap data is available for model training."
        )

    if "LapTime" not in laps.columns:
        raise ValueError(
            "LapTime is unavailable."
        )

    laps["LapTimeSeconds"] = (
        laps["LapTime"]
        .dt.total_seconds()
    )

    data = pd.DataFrame(
        index=laps.index
    )

    data["LapTimeSeconds"] = (
        laps["LapTimeSeconds"]
    )

    data["Driver"] = (
        laps["Driver"]
        if "Driver" in laps.columns
        else "Unknown"
    )

    data["Compound"] = (
        laps["Compound"]
        if "Compound" in laps.columns
        else "Unknown"
    )

    data["TyreLife"] = pd.to_numeric(
        laps["TyreLife"]
        if "TyreLife" in laps.columns
        else np.nan,
        errors="coerce",
    )

    data["Stint"] = pd.to_numeric(
        laps["Stint"]
        if "Stint" in laps.columns
        else np.nan,
        errors="coerce",
    )

    data["LapNumber"] = pd.to_numeric(
        laps["LapNumber"]
        if "LapNumber" in laps.columns
        else np.nan,
        errors="coerce",
    )

    if "LapStartTime" in laps.columns:
        data["SessionTimeSeconds"] = (
            laps["LapStartTime"]
            .dt.total_seconds()
        )
    else:
        data["SessionTimeSeconds"] = np.nan

    # Invalid or pit-affected laps should not become targets
    # for a normal racing-pace prediction model.
    valid_mask = (
        data["LapTimeSeconds"].notna()
        & (data["LapTimeSeconds"] > 0)
    )

    if "PitInTime" in laps.columns:
        valid_mask &= (
            laps["PitInTime"].isna()
        )

    if "PitOutTime" in laps.columns:
        valid_mask &= (
            laps["PitOutTime"].isna()
        )

    if "Deleted" in laps.columns:
        deleted = (
            laps["Deleted"]
            .fillna(False)
            .astype(bool)
        )

        valid_mask &= ~deleted

    data = data[
        valid_mask
    ].copy()

    # Extremely slow laps are usually traffic, incidents,
    # cool-down laps or other behaviour we do not want
    # dominating the first baseline model.
    data = remove_target_outliers(
        data
    )

    return data.reset_index(
        drop=True
    )


def remove_target_outliers(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove obvious lap-time anomalies using an IQR filter.
    """

    clean = data.copy()

    target = clean[
        "LapTimeSeconds"
    ]

    if len(target) < 10:
        return clean

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    iqr = q3 - q1

    if iqr <= 0:
        return clean

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return clean[
        target.between(
            lower,
            upper,
        )
    ].copy()


def split_by_session_progression(
    data: pd.DataFrame,
    test_fraction: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split laps chronologically instead of randomly.

    This reduces leakage because future laps are not used
    to predict earlier laps from the same session.
    """

    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError(
            "test_fraction should be between 0.1 and 0.5."
        )

    ordered = data.sort_values(
        "SessionTimeSeconds"
    ).reset_index(
        drop=True
    )

    split_index = int(
        len(ordered)
        * (1 - test_fraction)
    )

    if split_index < 5:
        raise ValueError(
            "Not enough training data."
        )

    train = ordered.iloc[
        :split_index
    ].copy()

    test = ordered.iloc[
        split_index:
    ].copy()

    if test.empty:
        raise ValueError(
            "Not enough test data."
        )

    return train, test


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build a reusable preprocessing pipeline.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelResult:
    """
    Train a model and calculate regression metrics.
    """

    pipeline.fit(
        x_train,
        y_train,
    )

    predictions = pipeline.predict(
        x_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    return ModelResult(
        name=name,
        model=pipeline,
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
    )


def train_lap_time_models(
    session,
) -> dict:
    """
    Train Linear Regression, Ridge and Random Forest models
    for lap-time prediction and compare their performance.
    """

    data = prepare_lap_prediction_data(
        session
    )

    if len(data) < 20:
        raise ValueError(
            "At least 20 clean laps are required "
            "for meaningful model comparison."
        )

    train, test = (
        split_by_session_progression(
            data
        )
    )

    target = "LapTimeSeconds"

    numerical_features = [
        "TyreLife",
        "Stint",
        "LapNumber",
        "SessionTimeSeconds",
    ]

    categorical_features = [
        "Driver",
        "Compound",
    ]

    features = (
        numerical_features
        + categorical_features
    )

    x_train = train[
        features
    ]

    y_train = train[
        target
    ]

    x_test = test[
        features
    ]

    y_test = test[
        target
    ]

    linear_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numerical_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    ridge_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numerical_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                Ridge(
                    alpha=1.0
                ),
            ),
        ]
    )

    forest_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numerical_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    results = [
        evaluate_model(
            "Linear Regression",
            linear_pipeline,
            x_train,
            y_train,
            x_test,
            y_test,
        ),
        evaluate_model(
            "Ridge Regression",
            ridge_pipeline,
            x_train,
            y_train,
            x_test,
            y_test,
        ),
        evaluate_model(
            "Random Forest",
            forest_pipeline,
            x_train,
            y_train,
            x_test,
            y_test,
        ),
    ]

    results = sorted(
        results,
        key=lambda result: result.mae,
    )

    return {
        "data": data,
        "train": train,
        "test": test,
        "results": results,
        "best_model": results[0],
        "features": features,
    }


def build_model_comparison_table(
    training_result: dict,
) -> pd.DataFrame:
    """
    Convert model evaluation results into a dashboard table.
    """

    rows = []

    for result in training_result[
        "results"
    ]:

        rows.append(
            {
                "Model": result.name,
                "MAE": result.mae,
                "RMSE": result.rmse,
                "R²": result.r2,
            }
        )

    return pd.DataFrame(
        rows
    )


def predict_next_lap(
    training_result: dict,
    driver: str,
    compound: str,
    tyre_life: float,
    stint: float,
    lap_number: float,
    session_time_seconds: float,
) -> float:
    """
    Predict a future lap using the best evaluated model.
    """

    best_model = training_result[
        "best_model"
    ]

    prediction_input = pd.DataFrame(
        [
            {
                "TyreLife": tyre_life,
                "Stint": stint,
                "LapNumber": lap_number,
                "SessionTimeSeconds": session_time_seconds,
                "Driver": driver,
                "Compound": compound,
            }
        ]
    )

    prediction = (
        best_model.model.predict(
            prediction_input
        )[0]
    )

    return float(
        prediction
    )