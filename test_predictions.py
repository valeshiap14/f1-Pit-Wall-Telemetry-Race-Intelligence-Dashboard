import numpy as np
import pandas as pd

from src.predictions import (
    build_preprocessor,
    remove_target_outliers,
    split_by_session_progression,
)


def build_prediction_data(
    rows: int = 40,
):
    """
    Create deterministic ML data without downloading F1 data.

    The synthetic dataset follows session progression so we
    can verify that our chronological split prevents future
    observations leaking into the training set.
    """

    return pd.DataFrame(
        {
            "LapTimeSeconds": (
                90
                + np.arange(rows)
                * 0.05
            ),
            "TyreLife": (
                np.arange(rows)
                % 15
            ) + 1,
            "Stint": (
                np.arange(rows)
                // 15
            ) + 1,
            "LapNumber": np.arange(
                1,
                rows + 1,
            ),
            "SessionTimeSeconds": (
                np.arange(rows)
                * 100
            ),
            "Driver": [
                "AAA"
                if index % 2 == 0
                else "BBB"
                for index in range(rows)
            ],
            "Compound": [
                "MEDIUM"
                if index < rows / 2
                else "HARD"
                for index in range(rows)
            ],
        }
    )


def test_chronological_split_creates_train_and_test():
    data = build_prediction_data()

    train, test = (
        split_by_session_progression(
            data,
            test_fraction=0.25,
        )
    )

    assert len(train) > 0
    assert len(test) > 0


def test_training_data_occurs_before_test_data():
    data = build_prediction_data()

    train, test = (
        split_by_session_progression(
            data,
            test_fraction=0.25,
        )
    )

    assert (
        train[
            "SessionTimeSeconds"
        ].max()
        <
        test[
            "SessionTimeSeconds"
        ].min()
    )


def test_split_uses_expected_proportion():
    data = build_prediction_data(
        rows=40
    )

    train, test = (
        split_by_session_progression(
            data,
            test_fraction=0.25,
        )
    )

    assert len(train) == 30
    assert len(test) == 10


def test_outlier_removal_detects_extreme_target():
    data = build_prediction_data()

    bad_row = data.iloc[
        [0]
    ].copy()

    bad_row[
        "LapTimeSeconds"
    ] = 200

    data = pd.concat(
        [
            data,
            bad_row,
        ],
        ignore_index=True,
    )

    cleaned = remove_target_outliers(
        data
    )

    assert (
        cleaned[
            "LapTimeSeconds"
        ].max()
        < 200
    )


def test_preprocessor_can_transform_features():
    data = build_prediction_data()

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

    preprocessor = (
        build_preprocessor(
            numerical_features,
            categorical_features,
        )
    )

    features = data[
        numerical_features
        + categorical_features
    ]

    transformed = (
        preprocessor.fit_transform(
            features
        )
    )

    assert (
        transformed.shape[0]
        == len(data)
    )


def test_preprocessor_handles_missing_values():
    data = build_prediction_data()

    data.loc[
        3,
        "TyreLife",
    ] = np.nan

    data.loc[
        5,
        "Compound",
    ] = None

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

    preprocessor = (
        build_preprocessor(
            numerical_features,
            categorical_features,
        )
    )

    features = data[
        numerical_features
        + categorical_features
    ]

    transformed = (
        preprocessor.fit_transform(
            features
        )
    )

    assert (
        transformed.shape[0]
        == len(data)
    )