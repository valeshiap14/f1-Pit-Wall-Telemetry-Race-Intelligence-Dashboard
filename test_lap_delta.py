import numpy as np
import pandas as pd

from src.telemetry import align_driver_telemetry


def build_fake_telemetry(
    distances,
    times,
    speeds,
):
    """
    Create a small telemetry dataframe for repeatable tests.

    Synthetic data is better than downloading a real F1 session
    because unit tests should be quick and deterministic.
    """

    return pd.DataFrame(
        {
            "Distance": distances,
            "Time": pd.to_timedelta(
                times,
                unit="s",
            ),
            "Speed": speeds,
            "Throttle": [100] * len(distances),
            "Brake": [False] * len(distances),
            "RPM": [10000] * len(distances),
            "nGear": [7] * len(distances),
            "DRS": [12] * len(distances),
        }
    )


def test_aligned_telemetry_has_expected_length():
    telemetry_a = build_fake_telemetry(
        distances=[0, 100, 200, 300],
        times=[0, 2, 4, 6],
        speeds=[200, 210, 220, 230],
    )

    telemetry_b = build_fake_telemetry(
        distances=[0, 100, 200, 300],
        times=[0, 2.1, 4.2, 6.3],
        speeds=[198, 208, 218, 228],
    )

    aligned = align_driver_telemetry(
        telemetry_a,
        telemetry_b,
        samples=500,
    )

    assert len(aligned) == 500


def test_distance_grid_starts_at_zero():
    telemetry_a = build_fake_telemetry(
        [0, 100, 200],
        [0, 2, 4],
        [200, 210, 220],
    )

    telemetry_b = build_fake_telemetry(
        [0, 100, 200],
        [0, 2, 4],
        [200, 210, 220],
    )

    aligned = align_driver_telemetry(
        telemetry_a,
        telemetry_b,
        samples=200,
    )

    assert np.isclose(
        aligned["Distance"].iloc[0],
        0,
    )


def test_delta_is_zero_for_identical_laps():
    telemetry_a = build_fake_telemetry(
        [0, 100, 200, 300],
        [0, 2, 4, 6],
        [200, 210, 220, 230],
    )

    telemetry_b = build_fake_telemetry(
        [0, 100, 200, 300],
        [0, 2, 4, 6],
        [200, 210, 220, 230],
    )

    aligned = align_driver_telemetry(
        telemetry_a,
        telemetry_b,
        samples=300,
    )

    assert np.allclose(
        aligned["Delta"],
        0,
        atol=1e-9,
    )


def test_positive_delta_means_driver_a_is_slower():
    telemetry_a = build_fake_telemetry(
        [0, 100, 200, 300],
        [0, 2.2, 4.4, 6.6],
        [200, 210, 220, 230],
    )

    telemetry_b = build_fake_telemetry(
        [0, 100, 200, 300],
        [0, 2, 4, 6],
        [200, 210, 220, 230],
    )

    aligned = align_driver_telemetry(
        telemetry_a,
        telemetry_b,
        samples=300,
    )

    assert (
        aligned["Delta"].iloc[-1]
        > 0
    )


def test_alignment_uses_shorter_lap_distance():
    telemetry_a = build_fake_telemetry(
        [0, 100, 200, 300],
        [0, 2, 4, 6],
        [200, 210, 220, 230],
    )

    telemetry_b = build_fake_telemetry(
        [0, 100, 200, 250],
        [0, 2, 4, 5],
        [200, 210, 220, 225],
    )

    aligned = align_driver_telemetry(
        telemetry_a,
        telemetry_b,
        samples=250,
    )

    assert np.isclose(
        aligned["Distance"].max(),
        250,
    )