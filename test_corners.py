import numpy as np
import pandas as pd

from src.corners import (
    analyse_corner,
    detect_corner_regions,
)


def build_corner_telemetry():
    """
    Build a synthetic lap containing one obvious braking
    and corner-speed minimum.
    """

    distance = np.linspace(
        0,
        500,
        101,
    )

    speed = np.full(
        len(distance),
        250.0,
    )

    # Create an obvious corner around 250 metres.
    speed -= (
        120
        * np.exp(
            -(
                (
                    distance - 250
                )
                / 50
            )
            ** 2
        )
    )

    brake = (
        (distance >= 180)
        & (distance <= 250)
    )

    throttle = np.where(
        distance < 250,
        30,
        100,
    )

    return pd.DataFrame(
        {
            "Distance": distance,
            "Speed": speed,
            "Brake": brake,
            "Throttle": throttle,
            "nGear": np.where(
                speed < 170,
                3,
                7,
            ),
        }
    )


def test_detects_corner_region():
    telemetry = build_corner_telemetry()

    corners = detect_corner_regions(
        telemetry,
        minimum_speed_drop=20,
    )

    assert len(corners) >= 1


def test_detected_corner_is_near_speed_minimum():
    telemetry = build_corner_telemetry()

    corners = detect_corner_regions(
        telemetry,
        minimum_speed_drop=20,
    )

    first_corner = corners[0]

    assert (
        200
        <= first_corner[
            "centre_distance"
        ]
        <= 300
    )


def test_corner_analysis_returns_minimum_speed():
    telemetry = build_corner_telemetry()

    corner = {
        "corner_number": 1,
        "start_distance": 130,
        "centre_distance": 250,
        "end_distance": 390,
    }

    result = analyse_corner(
        telemetry,
        corner,
    )

    assert (
        result[
            "minimum_speed"
        ]
        < 150
    )


def test_corner_analysis_finds_brake_point():
    telemetry = build_corner_telemetry()

    corner = {
        "corner_number": 1,
        "start_distance": 130,
        "centre_distance": 250,
        "end_distance": 390,
    }

    result = analyse_corner(
        telemetry,
        corner,
    )

    assert (
        result[
            "brake_distance"
        ]
        is not None
    )


def test_corner_analysis_finds_full_throttle_point():
    telemetry = build_corner_telemetry()

    corner = {
        "corner_number": 1,
        "start_distance": 130,
        "centre_distance": 250,
        "end_distance": 390,
    }

    result = analyse_corner(
        telemetry,
        corner,
    )

    assert (
        result[
            "full_throttle_distance"
        ]
        is not None
    )