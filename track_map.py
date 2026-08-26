from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def prepare_track_data(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean the GPS coordinates used to reconstruct the circuit.
    """

    required_columns = ["X", "Y"]

    missing = [
        column
        for column in required_columns
        if column not in telemetry.columns
    ]

    if missing:
        raise ValueError(
            "Track position data is unavailable for this lap."
        )

    track = telemetry.copy()

    track["X"] = pd.to_numeric(
        track["X"],
        errors="coerce",
    )

    track["Y"] = pd.to_numeric(
        track["Y"],
        errors="coerce",
    )

    track = track.dropna(
        subset=["X", "Y"]
    )

    if track.empty:
        raise ValueError(
            "No usable GPS coordinates were found."
        )

    return track.reset_index(drop=True)


def build_track_map(
    telemetry: pd.DataFrame,
    driver: str,
    metric: str = "Speed",
) -> go.Figure:
    """
    Build an interactive circuit map coloured by telemetry.
    """

    track = prepare_track_data(
        telemetry
    )

    metric_options = {
        "Speed": {
            "column": "Speed",
            "label": "Speed",
            "unit": "km/h",
        },
        "Throttle": {
            "column": "Throttle",
            "label": "Throttle",
            "unit": "%",
        },
        "Gear": {
            "column": "nGear",
            "label": "Gear",
            "unit": "",
        },
        "RPM": {
            "column": "RPM",
            "label": "RPM",
            "unit": "rpm",
        },
    }

    if metric not in metric_options:
        raise ValueError(
            f"Unsupported track metric: {metric}"
        )

    metric_info = metric_options[metric]
    column = metric_info["column"]

    if column not in track.columns:
        raise ValueError(
            f"{metric} telemetry is unavailable."
        )

    track["MapValue"] = pd.to_numeric(
        track[column],
        errors="coerce",
    )

    track = track.dropna(
        subset=["MapValue"]
    )

    if track.empty:
        raise ValueError(
            f"No usable {metric} values were found."
        )

    # Distance is optional for the hover tooltip.
    if "Distance" in track.columns:
        distance = pd.to_numeric(
            track["Distance"],
            errors="coerce",
        )
    else:
        distance = pd.Series(
            np.nan,
            index=track.index,
        )

    fig = go.Figure()

    # Draw the circuit underneath the telemetry points so the
    # overall shape remains visible even where points are sparse.
    fig.add_trace(
        go.Scatter(
            x=track["X"],
            y=track["Y"],
            mode="lines",
            line=dict(
                width=9,
                color="#242a33",
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=track["X"],
            y=track["Y"],
            mode="markers",
            marker=dict(
                size=6,
                color=track["MapValue"],
                colorscale="Turbo",
                showscale=True,
                colorbar=dict(
                    title=(
                        f"{metric_info['label']} "
                        f"{metric_info['unit']}"
                    ).strip()
                ),
            ),
            customdata=np.column_stack(
                [
                    track["MapValue"],
                    distance,
                ]
            ),
            hovertemplate=(
                f"<b>{driver}</b><br>"
                f"{metric_info['label']}: "
                "%{customdata[0]:.1f} "
                f"{metric_info['unit']}<br>"
                "Distance: %{customdata[1]:.0f} m"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"{driver} Circuit Map — {metric}",
        template="plotly_dark",
        height=650,
        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10,
        ),
        xaxis=dict(
            visible=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            visible=False,
        ),
        paper_bgcolor="#080b10",
        plot_bgcolor="#080b10",
    )

    return fig