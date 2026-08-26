import plotly.graph_objects as go


def build_session_pace_chart(
    analysis: dict,
):
    """
    Visualize how representative lap pace changes
    throughout the session.
    """

    data = analysis["data"]
    trend = analysis["trend"]

    fig = go.Figure()

    # Each point represents one cleaned lap from the session.
    fig.add_trace(
        go.Scatter(
            x=(
                data["SessionTimeSeconds"]
                / 60
            ),
            y=data["LapTimeSeconds"],
            mode="markers",
            name="Clean laps",
            text=data["Driver"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Session time: %{x:.1f} min<br>"
                "Lap time: %{y:.3f} s"
                "<extra></extra>"
            ),
        )
    )

    # The fitted line shows the broad session-wide pace trend.
    fig.add_trace(
        go.Scatter(
            x=(
                data["SessionTimeSeconds"]
                / 60
            ),
            y=trend["prediction"],
            mode="lines",
            name="Session trend",
            line=dict(
                dash="dash",
                width=3,
            ),
        )
    )

    fig.update_layout(
        title="Session Pace Evolution",
        xaxis_title="Session Time (minutes)",
        yaxis_title="Lap Time (seconds)",
        template="plotly_dark",
        height=430,
        margin=dict(
            l=50,
            r=25,
            t=60,
            b=50,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def build_temperature_pace_chart(
    analysis: dict,
):
    """
    Plot track temperature against cleaned lap times.

    The graph shows association only and should not be
    interpreted as evidence that temperature caused pace changes.
    """

    data = analysis["data"]

    if "TrackTemp" not in data.columns:
        raise ValueError(
            "Track-temperature data is unavailable."
        )

    usable = data.dropna(
        subset=[
            "TrackTemp",
            "LapTimeSeconds",
        ]
    )

    if usable.empty:
        raise ValueError(
            "No usable track-temperature data exists."
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=usable["TrackTemp"],
            y=usable["LapTimeSeconds"],
            mode="markers",
            name="Session laps",
            text=usable["Driver"],
            customdata=(
                usable["SessionTimeSeconds"]
                / 60
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Track temperature: %{x:.1f} °C<br>"
                "Lap time: %{y:.3f} s<br>"
                "Session time: %{customdata:.1f} min"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Track Temperature vs Lap Time",
        xaxis_title="Track Temperature (°C)",
        yaxis_title="Lap Time (seconds)",
        template="plotly_dark",
        height=400,
        margin=dict(
            l=50,
            r=25,
            t=60,
            b=50,
        ),
    )

    return fig