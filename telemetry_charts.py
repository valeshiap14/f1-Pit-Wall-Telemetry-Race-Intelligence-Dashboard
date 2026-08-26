import plotly.graph_objects as go


def build_speed_chart(aligned, driver_a: str, driver_b: str):
    """
    Compare both drivers' speed traces across lap distance.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Speed_A"],
            mode="lines",
            name=driver_a,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Speed_B"],
            mode="lines",
            name=driver_b,
        )
    )

    fig.update_layout(
        title="Speed Comparison",
        xaxis_title="Lap Distance (m)",
        yaxis_title="Speed (km/h)",
        template="plotly_dark",
        height=420,
        margin=dict(l=40, r=20, t=55, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def build_throttle_chart(aligned, driver_a: str, driver_b: str):
    """
    Compare throttle application across lap distance.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Throttle_A"],
            mode="lines",
            name=driver_a,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Throttle_B"],
            mode="lines",
            name=driver_b,
        )
    )

    fig.update_layout(
        title="Throttle Comparison",
        xaxis_title="Lap Distance (m)",
        yaxis_title="Throttle (%)",
        template="plotly_dark",
        height=300,
        margin=dict(l=40, r=20, t=55, b=40),
    )

    fig.update_yaxes(
        range=[0, 105]
    )

    return fig


def build_brake_chart(aligned, driver_a: str, driver_b: str):
    """
    Show where each driver applies the brakes.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Brake_A"],
            mode="lines",
            name=driver_a,
            line_shape="hv",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Brake_B"],
            mode="lines",
            name=driver_b,
            line_shape="hv",
        )
    )

    fig.update_layout(
        title="Brake Application",
        xaxis_title="Lap Distance (m)",
        yaxis_title="Brake",
        template="plotly_dark",
        height=250,
        margin=dict(l=40, r=20, t=55, b=40),
    )

    fig.update_yaxes(
        tickvals=[0, 1],
        ticktext=["OFF", "ON"],
        range=[-0.1, 1.1],
    )

    return fig


def build_rpm_chart(aligned, driver_a: str, driver_b: str):
    """
    Compare engine RPM across the lap.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["RPM_A"],
            mode="lines",
            name=driver_a,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["RPM_B"],
            mode="lines",
            name=driver_b,
        )
    )

    fig.update_layout(
        title="Engine RPM",
        xaxis_title="Lap Distance (m)",
        yaxis_title="RPM",
        template="plotly_dark",
        height=320,
        margin=dict(l=40, r=20, t=55, b=40),
    )

    return fig


def build_gear_chart(aligned, driver_a: str, driver_b: str):
    """
    Show gear selection throughout the lap.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["nGear_A"],
            mode="lines",
            name=driver_a,
            line_shape="hv",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["nGear_B"],
            mode="lines",
            name=driver_b,
            line_shape="hv",
        )
    )

    fig.update_layout(
        title="Gear Selection",
        xaxis_title="Lap Distance (m)",
        yaxis_title="Gear",
        template="plotly_dark",
        height=270,
        margin=dict(l=40, r=20, t=55, b=40),
    )

    fig.update_yaxes(
        dtick=1
    )

    return fig


def build_delta_chart(aligned, driver_a: str, driver_b: str):
    """
    Show cumulative lap-time difference across the lap.

    Positive delta means Driver A has taken longer to reach
    that point, so Driver B is ahead.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=aligned["Distance"],
            y=aligned["Delta"],
            mode="lines",
            name=f"{driver_a} - {driver_b}",
            fill="tozeroy",
        )
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
    )

    fig.update_layout(
        title="Lap Delta",
        xaxis_title="Lap Distance (m)",
        yaxis_title="Delta (s)",
        template="plotly_dark",
        height=340,
        margin=dict(l=40, r=20, t=55, b=40),
    )

    return fig