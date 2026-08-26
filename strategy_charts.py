import plotly.graph_objects as go


def build_stint_timeline(
    stints_a,
    stints_b,
    driver_a: str,
    driver_b: str,
):
    """
    Plot both drivers' stint timelines across race laps.
    """

    fig = go.Figure()

    for driver, stints, y_value in [
        (driver_a, stints_a, 1),
        (driver_b, stints_b, 0),
    ]:

        for _, stint in stints.iterrows():

            start = stint[
                "StartLap"
            ]

            end = stint[
                "EndLap"
            ]

            fig.add_trace(
                go.Scatter(
                    x=[
                        start,
                        end,
                    ],
                    y=[
                        y_value,
                        y_value,
                    ],
                    mode="lines+markers",
                    line=dict(
                        width=14,
                    ),
                    name=(
                        f"{driver} "
                        f"{stint['Compound']}"
                    ),
                    hovertemplate=(
                        f"<b>{driver}</b><br>"
                        f"Compound: {stint['Compound']}<br>"
                        f"Laps: {start}-{end}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title="Stint Timeline",
        xaxis_title="Race Lap",
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=[
                driver_b,
                driver_a,
            ],
        ),
        template="plotly_dark",
        height=320,
        showlegend=False,
    )

    return fig


def build_strategy_pace_chart(
    laps_a,
    laps_b,
    driver_a: str,
    driver_b: str,
):
    """
    Compare lap pace throughout the race.
    """

    fig = go.Figure()

    for laps, driver in [
        (laps_a, driver_a),
        (laps_b, driver_b),
    ]:

        usable = laps[
            laps[
                "LapTimeSeconds"
            ].notna()
        ]

        fig.add_trace(
            go.Scatter(
                x=usable[
                    "LapNumber"
                ],
                y=usable[
                    "LapTimeSeconds"
                ],
                mode="lines+markers",
                name=driver,
            )
        )

    fig.update_layout(
        title="Race Pace Comparison",
        xaxis_title="Race Lap",
        yaxis_title="Lap Time (s)",
        template="plotly_dark",
        height=420,
    )

    return fig