import plotly.graph_objects as go


def build_tyre_degradation_chart(
    analysis: dict,
):
    """
    Plot actual stint lap times and the fitted
    degradation trend where enough clean data exists.
    """

    stint = analysis[
        "laps"
    ]

    driver = analysis[
        "driver"
    ]

    model_result = analysis[
        "model"
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=stint["TyreLife"],
            y=stint["LapTimeSeconds"],
            mode="markers+lines",
            name=f"{driver} actual",
        )
    )

    if model_result is not None:

        fig.add_trace(
            go.Scatter(
                x=model_result[
                    "tyre_life"
                ],
                y=model_result[
                    "predictions"
                ],
                mode="lines",
                name="Linear trend",
                line=dict(
                    dash="dash",
                ),
            )
        )

    fig.update_layout(
        title=(
            f"{driver} Tyre Degradation"
        ),
        xaxis_title="Tyre Age (laps)",
        yaxis_title="Lap Time (s)",
        template="plotly_dark",
        height=420,
        margin=dict(
            l=40,
            r=20,
            t=55,
            b=40,
        ),
    )

    return fig