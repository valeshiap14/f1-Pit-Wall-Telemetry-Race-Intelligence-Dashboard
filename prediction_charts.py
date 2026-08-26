import plotly.graph_objects as go


def build_model_comparison_chart(
    comparison_table,
):
    """
    Compare model MAE values.

    Lower MAE indicates better predictive accuracy
    on the held-out portion of the session.
    """

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_table[
                "Model"
            ],
            y=comparison_table[
                "MAE"
            ],
            text=comparison_table[
                "MAE"
            ].round(3),
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Lap-Time Model Comparison",
        xaxis_title="Model",
        yaxis_title="MAE (seconds)",
        template="plotly_dark",
        height=380,
    )

    return fig


def build_actual_vs_predicted_chart(
    training_result: dict,
):
    """
    Plot actual test lap times against predictions
    from the best-performing model.
    """

    best_model = training_result[
        "best_model"
    ]

    test = training_result[
        "test"
    ].copy()

    features = training_result[
        "features"
    ]

    predictions = (
        best_model.model.predict(
            test[
                features
            ]
        )
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test[
                "LapNumber"
            ],
            y=test[
                "LapTimeSeconds"
            ],
            mode="lines+markers",
            name="Actual",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test[
                "LapNumber"
            ],
            y=predictions,
            mode="lines+markers",
            name="Predicted",
        )
    )

    fig.update_layout(
        title=(
            f"Actual vs Predicted — "
            f"{best_model.name}"
        ),
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        template="plotly_dark",
        height=420,
    )

    return fig