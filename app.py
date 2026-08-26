from __future__ import annotations

import fastf1
import pandas as pd
import streamlit as st

from config.settings import (
    APP_ICON,
    APP_LAYOUT,
    APP_SUBTITLE,
    APP_TITLE,
    DEFAULT_SEASON,
    DISCLAIMER,
    SESSION_OPTIONS,
    SUPPORTED_SEASONS,
)

from src.cleaning import (
    add_lap_quality_flags,
    attach_quality_reason,
    data_quality_summary,
    flag_lap_time_outliers,
)

from src.corners import generate_corner_insight

from src.database import (
    save_insight_report,
    save_tyre_stint_result,
)

from src.insights import (
    build_race_engineer_report,
    flatten_report,
)

from src.prediction_charts import (
    build_actual_vs_predicted_chart,
    build_model_comparison_chart,
)

from src.predictions import (
    predict_next_lap,
)

from src.session_service import (
    build_corner_analysis,
    build_driver_comparison,
    build_prediction_analysis,
    build_strategy_analysis,
    build_track_analysis,
    build_tyre_analysis,
    get_session_drivers,
    load_session_context,
)

from src.strategy_charts import (
    build_stint_timeline,
    build_strategy_pace_chart,
)

from src.telemetry_charts import (
    build_brake_chart,
    build_delta_chart,
    build_gear_chart,
    build_rpm_chart,
    build_speed_chart,
    build_throttle_chart,
)

from src.track_evolution import (
    generate_track_evolution_insight,
)

from src.track_evolution_charts import (
    build_session_pace_chart,
    build_temperature_pace_chart,
)

from src.track_map import build_track_map

from src.tyre_charts import (
    build_tyre_degradation_chart,
)

from src.tyres import (
    available_stints,
    prepare_driver_laps,
)

from src.utils import (
    format_degradation,
    format_lap_time,
    format_percentage,
    format_seconds,
    format_speed,
    format_temperature,
    round_dataframe,
)

from src.weather import (
    build_weather_insight,
    get_weather_summary,
)


# The page configuration must be the first Streamlit UI command.
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded",
)


# The interface intentionally resembles an engineering workstation
# rather than Streamlit's default application appearance.
st.html(
    """
    <style>

    :root {
        --bg: #070a0f;
        --panel: #0d1219;
        --panel-soft: #111720;
        --border: #232b35;
        --border-soft: #1a212a;
        --text: #f4f6f8;
        --muted: #808b99;
        --red: #e10600;
        --green: #37d996;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% -15%,
                rgba(225, 6, 0, 0.10),
                transparent 28%
            ),
            #070a0f;
    }

    .block-container {
        max-width: 1700px;
        padding-top: 1.1rem;
        padding-bottom: 4rem;
    }

    section[data-testid="stSidebar"] {
        background: #0a0e14;
        border-right: 1px solid #1d242e;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    div[data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                #111720,
                #0b1016
            );
        border: 1px solid #232b35;
        border-radius: 10px;
        padding: 17px;
    }

    div[data-testid="stMetricLabel"] {
        color: #7f8a98;
        font-size: 0.72rem;
        letter-spacing: 0.06rem;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff;
        font-family: "Courier New", monospace;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #202732;
        padding: 4px 0 17px 0;
        margin-bottom: 18px;
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .f1-badge {
        background: #e10600;
        color: white;
        font-weight: 900;
        font-size: 0.8rem;
        letter-spacing: 0.08rem;
        padding: 9px 11px;
        border-radius: 4px;
    }

    .brand-title {
        color: white;
        font-size: 1.55rem;
        font-weight: 750;
        letter-spacing: 0.03rem;
    }

    .brand-subtitle {
        color: #737f8d;
        font-size: 0.72rem;
        letter-spacing: 0.08rem;
        margin-top: 2px;
    }

    .system-status {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #8994a2;
        font-size: 0.68rem;
        letter-spacing: 0.07rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #37d996;
        box-shadow: 0 0 8px rgba(55, 217, 150, 0.7);
    }

    .session-card {
        background:
            linear-gradient(
                105deg,
                #111720,
                #0b1016
            );
        border: 1px solid #222a34;
        border-left: 3px solid #e10600;
        border-radius: 8px;
        padding: 19px 22px;
        margin-bottom: 22px;
    }

    .session-kicker {
        color: #e10600;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 0.11rem;
    }

    .session-title {
        color: white;
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 5px;
    }

    .session-meta {
        color: #818b99;
        font-size: 0.79rem;
        margin-top: 6px;
    }

    .section-title {
        color: white;
        font-size: 1rem;
        font-weight: 720;
        border-left: 3px solid #e10600;
        padding-left: 10px;
        margin-top: 22px;
        margin-bottom: 15px;
        letter-spacing: 0.02rem;
    }

    .info-panel {
        background: #0d131b;
        border: 1px solid #222a34;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 14px;
    }

    .info-label {
        color: #788391;
        font-size: 0.68rem;
        letter-spacing: 0.08rem;
        margin-bottom: 6px;
    }

    .info-content {
        color: #edf0f4;
        font-size: 0.88rem;
        line-height: 1.65;
    }

    .landing-card {
        background: #0d131b;
        border: 1px solid #222a34;
        border-radius: 8px;
        padding: 24px;
        min-height: 160px;
    }

    .landing-title {
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .landing-copy {
        color: #778290;
        font-size: 0.8rem;
        line-height: 1.6;
        margin-top: 9px;
    }

    .footer {
        margin-top: 40px;
        border-top: 1px solid #1c232c;
        padding-top: 18px;
        color: #56616f;
        text-align: center;
        font-size: 0.65rem;
        line-height: 1.6;
    }

    </style>
    """
)


def section(title: str) -> None:
    """Render a consistent dashboard section heading."""

    st.html(
        f"""
        <div class="section-title">
            {title}
        </div>
        """
    )


def insight_box(
    label: str,
    content: str,
) -> None:
    """Render a compact engineering insight panel."""

    st.html(
        f"""
        <div class="info-panel">

            <div class="info-label">
                {label}
            </div>

            <div class="info-content">
                {content}
            </div>

        </div>
        """
    )


def reset_dashboard() -> None:
    """
    Clear all session-dependent Streamlit state.

    Driver, tyre and strategy signatures belong to the currently
    loaded FastF1 session, so they must be discarded whenever the
    race or session selection changes.
    """

    keys = [
        "context",
        "comparison",
        "comparison_signature",
        "corner_table",
        "track_analysis",
        "strategy_analysis",
        "strategy_signature",
        "prediction_analysis",
        "tyre_analysis",
        "tyre_signature",
    ]

    for key in keys:
        st.session_state.pop(
            key,
            None,
        )


@st.cache_data(
    show_spinner=False
)
def get_schedule(
    season: int,
) -> pd.DataFrame:
    """
    Retrieve the FastF1 event calendar.
    """

    try:

        schedule = fastf1.get_event_schedule(
            season
        )

        if "EventFormat" in schedule.columns:

            schedule = schedule[
                schedule["EventFormat"]
                != "testing"
            ]

        return schedule

    except Exception:

        return pd.DataFrame()


def schedule_events(
    schedule: pd.DataFrame,
) -> list[str]:
    """Extract usable event names."""

    if (
        schedule.empty
        or "EventName" not in schedule.columns
    ):
        return []

    return (
        schedule["EventName"]
        .dropna()
        .astype(str)
        .tolist()
    )


# Persistent Streamlit state keeps expensive FastF1 work available
# while the user navigates between dashboard sections.
if "context" not in st.session_state:
    st.session_state.context = None

if "comparison" not in st.session_state:
    st.session_state.comparison = None

if "corner_table" not in st.session_state:
    st.session_state.corner_table = None

if "track_analysis" not in st.session_state:
    st.session_state.track_analysis = None

if "strategy_analysis" not in st.session_state:
    st.session_state.strategy_analysis = None

if "prediction_analysis" not in st.session_state:
    st.session_state.prediction_analysis = None

if "tyre_analysis" not in st.session_state:
    st.session_state.tyre_analysis = None


st.html(
    f"""
    <div class="topbar">

        <div class="brand-wrap">

            <div class="f1-badge">
                F1
            </div>

            <div>

                <div class="brand-title">
                    {APP_TITLE.upper()}
                </div>

                <div class="brand-subtitle">
                    {APP_SUBTITLE.upper()}
                </div>

            </div>

        </div>

        <div class="system-status">

            <div class="status-dot"></div>

            ANALYTICS SYSTEM ONLINE

        </div>

    </div>
    """
)


with st.sidebar:

    st.title("PIT WALL")

    st.caption(
        "SESSION CONTROL"
    )

    default_index = (
        SUPPORTED_SEASONS.index(
            DEFAULT_SEASON
        )
        if DEFAULT_SEASON
        in SUPPORTED_SEASONS
        else 0
    )

    season = st.selectbox(
        "Season",
        options=SUPPORTED_SEASONS,
        index=default_index,
        on_change=reset_dashboard,
    )

    schedule = get_schedule(
        season
    )

    events = schedule_events(
        schedule
    )

    if not events:

        events = [
            "Australian Grand Prix",
            "Japanese Grand Prix",
            "Monaco Grand Prix",
            "British Grand Prix",
            "Italian Grand Prix",
            "Singapore Grand Prix",
            "São Paulo Grand Prix",
            "Abu Dhabi Grand Prix",
        ]

    event_name = st.selectbox(
        "Grand Prix",
        options=events,
        on_change=reset_dashboard,
    )

    session_label = st.selectbox(
        "Session",
        options=list(
            SESSION_OPTIONS.keys()
        ),
        index=3,
        on_change=reset_dashboard,
    )

    session_type = (
        SESSION_OPTIONS[
            session_label
        ]
    )

    load_button = st.button(
        "LOAD SESSION",
        type="primary",
        width="stretch",
    )


if load_button:

    with st.spinner(
        f"Loading {season} {event_name} {session_label}..."
    ):

        try:

            context = load_session_context(
                season,
                event_name,
                session_type,
            )

            st.session_state.context = (
                context
            )

            st.session_state.comparison = None
            st.session_state.corner_table = None
            st.session_state.track_analysis = None
            st.session_state.strategy_analysis = None
            st.session_state.prediction_analysis = None
            st.session_state.tyre_analysis = None

        except Exception as error:

            st.error(
                "The selected FastF1 session could not be loaded."
            )

            st.exception(
                error
            )


context = (
    st.session_state.context
)


if context is None:

    st.html(
        """
        <div class="session-card">

            <div class="session-kicker">
                AWAITING SESSION
            </div>

            <div class="session-title">
                Formula 1 Telemetry & Race Intelligence
            </div>

            <div class="session-meta">
                Select a championship season, Grand Prix and session
                from the sidebar, then load the data.
            </div>

        </div>
        """
    )

    card_1, card_2, card_3 = (
        st.columns(3)
    )

    with card_1:

        st.html(
            """
            <div class="landing-card">

                <div class="landing-title">
                    TELEMETRY ENGINEERING
                </div>

                <div class="landing-copy">
                    Compare speed, throttle, braking,
                    gears, RPM and lap delta at synchronized
                    points around the circuit.
                </div>

            </div>
            """
        )

    with card_2:

        st.html(
            """
            <div class="landing-card">

                <div class="landing-title">
                    RACE INTELLIGENCE
                </div>

                <div class="landing-copy">
                    Analyse corners, tyre degradation,
                    stint performance, pit stops and
                    changing circuit conditions.
                </div>

            </div>
            """
        )

    with card_3:

        st.html(
            """
            <div class="landing-card">

                <div class="landing-title">
                    PREDICTIVE ANALYTICS
                </div>

                <div class="landing-copy">
                    Compare regression models and estimate
                    future lap performance from available
                    session features.
                </div>

            </div>
            """
        )

    st.stop()


drivers = get_session_drivers(
    context
)

if len(drivers) < 2:

    st.error(
        "The loaded session does not contain enough drivers "
        "for a comparison."
    )

    st.stop()


with st.sidebar:

    st.divider()

    st.caption(
        "DRIVER CONTROL"
    )

    driver_1 = st.selectbox(
        "Driver 1",
        options=drivers,
        index=0,
    )

    driver_2 = st.selectbox(
        "Driver 2",
        options=drivers,
        index=1,
    )

    st.divider()

    page = st.radio(
        "Analysis Module",
        options=[
            "Race Overview",
            "Driver Comparison",
            "Telemetry",
            "Track Map",
            "Corner Analysis",
            "Tyre Performance",
            "Track Evolution",
            "Strategy Analysis",
            "Predictive Analytics",
            "Session Insights",
            "Data Quality",
        ],
    )


if driver_1 == driver_2:

    st.warning(
        "Select two different drivers."
    )

    st.stop()


# Driver changes are inexpensive compared with loading the whole
# FastF1 session, so rebuild only when the selected pair changes
# or when no valid comparison currently exists.
comparison_signature = (
    driver_1,
    driver_2,
)

comparison_needs_rebuild = (
    st.session_state.get(
        "comparison_signature"
    )
    != comparison_signature
    or st.session_state.get(
        "comparison"
    )
    is None
)

if comparison_needs_rebuild:

    with st.spinner(
        "Preparing driver comparison..."
    ):

        try:

            comparison = (
                build_driver_comparison(
                    context,
                    driver_1,
                    driver_2,
                )
            )

            if not comparison:
                raise ValueError(
                    "The comparison engine returned no data."
                )

            telemetry = comparison.get(
                "telemetry"
            )

            if not telemetry:
                raise ValueError(
                    "No usable telemetry comparison was returned."
                )

            required_keys = [
                "telemetry_a",
                "telemetry_b",
                "aligned",
                "metrics_a",
                "metrics_b",
            ]

            missing_keys = [
                key
                for key in required_keys
                if key not in telemetry
            ]

            if missing_keys:
                raise ValueError(
                    "Telemetry comparison is incomplete. "
                    f"Missing: {', '.join(missing_keys)}"
                )

            st.session_state.comparison = (
                comparison
            )

            st.session_state.corner_table = None
            st.session_state.strategy_analysis = None
            st.session_state.tyre_analysis = None

            st.session_state.comparison_signature = (
                comparison_signature
            )

        except Exception as error:

            st.session_state.comparison = None

            st.session_state.pop(
                "comparison_signature",
                None,
            )

            st.error(
                f"{driver_1} and {driver_2} could not be compared."
            )

            st.info(
                "One of the selected drivers may not have a usable "
                "timed lap or complete telemetry in this session."
            )

            st.exception(
                error
            )

            st.stop()


comparison = (
    st.session_state.comparison
)

if comparison is None:

    st.error(
        "No driver comparison data is currently available."
    )

    st.stop()


telemetry_result = (
    comparison[
        "telemetry"
    ]
)

summary_a = (
    comparison[
        "summary_a"
    ]
)

summary_b = (
    comparison[
        "summary_b"
    ]
)

metrics_a = (
    telemetry_result[
        "metrics_a"
    ]
)

metrics_b = (
    telemetry_result[
        "metrics_b"
    ]
)

aligned = (
    telemetry_result[
        "aligned"
    ]
)


st.html(
    f"""
    <div class="session-card">

        <div class="session-kicker">
            ACTIVE SESSION
        </div>

        <div class="session-title">
            {context.season} {context.event_name}
        </div>

        <div class="session-meta">
            {context.circuit} • {context.country}
            • {context.session.name}
            • {driver_1} vs {driver_2}
        </div>

    </div>
    """
)


weather = get_weather_summary(
    context.session
)


if page == "Race Overview":

    section(
        "SESSION OVERVIEW"
    )

    lap_a = summary_a[
        "lap_time_seconds"
    ]

    lap_b = summary_b[
        "lap_time_seconds"
    ]

    if (
        lap_a is not None
        and lap_b is not None
    ):

        gap = abs(
            lap_a - lap_b
        )

        advantage = (
            driver_1
            if lap_a < lap_b
            else driver_2
        )

    else:

        gap = None
        advantage = "N/A"


    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            f"{driver_1} FASTEST",
            format_lap_time(
                lap_a
            ),
        )

    with c2:

        st.metric(
            f"{driver_2} FASTEST",
            format_lap_time(
                lap_b
            ),
        )

    with c3:

        st.metric(
            "LAP GAP",
            format_seconds(
                gap
            ),
        )

    with c4:

        st.metric(
            "PACE ADVANTAGE",
            advantage,
        )


    w1, w2, w3, w4 = (
        st.columns(4)
    )

    with w1:

        st.metric(
            "TRACK TEMP",
            format_temperature(
                weather[
                    "track_temperature"
                ]
            ),
        )

    with w2:

        st.metric(
            "AIR TEMP",
            format_temperature(
                weather[
                    "air_temperature"
                ]
            ),
        )

    with w3:

        st.metric(
            "HUMIDITY",
            format_percentage(
                weather[
                    "humidity"
                ],
                decimals=0,
            ),
        )

    with w4:

        st.metric(
            "CONDITIONS",
            weather[
                "condition"
            ],
        )


    section(
        "WEATHER CONTEXT"
    )

    insight_box(
        "SESSION CONDITIONS",
        build_weather_insight(
            context.session
        ),
    )


elif page == "Driver Comparison":

    section(
        "DRIVER PERFORMANCE"
    )

    comparison_table = pd.DataFrame(
        {
            "Metric": [
                "Fastest Lap",
                "Lap Number",
                "Tyre Compound",
                "Maximum Speed",
                "Average Speed",
                "Full Throttle",
                "Braking Events",
                "Average RPM",
            ],

            driver_1: [
                format_lap_time(
                    summary_a.get(
                        "lap_time_seconds"
                    )
                ),
                summary_a.get(
                    "lap_number"
                ),
                summary_a.get(
                    "compound"
                ),
                format_speed(
                    metrics_a.get(
                        "maximum_speed"
                    )
                ),
                format_speed(
                    metrics_a.get(
                        "average_speed"
                    )
                ),
                format_percentage(
                    metrics_a.get(
                        "full_throttle_percentage"
                    )
                ),
                metrics_a.get(
                    "braking_events"
                ),
                (
                    f"{metrics_a['average_rpm']:.0f}"
                    if metrics_a.get(
                        "average_rpm"
                    ) is not None
                    else "N/A"
                ),
            ],

            driver_2: [
                format_lap_time(
                    summary_b.get(
                        "lap_time_seconds"
                    )
                ),
                summary_b.get(
                    "lap_number"
                ),
                summary_b.get(
                    "compound"
                ),
                format_speed(
                    metrics_b.get(
                        "maximum_speed"
                    )
                ),
                format_speed(
                    metrics_b.get(
                        "average_speed"
                    )
                ),
                format_percentage(
                    metrics_b.get(
                        "full_throttle_percentage"
                    )
                ),
                metrics_b.get(
                    "braking_events"
                ),
                (
                    f"{metrics_b['average_rpm']:.0f}"
                    if metrics_b.get(
                        "average_rpm"
                    ) is not None
                    else "N/A"
                ),
            ],
        }
    )

    st.dataframe(
        comparison_table,
        width="stretch",
        hide_index=True,
    )


elif page == "Telemetry":

    section(
        "SYNCHRONIZED TELEMETRY"
    )

    t1, t2, t3, t4 = (
        st.columns(4)
    )

    with t1:

        st.metric(
            f"{driver_1} TOP SPEED",
            format_speed(
                metrics_a.get(
                    "maximum_speed"
                )
            ),
        )

    with t2:

        st.metric(
            f"{driver_2} TOP SPEED",
            format_speed(
                metrics_b.get(
                    "maximum_speed"
                )
            ),
        )

    with t3:

        st.metric(
            f"{driver_1} FULL THROTTLE",
            format_percentage(
                metrics_a.get(
                    "full_throttle_percentage"
                )
            ),
        )

    with t4:

        st.metric(
            f"{driver_2} FULL THROTTLE",
            format_percentage(
                metrics_b.get(
                    "full_throttle_percentage"
                )
            ),
        )


    charts = [
        build_speed_chart(
            aligned,
            driver_1,
            driver_2,
        ),
        build_throttle_chart(
            aligned,
            driver_1,
            driver_2,
        ),
        build_brake_chart(
            aligned,
            driver_1,
            driver_2,
        ),
        build_gear_chart(
            aligned,
            driver_1,
            driver_2,
        ),
        build_rpm_chart(
            aligned,
            driver_1,
            driver_2,
        ),
        build_delta_chart(
            aligned,
            driver_1,
            driver_2,
        ),
    ]

    for chart in charts:

        st.plotly_chart(
            chart,
            width="stretch",
        )


elif page == "Track Map":

    section(
        "CIRCUIT PERFORMANCE MAP"
    )

    control_1, control_2 = (
        st.columns(2)
    )

    with control_1:

        map_driver = st.radio(
            "Driver",
            options=[
                driver_1,
                driver_2,
            ],
            horizontal=True,
        )

    with control_2:

        map_metric = st.selectbox(
            "Colour circuit by",
            options=[
                "Speed",
                "Throttle",
                "Gear",
                "RPM",
            ],
        )


    if map_driver == driver_1:

        map_telemetry = (
            telemetry_result[
                "telemetry_a"
            ]
        )

    else:

        map_telemetry = (
            telemetry_result[
                "telemetry_b"
            ]
        )


    try:

        map_chart = build_track_map(
            map_telemetry,
            map_driver,
            metric=map_metric,
        )

        st.plotly_chart(
            map_chart,
            width="stretch",
        )

    except Exception as error:

        st.warning(
            "Circuit position data could not be rendered."
        )

        st.exception(
            error
        )


elif page == "Corner Analysis":

    section(
        "CORNER-BY-CORNER ANALYSIS"
    )

    if st.session_state.corner_table is None:

        with st.spinner(
            "Detecting and analysing corner regions..."
        ):

            try:

                st.session_state.corner_table = (
                    build_corner_analysis(
                        comparison,
                        driver_1,
                        driver_2,
                    )
                )

            except Exception as error:

                st.warning(
                    "Corner analysis could not be calculated."
                )

                st.exception(
                    error
                )


    corner_table = (
        st.session_state.corner_table
    )

    if (
        corner_table is None
        or corner_table.empty
    ):

        st.info(
            "No reliable analytical corner regions were detected."
        )

    else:

        display = round_dataframe(
            corner_table,
            decimals=3,
        )

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

        selected_corner = st.selectbox(
            "Inspect corner region",
            options=(
                corner_table[
                    "Corner"
                ]
                .astype(int)
                .tolist()
            ),
        )

        selected = corner_table[
            corner_table[
                "Corner"
            ]
            == selected_corner
        ].iloc[0]

        corner_insight = (
            generate_corner_insight(
                selected,
                driver_1,
                driver_2,
            )
        )

        insight_box(
            "RACE ENGINEER CORNER INSIGHT",
            corner_insight,
        )

        st.caption(
            "Corner regions are telemetry-derived analytical "
            "windows and are not yet guaranteed to match official "
            "FIA turn numbering."
        )


elif page == "Tyre Performance":

    section(
        "TYRE DEGRADATION"
    )

    tyre_driver = st.selectbox(
        "Driver",
        options=[
            driver_1,
            driver_2,
        ],
    )

    try:

        driver_laps = prepare_driver_laps(
            context.session,
            tyre_driver,
        )

        stints = available_stints(
            driver_laps
        )

        if not stints:

            st.info(
                "No usable stint information exists "
                "for this driver/session."
            )

        else:

            stint_number = st.selectbox(
                "Stint",
                options=stints,
            )

            tyre_key = (
                tyre_driver,
                stint_number,
            )

            if (
                st.session_state.get(
                    "tyre_signature"
                )
                != tyre_key
            ):

                st.session_state.tyre_analysis = (
                    build_tyre_analysis(
                        context,
                        tyre_driver,
                        stint_number,
                    )
                )

                st.session_state.tyre_signature = (
                    tyre_key
                )

                save_tyre_stint_result(
                    context.session_key,
                    tyre_driver,
                    stint_number,
                    st.session_state.tyre_analysis,
                )


            tyre_analysis = (
                st.session_state.tyre_analysis
            )

            tyre_summary = (
                tyre_analysis[
                    "summary"
                ]
            )

            tyre_model = (
                tyre_analysis[
                    "model"
                ]
            )


            c1, c2, c3, c4 = (
                st.columns(4)
            )

            with c1:

                st.metric(
                    "COMPOUND",
                    tyre_summary.get(
                        "compound"
                    ),
                )

            with c2:

                st.metric(
                    "CLEAN LAPS",
                    tyre_summary.get(
                        "clean_laps"
                    ),
                )

            with c3:

                st.metric(
                    "BEST LAP",
                    format_lap_time(
                        tyre_summary.get(
                            "best_lap"
                        )
                    ),
                )

            with c4:

                degradation = (
                    tyre_model.get(
                        "seconds_per_lap"
                    )
                    if tyre_model
                    else None
                )

                st.metric(
                    "DEGRADATION",
                    format_degradation(
                        degradation
                    ),
                )


            st.plotly_chart(
                build_tyre_degradation_chart(
                    tyre_analysis
                ),
                width="stretch",
            )


            if tyre_model:

                m1, m2 = (
                    st.columns(2)
                )

                with m1:

                    st.metric(
                        "MODEL MAE",
                        format_seconds(
                            tyre_model.get(
                                "mae"
                            )
                        ),
                    )

                with m2:

                    st.metric(
                        "MODEL RMSE",
                        format_seconds(
                            tyre_model.get(
                                "rmse"
                            )
                        ),
                    )

            else:

                st.info(
                    "The stint does not contain enough clean laps "
                    "for degradation modelling."
                )

    except Exception as error:

        st.warning(
            "Tyre analysis could not be generated."
        )

        st.exception(
            error
        )


elif page == "Track Evolution":

    section(
        "TRACK EVOLUTION"
    )

    if st.session_state.track_analysis is None:

        with st.spinner(
            "Analysing session-wide pace evolution..."
        ):

            try:

                st.session_state.track_analysis = (
                    build_track_analysis(
                        context
                    )
                )

            except Exception as error:

                st.warning(
                    "Track-evolution analysis could not be generated."
                )

                st.exception(
                    error
                )


    track_analysis = (
        st.session_state.track_analysis
    )

    if track_analysis:

        change = (
            track_analysis[
                "trend"
            ][
                "seconds_per_10_minutes"
            ]
        )

        correlation = (
            track_analysis[
                "correlations"
            ][
                "track_temperature"
            ]
        )


        e1, e2, e3 = (
            st.columns(3)
        )

        with e1:

            st.metric(
                "PACE / 10 MIN",
                format_seconds(
                    change,
                    signed=True,
                ),
            )

        with e2:

            st.metric(
                "TRACK TEMP CORRELATION",
                (
                    f"{correlation['correlation']:+.2f}"
                    if correlation
                    else "N/A"
                ),
            )

        with e3:

            st.metric(
                "CLEAN LAPS",
                len(
                    track_analysis[
                        "data"
                    ]
                ),
            )


        st.plotly_chart(
            build_session_pace_chart(
                track_analysis
            ),
            width="stretch",
        )


        data = track_analysis[
            "data"
        ]

        if (
            "TrackTemp"
            in data.columns
            and data[
                "TrackTemp"
            ].notna().any()
        ):

            st.plotly_chart(
                build_temperature_pace_chart(
                    track_analysis
                ),
                width="stretch",
            )


        insight_box(
            "TRACK EVOLUTION INSIGHT",
            generate_track_evolution_insight(
                track_analysis
            ),
        )

        st.caption(
            "Correlation does not establish causation. "
            "Fuel load, tyres, traffic and run plans can all "
            "contribute to session-wide lap-time trends."
        )


elif page == "Strategy Analysis":

    section(
        "RACE STRATEGY"
    )

    if session_type != "R":

        st.info(
            "Strategy analysis is most meaningful for a Race session."
        )


    if st.session_state.strategy_analysis is None:

        try:

            st.session_state.strategy_analysis = (
                build_strategy_analysis(
                    context,
                    driver_1,
                    driver_2,
                )
            )

        except Exception as error:

            st.warning(
                "Strategy analysis could not be generated."
            )

            st.exception(
                error
            )


    strategy = (
        st.session_state.strategy_analysis
    )

    if strategy:

        st.plotly_chart(
            build_stint_timeline(
                strategy[
                    "stints_a"
                ],
                strategy[
                    "stints_b"
                ],
                driver_1,
                driver_2,
            ),
            width="stretch",
        )

        st.plotly_chart(
            build_strategy_pace_chart(
                strategy[
                    "laps_a"
                ],
                strategy[
                    "laps_b"
                ],
                driver_1,
                driver_2,
            ),
            width="stretch",
        )


        strategy_col_a, strategy_col_b = (
            st.columns(2)
        )

        with strategy_col_a:

            st.caption(
                f"{driver_1} STINTS"
            )

            st.dataframe(
                round_dataframe(
                    strategy[
                        "stints_a"
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        with strategy_col_b:

            st.caption(
                f"{driver_2} STINTS"
            )

            st.dataframe(
                round_dataframe(
                    strategy[
                        "stints_b"
                    ]
                ),
                width="stretch",
                hide_index=True,
            )


elif page == "Predictive Analytics":

    section(
        "LAP-TIME PREDICTION"
    )

    st.caption(
        "Models use a chronological hold-out split so future "
        "session laps are not used to predict earlier laps."
    )


    train_button = st.button(
        "TRAIN & EVALUATE MODELS",
        type="primary",
    )


    if train_button:

        with st.spinner(
            "Training Linear Regression, Ridge and Random Forest..."
        ):

            try:

                st.session_state.prediction_analysis = (
                    build_prediction_analysis(
                        context
                    )
                )

            except Exception as error:

                st.warning(
                    "Predictive models could not be trained."
                )

                st.exception(
                    error
                )


    prediction_analysis = (
        st.session_state.prediction_analysis
    )


    if prediction_analysis:

        training_result = (
            prediction_analysis[
                "training_result"
            ]
        )

        comparison_table = (
            prediction_analysis[
                "comparison_table"
            ]
        )

        best_model = (
            training_result[
                "best_model"
            ]
        )


        p1, p2, p3 = (
            st.columns(3)
        )

        with p1:

            st.metric(
                "BEST MODEL",
                best_model.name,
            )

        with p2:

            st.metric(
                "TEST MAE",
                format_seconds(
                    best_model.mae
                ),
            )

        with p3:

            st.metric(
                "TEST RMSE",
                format_seconds(
                    best_model.rmse
                ),
            )


        st.dataframe(
            round_dataframe(
                comparison_table,
                decimals=4,
            ),
            width="stretch",
            hide_index=True,
        )


        st.plotly_chart(
            build_model_comparison_chart(
                comparison_table
            ),
            width="stretch",
        )

        st.plotly_chart(
            build_actual_vs_predicted_chart(
                training_result
            ),
            width="stretch",
        )


        section(
            "NEXT-LAP ESTIMATE"
        )

        input_1, input_2, input_3 = (
            st.columns(3)
        )

        with input_1:

            prediction_driver = st.selectbox(
                "Prediction driver",
                options=drivers,
            )

        with input_2:

            prediction_compound = st.selectbox(
                "Compound",
                options=[
                    "SOFT",
                    "MEDIUM",
                    "HARD",
                    "INTERMEDIATE",
                    "WET",
                ],
            )

        with input_3:

            prediction_tyre_life = st.number_input(
                "Tyre age",
                min_value=1.0,
                value=5.0,
                step=1.0,
            )


        input_4, input_5, input_6 = (
            st.columns(3)
        )

        test_data = (
            training_result[
                "test"
            ]
        )


        with input_4:

            prediction_stint = st.number_input(
                "Stint",
                min_value=1.0,
                value=1.0,
                step=1.0,
            )

        with input_5:

            default_lap = float(
                test_data[
                    "LapNumber"
                ].max()
                + 1
            )

            prediction_lap = st.number_input(
                "Lap number",
                min_value=1.0,
                value=default_lap,
                step=1.0,
            )

        with input_6:

            default_session_time = float(
                test_data[
                    "SessionTimeSeconds"
                ].max()
                + 90
            )

            prediction_time = st.number_input(
                "Session time (seconds)",
                min_value=0.0,
                value=default_session_time,
                step=60.0,
            )


        if st.button(
            "PREDICT LAP TIME"
        ):

            prediction = predict_next_lap(
                training_result,
                driver=prediction_driver,
                compound=prediction_compound,
                tyre_life=prediction_tyre_life,
                stint=prediction_stint,
                lap_number=prediction_lap,
                session_time_seconds=prediction_time,
            )

            st.metric(
                "PREDICTED LAP",
                format_lap_time(
                    prediction
                ),
            )

            st.caption(
                "This is a statistical estimate from the trained "
                "session model, not a guaranteed future lap time."
            )


elif page == "Session Insights":

    section(
        "AUTOMATED RACE ENGINEER"
    )


    if st.session_state.corner_table is None:

        try:

            st.session_state.corner_table = (
                build_corner_analysis(
                    comparison,
                    driver_1,
                    driver_2,
                )
            )

        except Exception:

            pass


    if st.session_state.track_analysis is None:

        try:

            st.session_state.track_analysis = (
                build_track_analysis(
                    context
                )
            )

        except Exception:

            pass


    report = build_race_engineer_report(
        driver_a=driver_1,
        driver_b=driver_2,
        summary_a=summary_a,
        summary_b=summary_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        corner_table=(
            st.session_state.corner_table
        ),
        tyre_analysis=(
            st.session_state.tyre_analysis
        ),
        track_analysis=(
            st.session_state.track_analysis
        ),
        strategy_analysis=(
            st.session_state.strategy_analysis
        ),
        training_result=(
            st.session_state.prediction_analysis[
                "training_result"
            ]
            if st.session_state.prediction_analysis
            else None
        ),
    )


    all_insights = flatten_report(
        report
    )


    if not all_insights:

        st.info(
            "No analytical insights are currently available."
        )

    else:

        for number, text in enumerate(
            all_insights,
            start=1,
        ):

            insight_box(
                f"ENGINEER NOTE {number:02d}",
                text,
            )


        if st.button(
            "SAVE INSIGHTS TO SQLITE"
        ):

            save_insight_report(
                context.session_key,
                "race_engineer",
                all_insights,
            )

            st.success(
                "Insight report saved to the local SQLite database."
            )


elif page == "Data Quality":

    section(
        "DATA QUALITY & CLEANING"
    )

    raw_laps = (
        context.session.laps.copy()
    )

    flagged = add_lap_quality_flags(
        raw_laps
    )

    flagged = flag_lap_time_outliers(
        flagged
    )

    flagged = attach_quality_reason(
        flagged
    )

    quality = data_quality_summary(
        flagged
    )


    q1, q2, q3 = (
        st.columns(3)
    )

    with q1:

        st.metric(
            "TOTAL LAPS",
            len(
                raw_laps
            ),
        )

    with q2:

        clean_count = int(
            (
                flagged[
                    "QualityReason"
                ]
                == "clean"
            ).sum()
        )

        st.metric(
            "UNFLAGGED LAPS",
            clean_count,
        )

    with q3:

        st.metric(
            "FLAGGED LAPS",
            len(flagged)
            - clean_count,
        )


    st.dataframe(
        round_dataframe(
            quality,
            decimals=2,
        ),
        width="stretch",
        hide_index=True,
    )


    with st.expander(
        "Inspect quality-flagged laps"
    ):

        columns = [
            column
            for column in [
                "Driver",
                "LapNumber",
                "LapTimeSeconds",
                "Compound",
                "Stint",
                "QualityReason",
            ]
            if column in flagged.columns
        ]

        st.dataframe(
            flagged[
                columns
            ],
            width="stretch",
            hide_index=True,
        )


    insight_box(
        "CLEANING PRINCIPLE",
        (
            "The system preserves explicit quality flags before "
            "excluding laps. This keeps analytical exclusions "
            "traceable instead of silently deleting observations."
        ),
    )


st.html(
    f"""
    <div class="footer">

        {APP_TITLE.upper()} • {APP_SUBTITLE.upper()}

        <br><br>

        {DISCLAIMER}

    </div>
    """
)