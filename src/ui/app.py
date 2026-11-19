"""Evaluator interface for PathSnap.

Run with::

    python -m streamlit run src/ui/app.py

One screen, one job: discover the trips the backend can serve, run one, show
the route and the numbers that came back. Reference data stays on the server,
so nothing here can leak a label.
"""

from __future__ import annotations

import html
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when invoked via standalone streamlit CLI
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components

from src.config import ConfigurationError, Settings, get_settings
from src.ui.api_client import ApiError, PathSnapClient, Prediction
from src.ui.components import (
    empty_state,
    inject_theme,
    masthead,
    note,
    readout_strip,
    warning_lines,
)
from src.visualization.maps import build_trip_map, normalise_payload

logger = logging.getLogger(__name__)

_MAP_HEIGHT_PX = 640

#: Streamlit reruns the page on every interaction, including the one the map
#: component fires after it renders. The last prediction lives in session state
#: or it would vanish the moment it appeared.
_RESULT_KEY = "pathsnap_prediction"


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_trips(settings: Settings) -> tuple[str, ...]:
    """Discover servable trips, cached briefly so reruns stay quick."""
    with PathSnapClient(settings) as client:
        return client.list_trips()


def _check_health(settings: Settings) -> bool:
    try:
        with PathSnapClient(settings) as client:
            client.health()
    except ApiError as exc:
        logger.warning("Health check failed: %s", exc)
        return False
    return True


def _run_prediction(settings: Settings, trip_id: str) -> Prediction:
    with PathSnapClient(settings) as client:
        return client.predict(trip_id)


def _rail(connected: bool, trip_ids: tuple[str, ...]) -> tuple[str | None, bool]:
    """The side panel: choose a trip, run it, reconnect if the link is down."""
    selected: str | None = None
    run_clicked = False

    with st.sidebar:
        st.markdown('<span class="ps-label">Trip</span>', unsafe_allow_html=True)

        if not connected:
            note("The trip list appears once the backend answers.")
        elif not trip_ids:
            note("The backend reports no trips.")
        else:
            selected = st.selectbox("Trip", trip_ids, label_visibility="collapsed")
            run_clicked = st.button("Reconstruct route", type="primary")

        st.markdown('<div class="ps-rule"></div>', unsafe_allow_html=True)
        if st.button("Reconnect", type="secondary"):
            st.cache_data.clear()
            st.rerun()

    return selected, run_clicked


def _render_map(prediction: Prediction) -> None:
    try:
        payload = normalise_payload(
            {
                "track_points": prediction.track_points,
                "road_context": prediction.road_context,
                "predicted_polylines": prediction.predicted_polylines,
            }
        )
    except (KeyError, ValueError) as exc:
        note(f"The backend sent geometry that cannot be drawn: {exc}", tone="alert")
        return

    if not payload.track_points:
        note("The backend returned no GPS points for this trip.", tone="alert")
        return

    # The map is rendered as standalone HTML rather than through the component
    # wrapper: the wrapper sized the iframe to zero height on this Streamlit
    # version, leaving a blank panel.
    trip_map = build_trip_map(payload)
    components.html(trip_map.get_root().render(), height=_MAP_HEIGHT_PX, scrolling=False)


def main() -> None:
    st.set_page_config(page_title="PathSnap", layout="wide", initial_sidebar_state="expanded")
    inject_theme()

    try:
        settings = get_settings()
    except ConfigurationError as exc:
        masthead(connected=False, detail="configuration")
        note(f"Configuration problem: {exc}", tone="alert")
        return

    connected = _check_health(settings)

    trip_ids: tuple[str, ...] = ()
    load_error: str | None = None
    if connected:
        try:
            trip_ids = _fetch_trips(settings)
        except ApiError as exc:
            load_error = str(exc)

    selected, run_clicked = _rail(connected, trip_ids)
    masthead(connected=connected, detail=settings.api_url)

    if not connected:
        note(
            f"Nothing answered at {settings.api_url}. Start the API on Laptop 1, "
            "then set PATHSNAP_API_URL to that machine's LAN address and reload.",
            tone="alert",
        )
        empty_state("Waiting for the backend", "The map appears once a prediction comes back.")
        return

    if load_error:
        note(load_error, tone="alert")
        return

    if not trip_ids:
        note(
            "The backend is up but reports no trips. Check that its data root "
            "contains the train and student_test folders.",
            tone="alert",
        )
        return

    if run_clicked and selected:
        try:
            with st.spinner("Reconstructing the route"):
                st.session_state[_RESULT_KEY] = _run_prediction(settings, selected)
        except ApiError as exc:
            st.session_state.pop(_RESULT_KEY, None)
            note(str(exc), tone="alert")
            return

    prediction = st.session_state.get(_RESULT_KEY)
    if not isinstance(prediction, Prediction):
        empty_state(
            "No trip loaded",
            "Pick a trip in the panel and run it. The route and the readout appear here.",
        )
        return

    st.markdown(
        f'<span class="ps-label">Trip {html.escape(prediction.trip_id)}</span>',
        unsafe_allow_html=True,
    )
    readout_strip(prediction.metrics, runtime_ms=prediction.runtime_ms)
    warning_lines(prediction.warnings)
    _render_map(prediction)


if __name__ == "__main__":
    main()
