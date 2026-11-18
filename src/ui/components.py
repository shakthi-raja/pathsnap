"""Presentational helpers for the PathSnap interface.

Everything here returns markup or calls Streamlit directly; none of it touches
HTTP or the dataset, which keeps the page itself readable and the pieces
testable.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

import streamlit as st

__all__ = [
    "empty_state",
    "inject_theme",
    "masthead",
    "note",
    "readout_strip",
    "sentence_label",
    "warning_lines",
]

_CSS_PATH = Path(__file__).parent / "assets" / "theme.css"

#: Metrics the backend reports as fractions.
_FRACTION_METRICS = frozenset(
    {
        "pointwise_accuracy",
        "route_continuity",
        "classifier_continuity",
        "top1_accuracy",
        "candidate_miss_rate",
    }
)

#: Metrics that read better with a fixed name than the generic transformation.
_LABEL_OVERRIDES = {
    "top1_accuracy": "Top-1 accuracy",
    "classifier_continuity": "Classifier continuity",
}

#: Metrics that are counts rather than scores.
_COUNT_METRICS = frozenset({"evaluated_observations"})


def inject_theme() -> None:
    """Load the stylesheet once per rerun."""
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sentence_label(name: str) -> str:
    """Turn a metric key into a sentence-case label.

    >>> sentence_label("pointwise_accuracy")
    'Pointwise accuracy'
    """
    if name in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[name]
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _format_value(name: str, value: float) -> str:
    if name in _FRACTION_METRICS:
        return f"{value:.1%}"
    if name in _COUNT_METRICS:
        return f"{int(value):,}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def masthead(*, connected: bool, detail: str) -> None:
    """Page header: wordmark, one line of context, connection state."""
    state = "Backend ready" if connected else "No backend"
    flag = "yes" if connected else "no"
    st.markdown(
        f"""
        <div class="ps-mast">
          <div>
            <div class="ps-word">PathSnap</div>
            <p class="ps-sub">Rebuilds the road a vehicle took from noisy GPS.</p>
          </div>
          <div class="ps-state" title="{html.escape(detail, quote=True)}">
            <span class="ps-pip" data-on="{flag}"></span>{state}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def readout_strip(metrics: Mapping[str, float | None], *, runtime_ms: float | None) -> None:
    """Show the backend's numbers as a single instrument strip.

    A metric the backend did not measure reads "not reported". Nothing is
    substituted here, so an unmeasured score can never look measured.
    """
    cells: list[tuple[str, str, bool]] = []
    for name, value in metrics.items():
        label = sentence_label(name)
        if value is None:
            cells.append((label, "not reported", True))
        else:
            cells.append((label, _format_value(name, value), False))
    if runtime_ms is not None:
        cells.append(("Runtime", f"{runtime_ms:,.0f} ms", False))
    if not cells:
        return

    body = "".join(
        f'<div class="ps-cell" data-empty="{"yes" if empty else "no"}">'
        f"<span>{html.escape(label)}</span>"
        f"<strong>{html.escape(value)}</strong></div>"
        for label, value, empty in cells
    )
    st.markdown(f'<div class="ps-strip">{body}</div>', unsafe_allow_html=True)


def note(text: str, *, tone: str = "plain") -> None:
    """A short message in the page's voice. Tones: plain, quiet, alert."""
    st.markdown(
        f'<div class="ps-note" data-tone="{tone}">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str) -> None:
    """What the map area says before there is anything to show."""
    st.markdown(
        f'<div class="ps-empty"><strong>{html.escape(title)}</strong>{html.escape(body)}</div>',
        unsafe_allow_html=True,
    )


def warning_lines(messages: Sequence[str]) -> None:
    """Render backend warnings one per line, unedited."""
    for message in messages:
        note(message, tone="caution")
