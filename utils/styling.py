from __future__ import annotations

import pandas as pd


def coerce_numeric(value):
    """Extract a numeric value from *value* when possible.

    Strings like "123 (4)" are coerced to 123.0, which allows coverage
    comparisons against formatted values without losing the textual
    representation in the UI.
    """

    if isinstance(value, str):
        try:
            numeric = value.strip().split(" ")[0].replace(",", "")
            return float(numeric)
        except (ValueError, IndexError):
            return None
    return value


def coverage_color(required, value):
    """Return a ``(background, text)`` tuple for coverage deltas.

    ``None`` is returned when the values cannot be compared.
    """

    value = coerce_numeric(value)
    required = coerce_numeric(required)

    if pd.isna(required) or pd.isna(value):
        return None

    if required == 0:
        if value == 0:
            return None
        return "#dbeafe", "#1e3a8a"

    gap_ratio = (value - required) / required
    if gap_ratio <= -0.15:
        return "#fee2e2", "#7f1d1d"
    if gap_ratio < 0:
        return "#ffe4e6", "#7f1d1d"
    if gap_ratio < 0.1:
        return "#fef9c3", "#854d0e"
    if gap_ratio < 0.25:
        return "#dcfce7", "#166534"
    return "#bbf7d0", "#166534"


def coverage_style(required, value, *, emphasize: bool = True) -> str:
    """Return an inline CSS declaration for coverage comparison."""

    colors = coverage_color(required, value)
    if not colors:
        return ""

    background, text = colors
    weight = " font-weight: 600" if emphasize else ""
    return f"background-color: {background}; color: {text};{weight}"
