"""Small deterministic SVG helpers for the Lean-evaluation reports.

The analysis package deliberately does not depend on a plotting library.  These
helpers produce a conservative SVG subset with fixed dimensions and stable
attribute ordering.  The input is treated as data, so labels are escaped and
the resulting document contains no timestamps, random IDs, or renderer
metadata.
"""

from __future__ import annotations

from html import escape
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_WIDTH = 760
DEFAULT_HEIGHT = 420
_FONT = "system-ui, -apple-system, sans-serif"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: Any) -> str:
    return escape(str(value), quote=True)


def _fmt(value: float) -> str:
    """Format a finite SVG number without locale or exponent drift."""

    if not math.isfinite(value):
        value = 0.0
    if abs(value) < 0.0000005:
        value = 0.0
    result = f"{value:.6f}".rstrip("0").rstrip(".")
    return result or "0"


def _document(
    body: str,
    *,
    title: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    description: str = "",
) -> str:
    if width <= 0 or height <= 0:
        raise ValueError("SVG dimensions must be positive")
    title_node = f"<title>{_text(title)}</title>"
    desc_node = f"<desc>{_text(description)}</desc>" if description else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg height="{int(height)}" role="img" viewBox="0 0 {int(width)} {int(height)}" '
        f'width="{int(width)}" xmlns="http://www.w3.org/2000/svg">'
        f"{title_node}{desc_node}{body}</svg>\n"
    )


def _header(title: str, *, width: int, margin: int, y: int = 30) -> str:
    return (
        f'<text fill="#172033" font-family="{_FONT}" font-size="18" font-weight="600" '
        f'text-anchor="start" x="{margin}" y="{y}">{_text(title)}</text>'
    )


def _axis(*, left: int, top: int, right: int, bottom: int, width: int, height: int) -> str:
    return (
        f'<line stroke="#a8b1c1" stroke-width="1" x1="{left}" x2="{left}" y1="{top}" y2="{bottom}"/>'
        f'<line stroke="#a8b1c1" stroke-width="1" x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}"/>'
    )


def _grid(*, left: int, top: int, right: int, bottom: int, ticks: int = 5) -> str:
    parts: list[str] = []
    for index in range(ticks + 1):
        y = bottom - (bottom - top) * index / ticks
        parts.append(
            f'<line stroke="#e3e7ee" stroke-width="1" x1="{left}" x2="{right}" '
            f'y1="{_fmt(y)}" y2="{_fmt(y)}"/>'
        )
    return "".join(parts)


def render_rates(
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str = "Arm rates",
    label_key: str = "label",
    value_key: str = "rate",
    lower_key: str = "lower",
    upper_key: str = "upper",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> str:
    """Render a deterministic rate bar chart.

    Rows are sorted by their label.  Values are clipped to the displayed
    ``[0, 1]`` scale while the original data remain untouched by the caller.
    Optional lower and upper values draw a thin interval whisker.
    """

    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row.get(label_key, "")))
    margin_left, margin_right, top, bottom = 70, 24, 60, height - 66
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, bottom - top)
    count = max(1, len(ordered))
    band = plot_width / count
    bar_width = min(72.0, band * 0.58)
    body: list[str] = [_header(title, width=width, margin=margin_left), _grid(left=margin_left, top=top, right=width-margin_right, bottom=bottom), _axis(left=margin_left, top=top, right=width-margin_right, bottom=bottom, width=width, height=height)]
    for tick in range(6):
        value = tick / 5
        y = bottom - value * plot_height
        body.append(
            f'<text fill="#4c5668" font-family="{_FONT}" font-size="11" text-anchor="end" '
            f'x="{margin_left - 8}" y="{_fmt(y + 4)}">{_fmt(value)}</text>'
        )
    for index, row in enumerate(ordered):
        value = min(1.0, max(0.0, _number(row.get(value_key))))
        x_center = margin_left + band * (index + 0.5)
        x = x_center - bar_width / 2
        y = bottom - value * plot_height
        body.append(
            f'<rect fill="#4472c4" height="{_fmt(bottom - y)}" rx="2" width="{_fmt(bar_width)}" '
            f'x="{_fmt(x)}" y="{_fmt(y)}"/>'
        )
        lower = row.get(lower_key)
        upper = row.get(upper_key)
        if lower is not None and upper is not None:
            low = min(1.0, max(0.0, _number(lower)))
            high = min(1.0, max(0.0, _number(upper)))
            if high < low:
                low, high = high, low
            y_low = bottom - low * plot_height
            y_high = bottom - high * plot_height
            body.append(
                f'<line stroke="#172033" stroke-width="2" x1="{_fmt(x_center)}" x2="{_fmt(x_center)}" '
                f'y1="{_fmt(y_low)}" y2="{_fmt(y_high)}"/>'
                f'<line stroke="#172033" stroke-width="2" x1="{_fmt(x_center - 5)}" x2="{_fmt(x_center + 5)}" '
                f'y1="{_fmt(y_low)}" y2="{_fmt(y_low)}"/>'
                f'<line stroke="#172033" stroke-width="2" x1="{_fmt(x_center - 5)}" x2="{_fmt(x_center + 5)}" '
                f'y1="{_fmt(y_high)}" y2="{_fmt(y_high)}"/>'
            )
        label = _text(row.get(label_key, ""))
        body.append(
            f'<text fill="#4c5668" font-family="{_FONT}" font-size="11" text-anchor="middle" '
            f'x="{_fmt(x_center)}" y="{height - 34}">{label}</text>'
        )
    return _document("".join(body), title=title, width=width, height=height)


def render_contrast(
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str = "Paired contrast",
    label_key: str = "label",
    value_key: str = "difference",
    lower_key: str = "lower",
    upper_key: str = "upper",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> str:
    """Render paired differences on a fixed ``[-1, 1]`` scale."""

    ordered = sorted((dict(row) for row in rows), key=lambda row: str(row.get(label_key, "")))
    margin_left, margin_right, top, bottom = 70, 24, 60, height - 66
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, bottom - top)
    count = max(1, len(ordered))
    band = plot_width / count
    scale = plot_height / 2
    zero = top + scale
    body: list[str] = [_header(title, width=width, margin=margin_left), _grid(left=margin_left, top=top, right=width-margin_right, bottom=bottom), _axis(left=margin_left, top=top, right=width-margin_right, bottom=bottom, width=width, height=height)]
    body.append(
        f'<line stroke="#7d8798" stroke-dasharray="4 3" stroke-width="1" x1="{margin_left}" x2="{width-margin_right}" y1="{_fmt(zero)}" y2="{_fmt(zero)}"/>'
    )
    for tick in range(-2, 3):
        value = tick / 2
        y = zero - value * scale
        body.append(
            f'<text fill="#4c5668" font-family="{_FONT}" font-size="11" text-anchor="end" '
            f'x="{margin_left - 8}" y="{_fmt(y + 4)}">{_fmt(value)}</text>'
        )
    for index, row in enumerate(ordered):
        value = min(1.0, max(-1.0, _number(row.get(value_key))))
        x_center = margin_left + band * (index + 0.5)
        x = x_center - min(72.0, band * 0.58) / 2
        bar_width = min(72.0, band * 0.58)
        y_value = zero - value * scale
        y = min(zero, y_value)
        body.append(
            f'<rect fill="#5a9b78" height="{_fmt(abs(y_value - zero))}" rx="2" width="{_fmt(bar_width)}" '
            f'x="{_fmt(x)}" y="{_fmt(y)}"/>'
        )
        low_raw, high_raw = row.get(lower_key), row.get(upper_key)
        if low_raw is not None and high_raw is not None:
            low = min(1.0, max(-1.0, _number(low_raw)))
            high = min(1.0, max(-1.0, _number(high_raw)))
            if high < low:
                low, high = high, low
            y_low = zero - low * scale
            y_high = zero - high * scale
            body.append(
                f'<line stroke="#172033" stroke-width="2" x1="{_fmt(x_center)}" x2="{_fmt(x_center)}" '
                f'y1="{_fmt(y_low)}" y2="{_fmt(y_high)}"/>'
            )
        body.append(
            f'<text fill="#4c5668" font-family="{_FONT}" font-size="11" text-anchor="middle" '
            f'x="{_fmt(x_center)}" y="{height - 34}">{_text(row.get(label_key, ""))}</text>'
        )
    return _document("".join(body), title=title, width=width, height=height)


def render_table(
    rows: Iterable[Mapping[str, Any]],
    *,
    title: str = "Analysis table",
    columns: Sequence[str] | None = None,
    width: int = DEFAULT_WIDTH,
    row_height: int = 22,
) -> str:
    """Render a compact text table for small deterministic evidence files."""

    ordered = [dict(row) for row in rows]
    names = list(columns) if columns is not None else sorted({key for row in ordered for key in row})
    names = [str(name) for name in names]
    width = max(width, 80)
    height = max(100, 66 + row_height * (len(ordered) + 2))
    body: list[str] = [_header(title, width=width, margin=18, y=28)]
    if not names:
        body.append(f'<text fill="#4c5668" font-family="{_FONT}" font-size="12" x="18" y="62">No rows</text>')
        return _document("".join(body), title=title, width=width, height=height)
    cell_width = max(80, (width - 36) // len(names))
    x_values = [18 + index * cell_width for index in range(len(names))]
    y = 58
    for index, name in enumerate(names):
        body.append(
            f'<text fill="#172033" font-family="{_FONT}" font-size="11" font-weight="600" '
            f'x="{x_values[index]}" y="{y}">{_text(name)}</text>'
        )
    for row_index, row in enumerate(ordered):
        y += row_height
        for index, name in enumerate(names):
            value = row.get(name, "")
            body.append(
                f'<text fill="#4c5668" font-family="{_FONT}" font-size="11" '
                f'x="{x_values[index]}" y="{y}">{_text(value)}</text>'
            )
    return _document("".join(body), title=title, width=width, height=height)


def write_svg(path: str | Path, svg: str) -> Path:
    """Write already-rendered SVG bytes with stable UTF-8 encoding."""

    if not isinstance(svg, str) or not svg.endswith("\n"):
        raise ValueError("SVG must be a text document ending in one newline")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(svg.encode("utf-8", "strict"))
    return target


svg_document = _document
render_rate_chart = render_rates
render_difference_chart = render_contrast


__all__ = [
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "render_contrast",
    "render_difference_chart",
    "render_rate_chart",
    "render_rates",
    "render_table",
    "svg_document",
    "write_svg",
]
