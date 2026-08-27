"""Dependency-free deterministic SVG figures from compact CSV tables."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .analysis import final_summary_rows, load_final_rows
from .schemas import (
    BundleValidationError,
    load_table,
    manifest_payload,
    sha256_file,
    validate_compact_bundle,
)
from .stats import finite_values


PLOT_VERSION = "2"
WIDTH = 900
HEIGHT = 560
MARGIN_LEFT = 78
MARGIN_RIGHT = 28
MARGIN_TOP = 62
MARGIN_BOTTOM = 68
COLORS = {
    "blue": "#2563eb",
    "orange": "#d97706",
    "green": "#15803d",
    "red": "#b91c1c",
    "purple": "#7c3aed",
    "grey": "#64748b",
    "light": "#e2e8f0",
    "dark": "#0f172a",
}


class PlotScopeError(ValueError):
    """Raised when a compact bundle cannot identify one plotting sample."""


_TABLE_NAMES = (
    "runs.csv",
    "pair_counts.csv",
    "checkpoint_metrics.csv",
    "final_summary.csv",
    "basin_cells.csv",
    "perturbation_trajectory.csv",
    "audit_control.csv",
    "threshold_sensitivity.csv",
)


def _text_value(row: Mapping[str, object], field: str) -> str:
    value = row.get(field)
    return str(value).strip() if value not in (None, "") else ""


def _build_run_experiment_map(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, str]:
    """Build a checked run-to-experiment map from every tagged run table."""

    mapping: dict[str, str] = {}
    for name in _TABLE_NAMES:
        if name == "basin_cells.csv" or name == "perturbation_trajectory.csv":
            continue
        for row in tables.get(name, ()):
            run_id = _text_value(row, "run_id")
            experiment = _text_value(row, "experiment")
            if not run_id or not experiment:
                continue
            previous = mapping.get(run_id)
            if previous is not None and previous != experiment:
                raise PlotScopeError(
                    f"run {run_id!r} is assigned to multiple experiments: "
                    f"{previous!r} and {experiment!r}"
                )
            mapping[run_id] = experiment
    return mapping


def _row_experiment(
    row: Mapping[str, object],
    run_experiments: Mapping[str, str],
) -> str | None:
    """Resolve a row's experiment and reject conflicting row/run metadata."""

    explicit = _text_value(row, "experiment")
    run_id = _text_value(row, "run_id")
    mapped = run_experiments.get(run_id) if run_id else None
    if explicit and mapped and explicit != mapped:
        raise PlotScopeError(
            f"run {run_id!r} has conflicting experiment labels: "
            f"row={explicit!r}, runs.csv={mapped!r}"
        )
    return explicit or mapped


def _observed_experiments(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    run_experiments: Mapping[str, str],
) -> set[str]:
    observed = set(run_experiments.values())
    for name in _TABLE_NAMES:
        for row in tables.get(name, ()):
            if name == "perturbation_trajectory.csv":
                experiment = _text_value(row, "experiment")
            else:
                experiment = _row_experiment(row, run_experiments)
            if experiment:
                observed.add(experiment)
    return observed


def _resolve_plot_scope(
    manifest: Mapping[str, object],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    run_experiments: Mapping[str, str],
) -> str | None:
    """Select the manifest-declared experiment, or infer one unambiguously."""

    analysis_experiment = _text_value(manifest, "analysis_experiment")
    export_scope = _text_value(manifest, "experiment_scope")
    if analysis_experiment and export_scope and analysis_experiment != export_scope:
        raise PlotScopeError(
            "manifest analysis_experiment and experiment_scope disagree: "
            f"{analysis_experiment!r} versus {export_scope!r}"
        )
    scope = analysis_experiment or export_scope
    observed = _observed_experiments(tables, run_experiments)
    if scope:
        if observed and scope not in observed:
            raise PlotScopeError(
                f"manifest selected experiment {scope!r}, but bundle rows contain "
                f"{', '.join(sorted(observed))}"
            )
        return scope
    if len(observed) > 1:
        raise PlotScopeError(
            "bundle contains multiple experiments without manifest "
            f"analysis_experiment/experiment_scope: {', '.join(sorted(observed))}"
        )
    return next(iter(observed), None)


def _filter_plot_tables(
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    scope: str | None,
    run_experiments: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, object]]], set[str]]:
    """Filter every plotted table to one scope and return its source run IDs."""

    scoped_run_ids = {
        run_id
        for run_id, experiment in run_experiments.items()
        if scope is None or experiment == scope
    }
    filtered: dict[str, list[dict[str, object]]] = {}
    for name in _TABLE_NAMES:
        selected: list[dict[str, object]] = []
        for row in tables.get(name, ()):
            if name == "perturbation_trajectory.csv":
                source_run_id = _text_value(row, "source_run_id")
                if source_run_id and source_run_id in scoped_run_ids:
                    selected.append(dict(row))
                continue
            experiment = _row_experiment(row, run_experiments)
            if scope is None or experiment == scope:
                selected.append(dict(row))
            elif experiment is None and _text_value(row, "run_id"):
                raise PlotScopeError(
                    f"row in {name} has no experiment mapping for scoped bundle "
                    f"{scope!r}: run {_text_value(row, 'run_id')!r}"
                )
        filtered[name] = selected

    # A scoped perturbation plot is meaningful only for source checkpoints
    # that survived the same run-to-experiment filter.
    if scope and not scoped_run_ids and filtered["perturbation_trajectory.csv"]:
        raise PlotScopeError(
            f"perturbation rows refer to {scope!r}, but no scoped source run IDs exist"
        )
    return filtered, scoped_run_ids


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _num(row: Mapping[str, object], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bounds(values: Iterable[float], default: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    finite = finite_values(values)
    if not finite:
        return default
    low, high = min(finite), max(finite)
    if low == high:
        pad = max(0.5, abs(low) * 0.1)
        return low - pad, high + pad
    pad = (high - low) * 0.05
    return low - pad, high + pad


def _scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high == low:
        return (start + end) / 2.0
    return start + (value - low) * (end - start) / (high - low)


def _header(title: str, *, x_label: str, y_label: str) -> str:
    return (
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{WIDTH / 2:.1f}" y="30" text-anchor="middle" '
        f'font-family="monospace" font-size="20" font-weight="600" fill="{COLORS["dark"]}">{_esc(title)}</text>'
        f'<text x="{WIDTH / 2:.1f}" y="{HEIGHT - 20}" text-anchor="middle" '
        f'font-family="monospace" font-size="13" fill="{COLORS["grey"]}">{_esc(x_label)}</text>'
        f'<text x="18" y="{HEIGHT / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {HEIGHT / 2:.1f})" '
        f'font-family="monospace" font-size="13" fill="{COLORS["grey"]}">{_esc(y_label)}</text>'
    )


def _axes(x_low: float, x_high: float, y_low: float, y_high: float) -> str:
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = HEIGHT - MARGIN_BOTTOM, MARGIN_TOP
    pieces = [
        f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="{COLORS["dark"]}"/>',
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="{COLORS["dark"]}"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = x0 + (x1 - x0) * fraction
        y = y0 - (y0 - y1) * fraction
        x_value = x_low + (x_high - x_low) * fraction
        y_value = y_low + (y_high - y_low) * fraction
        pieces.append(
            f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="{COLORS["light"]}" stroke-width="1"/>'
        )
        pieces.append(
            f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{COLORS["light"]}" stroke-width="1"/>'
        )
        pieces.append(
            f'<text x="{x:.2f}" y="{y0 + 22}" text-anchor="middle" font-family="monospace" font-size="11" fill="{COLORS["grey"]}">{x_value:.3g}</text>'
        )
        pieces.append(
            f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="monospace" font-size="11" fill="{COLORS["grey"]}">{y_value:.3g}</text>'
        )
    return "".join(pieces)


def _svg(title: str, body: str, metadata: Mapping[str, object], *, x_label: str, y_label: str) -> str:
    metadata_json = _esc(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
        f'<metadata>{metadata_json}</metadata>'
        f'<g>{_header(title, x_label=x_label, y_label=y_label)}{body}</g></svg>\n'
    )


def _write_figure(
    path: Path,
    svg: str,
    metadata: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    sidecar = path.with_suffix(".metadata.json")
    sidecar.write_text(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_final_gap_histogram(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Render a fixed-bin histogram of independent final compliance gaps."""

    values = [_num(row, "gap") for row in rows]
    gaps = [value for value in values if value is not None]
    bins = 20
    lower, upper = -1.0, 1.0
    counts = [0] * bins
    for gap in gaps:
        index = min(bins - 1, max(0, int((gap - lower) / (upper - lower) * bins)))
        counts[index] += 1
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = HEIGHT - MARGIN_BOTTOM, MARGIN_TOP
    max_count = max(counts) if counts else 1
    body = [_axes(lower, upper, 0.0, float(max_count))]
    bar_width = (x1 - x0) / bins
    for index, count in enumerate(counts):
        height = (y0 - y1) * count / max_count
        x = x0 + index * bar_width + 1
        body.append(
            f'<rect x="{x:.2f}" y="{y0 - height:.2f}" width="{max(1.0, bar_width - 2):.2f}" height="{height:.2f}" fill="{COLORS["blue"]}" opacity="0.78"/>'
        )
    body.append(
        f'<line x1="{_scale(0.1, lower, upper, x0, x1):.2f}" y1="{y1}" x2="{_scale(0.1, lower, upper, x0, x1):.2f}" y2="{y0}" stroke="{COLORS["orange"]}" stroke-dasharray="5 4"/>'
    )
    body.append(
        f'<line x1="{_scale(0.85, lower, upper, x0, x1):.2f}" y1="{y1}" x2="{_scale(0.85, lower, upper, x0, x1):.2f}" y2="{y0}" stroke="{COLORS["red"]}" stroke-dasharray="5 4"/>'
    )
    body.append(
        f'<text x="{x1}" y="{y1 - 12}" text-anchor="end" font-family="monospace" font-size="12" fill="{COLORS["grey"]}">n={len(gaps)}</text>'
    )
    payload = dict(metadata or {})
    payload.setdefault("figure", "fig01_final_gap_histogram")
    payload.setdefault("plot_version", PLOT_VERSION)
    svg = _svg("Independent final compliance gaps", "".join(body), payload, x_label="C_on - C_off", y_label="runs")
    destination = Path(path)
    _write_figure(destination, svg, payload)
    return destination


def _line_path(points: Sequence[tuple[float, float]], x_low: float, x_high: float, y_low: float, y_high: float) -> str:
    if not points:
        return ""
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = HEIGHT - MARGIN_BOTTOM, MARGIN_TOP
    commands = [
        f"M {_scale(points[0][0], x_low, x_high, x0, x1):.2f} {_scale(points[0][1], y_low, y_high, y0, y1):.2f}"
    ]
    for x_value, y_value in points[1:]:
        commands.append(f"L {_scale(x_value, x_low, x_high, x0, x1):.2f} {_scale(y_value, y_low, y_high, y0, y1):.2f}")
    return " ".join(commands)


def render_training_trajectories(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Render deterministic checkpoint trajectories for the main runs."""

    groups: dict[str, list[tuple[float, Mapping[str, object]]]] = defaultdict(list)
    for row in rows:
        if str(row.get("branch_id") or "").strip().lower() not in {"", "main", "primary", "none"}:
            continue
        step = _num(row, "step")
        if step is not None and row.get("run_id"):
            groups[str(row["run_id"])].append((step, row))
    x_values = [step for points in groups.values() for step, _ in points]
    y_values = [value for points in groups.values() for _, row in points for field in ("c_on", "c_off", "gap") if (value := _num(row, field)) is not None]
    x_low, x_high = _bounds(x_values, (0.0, 1.0))
    y_low, y_high = _bounds(y_values, (-1.0, 1.0))
    body = [_axes(x_low, x_high, y_low, y_high)]
    for index, run_id in enumerate(sorted(groups)):
        points = sorted(groups[run_id], key=lambda item: item[0])
        for field, color in (("c_on", COLORS["blue"]), ("c_off", COLORS["orange"]), ("gap", COLORS["green"])):
            values = [(step, value) for step, row in points if (value := _num(row, field)) is not None]
            body.append(f'<path d="{_line_path(values, x_low, x_high, y_low, y_high)}" fill="none" stroke="{color}" stroke-width="1" opacity="{0.20 if index else 0.45}"/>')
    body.append(
        f'<text x="{WIDTH - MARGIN_RIGHT}" y="{MARGIN_TOP - 14}" text-anchor="end" font-family="monospace" font-size="12" fill="{COLORS["grey"]}">runs={len(groups)}; blue=C_on orange=C_off green=gap</text>'
    )
    payload = dict(metadata or {})
    payload.setdefault("figure", "fig02_training_trajectories")
    payload.setdefault("plot_version", PLOT_VERSION)
    svg = _svg("Training trajectories", "".join(body), payload, x_label="checkpoint step", y_label="metric")
    destination = Path(path)
    _write_figure(destination, svg, payload)
    return destination


def render_basin_phase_diagram(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Render strategic-mode probability over the registered basin grid."""

    points = [
        (_num(row, "harm_strength"), _num(row, "audit_sensitivity"), _num(row, "p_strategic"), _num(row, "n_seeds"))
        for row in rows
    ]
    points = [point for point in points if point[0] is not None and point[1] is not None]
    xs = sorted({float(point[0]) for point in points}) or [0.0, 1.0]
    ys = sorted({float(point[1]) for point in points}) or [0.0, 1.0]
    x_low, x_high = _bounds(xs, (0.0, 1.0))
    y_low, y_high = _bounds(ys, (0.0, 1.0))
    body = [_axes(x_low, x_high, y_low, y_high)]
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = HEIGHT - MARGIN_BOTTOM, MARGIN_TOP
    cell_width = (x1 - x0) / max(1, len(xs))
    cell_height = (y0 - y1) / max(1, len(ys))
    for x_value, y_value, probability, n_seeds in points:
        p = max(0.0, min(1.0, probability if probability is not None else 0.0))
        x_center = _scale(float(x_value), x_low, x_high, x0, x1)
        y_center = _scale(float(y_value), y_low, y_high, y0, y1)
        fill = f"rgb({int(37 + 180 * p)},{int(99 - 55 * p)},{int(235 - 180 * p)})"
        body.append(
            f'<rect x="{x_center - cell_width / 2:.2f}" y="{y_center - cell_height / 2:.2f}" width="{max(4.0, cell_width - 2):.2f}" height="{max(4.0, cell_height - 2):.2f}" fill="{fill}" stroke="white"/>'
        )
        if n_seeds is not None:
            body.append(f'<text x="{x_center:.2f}" y="{y_center + 4:.2f}" text-anchor="middle" font-family="monospace" font-size="10" fill="white">{int(n_seeds)}</text>')
    body.append(
        f'<text x="{x1}" y="{y1 - 12}" text-anchor="end" font-family="monospace" font-size="12" fill="{COLORS["grey"]}">cell color=p(strategic); text=n</text>'
    )
    payload = dict(metadata or {})
    payload.setdefault("figure", "fig03_basin_phase_diagram")
    payload.setdefault("plot_version", PLOT_VERSION)
    svg = _svg("Basin phase diagram", "".join(body), payload, x_label="initial harmful-goal strength", y_label="initial audit sensitivity")
    destination = Path(path)
    _write_figure(destination, svg, payload)
    return destination


def render_perturbation_recovery(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Render source-distance and opposite-distance continuation curves."""

    groups: dict[str, list[tuple[float, Mapping[str, object]]]] = defaultdict(list)
    for row in rows:
        branch = str(row.get("branch_id") or "")
        step = _num(row, "step_since_branch")
        if branch and step is not None:
            groups[branch].append((step, row))
    x_values = [step for points in groups.values() for step, _ in points]
    y_values = [value for points in groups.values() for _, row in points for field in ("d_source", "d_opposite") if (value := _num(row, field)) is not None]
    x_low, x_high = _bounds(x_values, (0.0, 1.0))
    y_low, y_high = _bounds(y_values, (0.0, 1.0))
    body = [_axes(x_low, x_high, y_low, y_high)]
    for index, branch in enumerate(sorted(groups)):
        points = sorted(groups[branch], key=lambda item: item[0])
        for field, color in (("d_source", COLORS["blue"]), ("d_opposite", COLORS["red"])):
            values = [(step, value) for step, row in points if (value := _num(row, field)) is not None]
            body.append(f'<path d="{_line_path(values, x_low, x_high, y_low, y_high)}" fill="none" stroke="{color}" stroke-width="1.5" opacity="{0.30 if index else 0.70}"/>')
    body.append(
        f'<text x="{WIDTH - MARGIN_RIGHT}" y="{MARGIN_TOP - 14}" text-anchor="end" font-family="monospace" font-size="12" fill="{COLORS["grey"]}">branches={len(groups)}; blue=d_source red=d_opposite</text>'
    )
    payload = dict(metadata or {})
    payload.setdefault("figure", "fig04_perturbation_recovery")
    payload.setdefault("plot_version", PLOT_VERSION)
    svg = _svg("Perturbation continuation", "".join(body), payload, x_label="steps since branch", y_label="endpoint distance")
    destination = Path(path)
    _write_figure(destination, svg, payload)
    return destination


def render_reward_misalignment(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Render checkpoint reward against hidden-misalignment rate."""

    points = [(_num(row, "train_reward"), _num(row, "hidden_misalignment_rate")) for row in rows]
    points = [(x, y) for x, y in points if x is not None and y is not None]
    x_low, x_high = _bounds((x for x, _ in points), (0.0, 1.0))
    y_low, y_high = _bounds((y for _, y in points), (0.0, 1.0))
    body = [_axes(x_low, x_high, y_low, y_high)]
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = HEIGHT - MARGIN_BOTTOM, MARGIN_TOP
    for x_value, y_value in points:
        body.append(
            f'<circle cx="{_scale(x_value, x_low, x_high, x0, x1):.2f}" cy="{_scale(y_value, y_low, y_high, y0, y1):.2f}" r="3" fill="{COLORS["purple"]}" opacity="0.55"/>'
        )
    body.append(f'<text x="{x1}" y="{y1 - 12}" text-anchor="end" font-family="monospace" font-size="12" fill="{COLORS["grey"]}">checkpoints={len(points)}</text>')
    payload = dict(metadata or {})
    payload.setdefault("figure", "fig05_reward_vs_hidden_misalignment")
    payload.setdefault("plot_version", PLOT_VERSION)
    svg = _svg("Training reward and hidden misalignment", "".join(body), payload, x_label="training reward", y_label="hidden misalignment rate")
    destination = Path(path)
    _write_figure(destination, svg, payload)
    return destination


def render_control_audit_swaps(
    rows: Sequence[Mapping[str, object]],
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    """Render audit-variant endpoints when a control table is available."""

    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("eval_variant") or row.get("condition") or "default")
        c_on, c_off = _num(row, "c_on"), _num(row, "c_off")
        if c_on is not None and c_off is not None:
            groups[variant].append((c_on, c_off))
    points = [(index, c_off, variant) for index, variant in enumerate(sorted(groups)) for _, c_off in groups[variant]]
    x_low, x_high = 0.0, max(1.0, float(len(groups)))
    y_low, y_high = 0.0, 1.0
    body = [_axes(x_low, x_high, y_low, y_high)]
    x0, x1 = MARGIN_LEFT, WIDTH - MARGIN_RIGHT
    y0, y1 = HEIGHT - MARGIN_BOTTOM, MARGIN_TOP
    for index, c_off, variant in points:
        body.append(
            f'<circle cx="{_scale(index + 0.5, x_low, x_high, x0, x1):.2f}" cy="{_scale(c_off, y_low, y_high, y0, y1):.2f}" r="4" fill="{COLORS["green"]}" opacity="0.65"/>'
        )
    for index, variant in enumerate(sorted(groups)):
        x = _scale(index + 0.5, x_low, x_high, x0, x1)
        body.append(f'<text x="{x:.2f}" y="{y0 + 40}" text-anchor="middle" font-family="monospace" font-size="10" fill="{COLORS["grey"]}">{_esc(variant[:14])}</text>')
    payload = dict(metadata or {})
    payload.setdefault("figure", "fig06_control_audit_swaps")
    payload.setdefault("plot_version", PLOT_VERSION)
    svg = _svg("Audit-variant control endpoints", "".join(body), payload, x_label="evaluation variant", y_label="C_off")
    destination = Path(path)
    _write_figure(destination, svg, payload)
    return destination


FIGURE_SPECS = (
    ("fig01_final_gap_histogram.svg", "final_gap"),
    ("fig02_training_trajectories.svg", "trajectories"),
    ("fig03_basin_phase_diagram.svg", "basin"),
    ("fig04_perturbation_recovery.svg", "perturbation"),
    ("fig05_reward_vs_hidden_misalignment.svg", "reward"),
    ("fig06_control_audit_swaps.svg", "control"),
)


def _source_metadata(bundle: Path, source_names: Sequence[str], command: str) -> dict[str, object]:
    records: list[dict[str, str]] = []
    for name in source_names:
        path = bundle / name
        if path.is_file():
            records.append({"path": name, "sha256": sha256_file(path)})
    return {
        "plot_version": PLOT_VERSION,
        "command": command,
        "source_files": records,
    }


def plot_all(
    bundle: str | Path,
    output_dir: str | Path,
    *,
    figures: Sequence[str] | None = None,
    strict: bool = False,
    command: str = "python -m lean_reward_hacking.analysis plots",
) -> list[Path]:
    """Regenerate all requested figures from compact tables only."""

    root = Path(bundle)
    if strict:
        problems = validate_compact_bundle(root)
        if problems:
            raise ValueError("; ".join(problems))
    out = Path(output_dir)
    selected = set(figures or [name[:-4] for name, _ in FIGURE_SPECS])
    tables = {name: load_table(root, name) for name in _TABLE_NAMES}
    try:
        manifest = manifest_payload(root)
    except BundleValidationError as exc:
        raise PlotScopeError(f"cannot read plotting manifest: {exc}") from exc
    run_experiments = _build_run_experiment_map(tables)
    scope = _resolve_plot_scope(manifest, tables, run_experiments)
    scoped_tables, scoped_run_ids = _filter_plot_tables(
        tables,
        scope=scope,
        run_experiments=run_experiments,
    )
    checkpoints = scoped_tables["checkpoint_metrics.csv"]
    finals = load_final_rows(root) if scope is None else final_summary_rows(
        scoped_tables["final_summary.csv"] or checkpoints
    )
    basin = scoped_tables["basin_cells.csv"]
    perturbations = scoped_tables["perturbation_trajectory.csv"]
    control = scoped_tables["final_summary.csv"] or checkpoints
    source_names = [
        name
        for name in (
            "runs.csv",
            "pair_counts.csv",
            "final_summary.csv",
            "checkpoint_metrics.csv",
            "basin_cells.csv",
            "perturbation_trajectory.csv",
        )
        if (root / name).is_file()
    ]
    rendered: list[Path] = []
    for filename, kind in FIGURE_SPECS:
        stem = filename[:-4]
        if stem not in selected and kind not in selected and "all" not in selected:
            continue
        path = out / filename
        metadata = _source_metadata(root, source_names, command)
        metadata["figure"] = stem
        metadata["analysis_experiment"] = scope
        metadata["experiment_scope"] = _text_value(manifest, "experiment_scope") or None
        metadata["scoped_run_ids"] = sorted(scoped_run_ids)
        if kind == "final_gap":
            render_final_gap_histogram(finals, path, metadata=metadata)
        elif kind == "trajectories":
            render_training_trajectories(checkpoints, path, metadata=metadata)
        elif kind == "basin":
            render_basin_phase_diagram(basin, path, metadata=metadata)
        elif kind == "perturbation":
            render_perturbation_recovery(perturbations, path, metadata=metadata)
        elif kind == "reward":
            render_reward_misalignment(checkpoints, path, metadata=metadata)
        elif kind == "control":
            render_control_audit_swaps(control, path, metadata=metadata)
        rendered.append(path)
    return rendered


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lean-reward-hacking-plots")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--figure", action="append", dest="figures")
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plot_all(args.bundle, args.out, figures=args.figures, strict=args.strict)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FIGURE_SPECS",
    "PlotScopeError",
    "PLOT_VERSION",
    "main",
    "plot_all",
    "render_basin_phase_diagram",
    "render_control_audit_swaps",
    "render_final_gap_histogram",
    "render_perturbation_recovery",
    "render_reward_misalignment",
    "render_training_trajectories",
]
