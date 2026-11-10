from __future__ import annotations

import csv
import re
from decimal import Decimal
from pathlib import Path

METHODS = ("wald", "cal", "raw")
METRICS = ("coverage", "mean_width", "mean_bias", "sd_bias")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")

type SummaryRow = dict[str, str]
type SummaryRows = dict[str, dict[Decimal, SummaryRow]]


def read_summary(path: Path) -> tuple[SummaryRows, list[Decimal]]:
    """Read the plotting columns without rounding their numerical strings.

    Args:
        path (Path): CSV summary containing each method at every learning rate.

    Returns:
        tuple[SummaryRows, list[Decimal]]: Rows grouped by method and the sorted
            learning-rate grid.
    """
    rows: SummaryRows = {method: {} for method in METHODS}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"eta", "method", *METRICS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing summary columns: " + ", ".join(sorted(missing)))
        for line, raw in enumerate(reader, start=2):
            method = raw["method"].strip()
            if method not in rows:
                raise ValueError(f"Unknown method {method!r} at CSV line {line}")
            row = {name: (raw[name] or "").strip() for name in ("eta", *METRICS)}
            if any(not NUMBER.fullmatch(value) for value in row.values()):
                raise ValueError(f"Invalid numerical value at CSV line {line}")
            eta = Decimal(row["eta"])
            if eta <= 0 or eta in rows[method]:
                raise ValueError(f"Nonpositive or duplicate rate at CSV line {line}")
            rows[method][eta] = row
    grid = sorted(rows[METHODS[0]])
    if not grid or any(sorted(rows[method]) != grid for method in METHODS):
        raise ValueError("All methods must have the same nonempty learning-rate grid")
    return rows, grid


def coordinate_pairs(
    rows: SummaryRows, grid: list[Decimal], metric: str, method: str
) -> str:
    """Format one metric and method as PGFPlots coordinates.

    Args:
        rows (SummaryRows): Summary values grouped by method and learning rate.
        grid (list[Decimal]): Sorted learning rates.
        metric (str): Name of the metric to plot.
        method (str): Name of the inference method to plot.

    Returns:
        str: Coordinate pairs retaining the CSV's original number strings.
    """
    return " ".join(
        f"({rows[method][eta]['eta']},{rows[method][eta][metric]})" for eta in grid
    )


def render_figure(template: Path, rows: SummaryRows, grid: list[Decimal]) -> str:
    """Fill all coordinate series in the figure template.

    Args:
        template (Path): TeX template containing metric and method markers.
        rows (SummaryRows): Summary values grouped by method and learning rate.
        grid (list[Decimal]): Sorted learning rates.

    Returns:
        str: Complete TeX figure with embedded plotting data.
    """
    text = template.read_text(encoding="utf-8")
    for metric in METRICS:
        for method in METHODS:
            marker = f"@@{metric.upper()}_{method.upper()}@@"
            text = text.replace(marker, coordinate_pairs(rows, grid, metric, method))
    return "% Generated from summary_results.csv.\n" + text.rstrip() + "\n"


def export_tex(output: Path = DEFAULT_OUTPUT) -> Path:
    """Generate the four-panel figure from the saved CSV summary.

    Args:
        output (Path): Directory containing summary_results.csv.

    Returns:
        Path: Path to the generated gibbs_four_panel.tex file.
    """
    rows, grid = read_summary(output / "summary_results.csv")
    template = Path(__file__).with_name("figure_template.tex")
    destination = output / "gibbs_four_panel.tex"
    destination.write_text(render_figure(template, rows, grid), encoding="utf-8")
    return destination
