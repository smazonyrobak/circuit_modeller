from __future__ import annotations

import time
from pathlib import Path

from plotly import graph_objects as go

from .simulator import FISweepResult, IVSweepResult, SimulationResult

try:
    from plotly_resampler import FigureResampler
except ImportError:
    FigureResampler = None


ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT_DIR / "plotly_outputs"
LIVE_DASHBOARD_MAX_POINTS = 1200

PANEL_TRACE_COLORS = {
    "V_m": "#1d4ed8",
    "Total Applied Current": "#dc2626",
    "Configured Command Current": "#059669",
    "Configured Command Density": "#059669",
    "I_Na": "#0ea5e9",
    "I_K": "#eab308",
    "I_leak": "#ec4899",
    "I_Cl": "#8b5cf6",
    "I_shunt": "#f97316",
    "I_inhib_total": "#6b7280",
    "g_Na": "#06b6d4",
    "g_K": "#84cc16",
    "g_leak": "#ef4444",
    "g_Cl": "#8b5cf6",
    "g_shunt": "#f97316",
    "g_inhib_total": "#6b7280",
    "m": "#ef4444",
    "h": "#22c55e",
    "n": "#2563eb",
}


def _ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _rounded(values: list[float], digits: int = 5) -> list[float]:
    return [round(float(value), digits) for value in values]


def _downsample_series(
    x_values: list[float],
    y_values: list[float],
    max_points: int | None,
) -> tuple[list[float], list[float]]:
    if max_points is None or len(x_values) <= max_points or len(y_values) <= max_points:
        return x_values, y_values
    if max_points < 3:
        return [x_values[0], x_values[-1]], [y_values[0], y_values[-1]]

    interior_count = len(x_values) - 2
    bucket_count = max(1, (max_points - 2) // 2)
    bucket_size = max(1, int((interior_count + bucket_count - 1) / bucket_count))

    selected_indices = [0]
    start = 1
    while start < len(x_values) - 1:
        end = min(len(x_values) - 1, start + bucket_size)
        bucket = y_values[start:end]
        if not bucket:
            break
        min_offset = min(range(len(bucket)), key=bucket.__getitem__)
        max_offset = max(range(len(bucket)), key=bucket.__getitem__)
        first = start + min(min_offset, max_offset)
        second = start + max(min_offset, max_offset)
        if first not in selected_indices:
            selected_indices.append(first)
        if second not in selected_indices:
            selected_indices.append(second)
        start = end
    if selected_indices[-1] != len(x_values) - 1:
        selected_indices.append(len(x_values) - 1)

    if len(selected_indices) > max_points:
        stride = max(1, int((len(selected_indices) - 1) / (max_points - 1)))
        selected_indices = selected_indices[::stride]
        if selected_indices[-1] != len(x_values) - 1:
            selected_indices.append(len(x_values) - 1)

    selected_indices = sorted(set(selected_indices))
    return (
        _rounded([x_values[index] for index in selected_indices]),
        _rounded([y_values[index] for index in selected_indices]),
    )


def _base_figure(
    title: str,
    yaxis_title: str,
    theme: str,
    height: int = 220,
    yaxis_range: list[float] | None = None,
    xaxis_range: list[float] | None = None,
):
    figure = go.Figure()
    figure.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=height,
        margin={"l": 60, "r": 18, "t": 42, "b": 58},
        title=title,
        showlegend=False,
        hovermode="x unified",
        xaxis={"title": "Time (ms)", "autorange": xaxis_range is None, "fixedrange": False},
        yaxis={"title": yaxis_title, "autorange": yaxis_range is None, "fixedrange": False},
        datarevision=str(time.time_ns()),
    )
    if xaxis_range is not None:
        figure.update_xaxes(range=xaxis_range)
    if yaxis_range is not None:
        figure.update_yaxes(range=yaxis_range)
    return figure


def _compute_axis_range(traces: list[tuple[str, list[float]]], fixed_range: list[float] | None = None) -> list[float] | None:
    if fixed_range is not None:
        return fixed_range
    values = [float(value) for _, trace_values in traces for value in trace_values]
    if not values:
        return None
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        padding = max(1.0, abs(min_value) * 0.1)
        return [min_value - padding, max_value + padding]
    padding = (max_value - min_value) * 0.08
    return [min_value - padding, max_value + padding]


def _compute_value_range(values: list[float], pad_ratio: float = 0.02) -> list[float] | None:
    if not values:
        return None
    min_value = float(min(values))
    max_value = float(max(values))
    if min_value == max_value:
        padding = max(1.0, abs(min_value) * max(pad_ratio, 0.02))
        return [min_value - padding, max_value + padding]
    padding = (max_value - min_value) * pad_ratio
    return [min_value - padding, max_value + padding]


def _resampler_figure(
    title: str,
    yaxis_title: str,
    theme: str,
    height: int = 220,
    yaxis_range: list[float] | None = None,
    xaxis_range: list[float] | None = None,
):
    if FigureResampler is None:
        return _base_figure(title, yaxis_title, theme, height=height, yaxis_range=yaxis_range, xaxis_range=xaxis_range)
    figure = FigureResampler(
        _base_figure(title, yaxis_title, theme, height=height, yaxis_range=yaxis_range, xaxis_range=xaxis_range)
    )
    return figure


def _add_trace(
    figure,
    x_values: list[float],
    y_values: list[float],
    name: str,
    color: str | None,
    max_points_per_trace: int | None,
):
    trace_kwargs = {
        "mode": "lines",
        "name": name,
    }
    if color:
        trace_kwargs["line"] = {"color": color, "width": 2}
    if FigureResampler is not None and isinstance(figure, FigureResampler):
        sampled_x, sampled_y = _downsample_series(x_values, y_values, min(max_points_per_trace or LIVE_DASHBOARD_MAX_POINTS, 800))
        figure.add_trace(
            go.Scattergl(x=sampled_x, y=sampled_y, **trace_kwargs),
            hf_x=x_values,
            hf_y=y_values,
        )
        return
    sampled_x, sampled_y = _downsample_series(x_values, y_values, max_points_per_trace)
    figure.add_trace(go.Scattergl(x=sampled_x, y=sampled_y, **trace_kwargs))


def _panel_specifications(result: SimulationResult, selected_panels: set[str] | None):
    panels = selected_panels if selected_panels is not None else {"applied_trace", "command_trace", "ionic_currents", "gating", "conductances"}
    specs = [
        ("voltage", "Membrane Voltage", "mV", [("V_m", result.voltage_mV)], None, 240),
    ]
    if "applied_trace" in panels and result.current_trace:
        specs.append(("applied_trace", result.current_trace_label, result.current_trace_unit, [(result.current_trace_label, result.current_trace)], None, 220))
    if "command_trace" in panels and result.command_trace:
        specs.append(("command_trace", result.command_trace_label, result.command_trace_unit, [(result.command_trace_label, result.command_trace)], None, 220))
    if "ionic_currents" in panels and result.ionic_currents_uA_cm2:
        specs.append(("ionic_currents", "Ionic Currents", "uA/cm2", list(result.ionic_currents_uA_cm2.items()), None, 220))
    if "gating" in panels and result.gating_variables:
        specs.append(("gating", "Gating Variables", "Probability", list(result.gating_variables.items()), [-0.02, 1.02], 220))
    if "conductances" in panels and result.conductances_mS_cm2:
        specs.append(("conductances", "Conductances", "mS/cm2", list(result.conductances_mS_cm2.items()), None, 220))
    return specs


def build_trace_panel_figures(
    result: SimulationResult,
    selected_panels: set[str] | None = None,
    theme: str = "light",
    max_points_per_trace: int | None = LIVE_DASHBOARD_MAX_POINTS,
) -> dict[str, go.Figure]:
    figures: dict[str, go.Figure] = {}
    time_axis_range = _compute_value_range(result.times_ms, pad_ratio=0.0)
    for panel_key, title, unit, traces, fixed_range, height in _panel_specifications(result, selected_panels):
        figure = _resampler_figure(
            title,
            unit,
            theme,
            height=height,
            yaxis_range=_compute_axis_range(traces, fixed_range),
            xaxis_range=time_axis_range,
        )
        for trace_name, trace_values in traces:
            color = PANEL_TRACE_COLORS.get(trace_name)
            if color is None and panel_key == "applied_trace":
                color = "#dc2626"
            if color is None and panel_key == "command_trace":
                color = "#059669"
            _add_trace(
                figure,
                result.times_ms,
                trace_values,
                trace_name,
                color,
                max_points_per_trace,
            )
        if panel_key == "voltage":
            figure.add_annotation(
                text=f"E_leak = {result.eleak_mV:.3f} mV    E_Cl = {result.ecl_mV:.3f} mV",
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.12,
                showarrow=False,
                xanchor="right",
            )
        figures[panel_key] = figure
    return figures


def build_live_trace_dashboard_figure(
    result: SimulationResult,
    title: str = "Clamp Dashboard",
    selected_panels: set[str] | None = None,
    theme: str = "light",
):
    figures = build_trace_panel_figures(result, selected_panels, theme)
    return figures.get("voltage", go.Figure())


def build_fi_curve_figure(fi_result: FISweepResult, title: str = "F-I Curve", theme: str = "light") -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=fi_result.current_values,
            y=fi_result.spike_count,
            mode="lines+markers",
            line={"color": "#1d4ed8", "width": 2},
            marker={"size": 7},
            name="Spike count",
        )
    )
    figure.update_layout(
        title=title,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=500,
        showlegend=False,
        xaxis={
            "title": f"Injected current ({fi_result.current_unit})",
            "autorange": False,
            "range": _compute_value_range(fi_result.current_values, pad_ratio=0.02),
            "fixedrange": False,
        },
        yaxis={
            "title": "Spikes",
            "autorange": False,
            "range": _compute_axis_range([("Spike count", fi_result.spike_count)]),
            "fixedrange": False,
        },
        datarevision=str(time.time_ns()),
    )
    figure.add_annotation(
        text=(
            f"Pulse start = {fi_result.pulse_start_ms:.2f} ms, "
            f"pulse width = {fi_result.pulse_width_ms:.2f} ms"
        ),
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.08,
        showarrow=False,
        xanchor="right",
    )
    return figure


def write_trace_dashboard(result: SimulationResult, title: str = "Clamp Dashboard", selected_panels: set[str] | None = None, theme: str = "light") -> Path:
    output_dir = _ensure_output_dir()
    output_path = output_dir / "latest_clamp_dashboard.html"
    figures = build_trace_panel_figures(result, selected_panels, theme, max_points_per_trace=None)
    wrapper = go.Figure()
    if figures:
        wrapper = next(iter(figures.values()))
    wrapper.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def write_live_trace_dashboard(result: SimulationResult, title: str = "Clamp Dashboard", selected_panels: set[str] | None = None, theme: str = "light") -> Path:
    return write_trace_dashboard(result, title, selected_panels, theme)


def build_iv_curve_figure(iv_result: IVSweepResult, title: str = "I-V Curve", theme: str = "light") -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=iv_result.command_voltage_mV, y=iv_result.peak_inward_current_nA, mode="lines+markers", name="Peak inward"))
    figure.add_trace(go.Scatter(x=iv_result.command_voltage_mV, y=iv_result.peak_outward_current_nA, mode="lines+markers", name="Peak outward"))
    figure.add_trace(go.Scatter(x=iv_result.command_voltage_mV, y=iv_result.steady_current_nA, mode="lines+markers", name="Steady-state"))
    figure.update_layout(
        title=title,
        template="plotly_dark" if theme == "dark" else "plotly_white",
        height=620,
        xaxis_title="Command voltage (mV)",
        yaxis_title="Clamp current (nA)",
        legend_title="Measurement",
    )
    return figure


def write_iv_curve(iv_result: IVSweepResult, title: str = "I-V Curve") -> Path:
    output_dir = _ensure_output_dir()
    output_path = output_dir / "latest_iv_curve.html"
    figure = build_iv_curve_figure(iv_result, title)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def write_fi_curve(fi_result: FISweepResult, title: str = "F-I Curve") -> Path:
    output_dir = _ensure_output_dir()
    output_path = output_dir / "latest_fi_curve.html"
    figure = build_fi_curve_figure(fi_result, title)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path
