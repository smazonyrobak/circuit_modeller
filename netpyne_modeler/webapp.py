from __future__ import annotations

import math
import socket
import threading
import time
import webbrowser

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update
from dash.exceptions import PreventUpdate

try:
    import dash_cytoscape as cyto
except ImportError:  # pragma: no cover - optional at runtime
    cyto = None

from .circuit_state import CircuitProject
from .plotly_views import build_fi_curve_figure, build_trace_panel_figures
from .simulator import (
    CURRENT_CLAMP,
    CURRENT_INJECTION_UNITS,
    CURRENT_UNIT_DENSITY,
    CURRENT_UNIT_NA,
    CircuitConnectionSpec,
    CircuitNeuronSpec,
    FISweepConfig,
    MorphologyPreview,
    MorphologySection,
    MorphologySite,
    NeuronConfig,
    VOLTAGE_CLAMP,
    VoltagePulseTrain,
    current_nA_to_value,
    current_value_to_nA,
    default_recording_site,
    default_setup,
    estimate_site_segment_area_cm2,
    interpolate_section_site,
    list_available_swc_files,
    load_morphology_preview,
    resolve_morphology_path,
    simulate_circuit_current_clamp,
    simulate_current_clamp,
    simulate_fi_sweep,
    simulate_voltage_clamp,
)


PLOT_PANEL_LABELS = {
    "applied_trace": "Applied current trace",
    "command_trace": "Stimulus / command trace",
    "ionic_currents": "I_Na, I_K, I_leak, I_Cl, I_shunt",
    "gating": "m, h, n gating variables",
    "conductances": "g_Na, g_K, g_leak, g_Cl, g_shunt",
}
PLOT_PANEL_DEFAULTS = {
    "applied_trace": True,
    "command_trace": True,
    "ionic_currents": False,
    "gating": False,
    "conductances": False,
}
PLOT_PANEL_DEFAULT_SELECTION = [key for key, enabled in PLOT_PANEL_DEFAULTS.items() if enabled]

RECORDING_SOURCE_OPTIONS = [
    {"label": "Dedicated patch site", "value": "patch"},
    {"label": "Selected stimulus input", "value": "train"},
    {"label": "Selected incoming connection", "value": "connection"},
    {"label": "Neuron output site", "value": "output"},
]

SECTION_COLORS = {
    "soma": "#1f2937",
    "axon": "#ea580c",
    "dend": "#2563eb",
    "apic": "#059669",
}

NEURON_FIELDS = [
    ("duration_ms", "Duration (ms)", 50.0, 500.0, 1.0),
    ("dt_ms", "dt (ms)", 0.005, 0.2, 0.001),
    ("v_rest_mV", "Initial V (mV)", -100.0, 20.0, 0.5),
    ("holding_mV", "Holding V (mV)", -120.0, 40.0, 0.5),
    ("holding_current", "Holding current", -20.0, 20.0, 0.05),
    ("cm_uF_cm2", "Cm (uF/cm2)", 0.1, 5.0, 0.05),
    ("gna_mS_cm2", "gNa (mS/cm2)", 0.0, 500.0, 1.0),
    ("gk_mS_cm2", "gK (mS/cm2)", 0.0, 200.0, 0.5),
    ("gl_mS_cm2", "gL (mS/cm2)", 0.0, 5.0, 0.01),
    ("ena_mV", "ENa (mV)", 20.0, 100.0, 0.5),
    ("ek_mV", "EK (mV)", -120.0, -20.0, 0.5),
    ("eleak_mV", "E_leak (mV)", -100.0, 20.0, 0.5),
    ("ecl_mV", "E_Cl (mV)", -120.0, 20.0, 0.5),
    ("gcl_mS_cm2", "gCl (mS/cm2)", 0.0, 5.0, 0.01),
    ("gshunt_mS_cm2", "g_shunt (mS/cm2)", 0.0, 5.0, 0.01),
]

FI_FIELDS = [
    ("start_current", "Start current", -200.0, 200.0, 0.1),
    ("end_current", "End current", -200.0, 200.0, 0.1),
    ("step_current", "Current increment", 0.01, 50.0, 0.01),
    ("pulse_start_ms", "Pulse start (ms)", 0.0, 500.0, 0.5),
    ("pulse_width_ms", "Pulse width (ms)", 0.5, 500.0, 0.5),
]

TRAIN_COLUMNS = [
    {"id": "label", "name": "Label", "type": "text", "editable": True},
    {"id": "start_ms", "name": "Start (ms)", "type": "numeric", "editable": True},
    {"id": "pulse_width_ms", "name": "Width (ms)", "type": "numeric", "editable": True},
    {"id": "interval_ms", "name": "Interval (ms)", "type": "numeric", "editable": True},
    {"id": "pulse_count", "name": "Count", "type": "numeric", "editable": True},
    {"id": "amplitude", "name": "Amplitude", "type": "numeric", "editable": True},
    {"id": "section_name", "name": "Section", "type": "text", "editable": False},
    {"id": "section_x", "name": "Loc", "type": "numeric", "editable": False},
]

VOLTAGE_TRAIN_COLUMNS = [
    {"id": "label", "name": "Label", "type": "text", "editable": True},
    {"id": "start_ms", "name": "Start (ms)", "type": "numeric", "editable": True},
    {"id": "pulse_width_ms", "name": "Width (ms)", "type": "numeric", "editable": True},
    {"id": "interval_ms", "name": "Interval (ms)", "type": "numeric", "editable": True},
    {"id": "pulse_count", "name": "Count", "type": "numeric", "editable": True},
    {"id": "amplitude", "name": "Step from hold (mV)", "type": "numeric", "editable": True},
]

MORPH_SAMPLE_SPACING_UM = 18.0


def _slider_id(prefix: str, key: str) -> str:
    return f"{prefix}-slider-{key}"


def _display_id(prefix: str, key: str) -> str:
    return f"{prefix}-display-{key}"


def _format_numeric_value(value: float | int | None) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if abs(numeric) >= 100 or numeric.is_integer():
        return f"{numeric:.0f}" if numeric.is_integer() else f"{numeric:.2f}"
    if abs(numeric) >= 10:
        return f"{numeric:.2f}"
    if abs(numeric) >= 1:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def _coerce_float(value: object, fallback: float) -> float:
    if value in (None, ""):
        return float(fallback)
    return float(value)


def _coerce_int(value: object, fallback: int) -> int:
    if value in (None, ""):
        return int(fallback)
    return int(float(value))


def _number_field(prefix: str, key: str, label: str, value: float, minimum: float, maximum: float, step: float):
    return html.Div(
        html.Div(
            [
                # Keep these controls slider-only with a single readout on the right.
                # Do not add a second editable value field under or beside the slider.
                html.Label(label, className="control-label", id=f"{prefix}-label-{key}"),
                dcc.Slider(
                    id=_slider_id(prefix, key),
                    min=minimum,
                    max=maximum,
                    step=step,
                    value=value,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="mouseup",
                    className="control-slider",
                ),
                html.Div(
                    _format_numeric_value(value),
                    id=_display_id(prefix, key),
                    className="slider-value",
                ),
            ],
            className="compact-field",
        ),
        className="control-col",
    )


def _project_from_dict(data: dict | None) -> CircuitProject:
    return CircuitProject.from_dict(data or CircuitProject.default().to_dict())


def _project_to_dict(project: CircuitProject) -> dict:
    return project.to_dict()


def _selected_neuron(project: CircuitProject, neuron_id: str | None):
    neuron = project.neuron_by_id(neuron_id) if neuron_id else None
    return neuron or (project.neurons[0] if project.neurons else None)


def _selected_connection(project: CircuitProject, connection_id: str | None):
    return project.connection_by_id(connection_id) if connection_id else None


def _site_to_dict(site: MorphologySite) -> dict:
    return {
        "section_name": site.section_name,
        "section_x": site.section_x,
        "x_um": site.x_um,
        "y_um": site.y_um,
        "z_um": site.z_um,
    }


def _site_from_dict(data: dict | None) -> MorphologySite:
    if not data:
        return MorphologySite()
    return MorphologySite(**data)


def _format_site(site: MorphologySite | None) -> str:
    if site is None:
        return "Not assigned"
    return f"{site.section_name} @ {site.section_x:.2f} ({site.x_um:.1f}, {site.y_um:.1f}, {site.z_um:.1f}) um"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    raw = color.lstrip("#")
    if len(raw) != 6:
        return (37, 99, 235)
    try:
        return tuple(int(raw[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError:
        return (37, 99, 235)


def _darken_hex_color(color: str, factor: float = 0.58) -> str:
    red, green, blue = _hex_to_rgb(color)
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(red * factor))),
        max(0, min(255, int(green * factor))),
        max(0, min(255, int(blue * factor))),
    )


def _contrast_text_color(color: str) -> str:
    red, green, blue = _hex_to_rgb(color)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
    return "#172033" if luminance > 0.62 else "#f8fafc"


def _site_matches(first: MorphologySite | None, second: MorphologySite | None) -> bool:
    if first is None or second is None:
        return False
    return first.section_name == second.section_name and abs(first.section_x - second.section_x) < 1e-9


def _materialize_site(site: MorphologySite, morphology_name: str) -> MorphologySite:
    if any(abs(value) > 1e-9 for value in (site.x_um, site.y_um, site.z_um)):
        return site
    preview = load_morphology_preview(morphology_name)
    section = next((item for item in preview.sections if item.name == site.section_name), None)
    if section is None:
        return site
    return interpolate_section_site(section, site.section_x)


def _train_to_row(train: VoltagePulseTrain) -> dict:
    return {
        "label": train.label,
        "start_ms": train.start_ms,
        "pulse_width_ms": train.pulse_width_ms,
        "interval_ms": train.interval_ms,
        "pulse_count": train.pulse_count,
        "amplitude": train.amplitude,
        "section_name": train.section_name,
        "section_x": round(train.section_x, 4),
    }


def _row_to_train(row: dict) -> VoltagePulseTrain:
    return VoltagePulseTrain(
        label=str(row.get("label", "")).strip() or "Pulse Train",
        start_ms=_coerce_float(row.get("start_ms"), 5.0),
        pulse_width_ms=_coerce_float(row.get("pulse_width_ms"), 25.0),
        interval_ms=_coerce_float(row.get("interval_ms"), 30.0),
        pulse_count=max(1, _coerce_int(row.get("pulse_count"), 1)),
        amplitude=_coerce_float(row.get("amplitude"), 0.2),
        section_name=str(row.get("section_name") or "soma_0"),
        section_x=_coerce_float(row.get("section_x"), 0.5),
    )


def _serialize_trains(trains: list[VoltagePulseTrain]) -> list[dict]:
    return [_train_to_row(train) for train in trains]


def _deserialize_trains(rows: list[dict]) -> list[VoltagePulseTrain]:
    return [_row_to_train(row) for row in rows]


def _sample_section_points(section: MorphologySection) -> list[MorphologySite]:
    if section.total_length_um <= 0:
        return [interpolate_section_site(section, 0.5)]
    sample_count = max(2, min(18, int(math.ceil(section.total_length_um / MORPH_SAMPLE_SPACING_UM)) + 1))
    return [interpolate_section_site(section, index / (sample_count - 1)) for index in range(sample_count)]


def _soma_marker_size_px(section: MorphologySection) -> float:
    if not section.points_3d:
        return 12.0
    max_diameter_um = max(point[3] for point in section.points_3d)
    return max(10.0, min(16.0, max_diameter_um * 0.95))


def _soma_fill_points(preview: MorphologyPreview, soma_sections: list[MorphologySection]) -> list[tuple[float, float, float]]:
    if not soma_sections:
        return []
    soma_section = soma_sections[0]
    soma_center = interpolate_section_site(soma_section, 0.5)
    soma_radius_um = max(point[3] for point in soma_section.points_3d) / 2.0 if soma_section.points_3d else 7.0
    attachment_threshold_um = max(8.0, soma_radius_um * 1.3)

    fill_points = [point[:3] for section in soma_sections for point in section.points_3d]
    for section in preview.sections:
        if section.section_type == "soma" or not section.points_3d:
            continue
        proximal_point = section.points_3d[0][:3]
        if math.dist(proximal_point, (soma_center.x_um, soma_center.y_um, soma_center.z_um)) <= attachment_threshold_um:
            fill_points.append(proximal_point)
    # Keep ordering stable while removing duplicates from overlapping section starts.
    deduped = list(dict.fromkeys((round(x, 4), round(y, 4), round(z, 4)) for x, y, z in fill_points))
    return deduped


def _site_annotation_positions(sites: list[tuple[str, MorphologySite | None]]) -> dict[str, str]:
    position_cycle = [
        "top center",
        "bottom center",
        "middle right",
        "middle left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    ]
    grouped: dict[tuple[str, float], list[str]] = {}
    for label, site in sites:
        if site is None:
            continue
        key = (site.section_name, round(site.section_x, 4))
        grouped.setdefault(key, []).append(label)

    positions: dict[str, str] = {}
    for labels in grouped.values():
        for index, label in enumerate(labels):
            positions[label] = position_cycle[index % len(position_cycle)]
    return positions


def _site_customdata(site: MorphologySite) -> list[list[object]]:
    return [[site.section_name, float(site.section_x), site.x_um, site.y_um, site.z_um]]


def build_morphology_figure(
    preview: MorphologyPreview,
    pulse_trains: list[VoltagePulseTrain],
    recording_site: MorphologySite | None,
    fi_site: MorphologySite | None,
    output_site: MorphologySite | None,
    active_recording_site: MorphologySite | None,
    incoming_connections: list[tuple[str, MorphologySite, bool]],
    selected_train_index: int | None,
    show_axes_and_coordinates: bool = False,
    soma_color: str | None = None,
) -> go.Figure:
    figure = go.Figure()
    section_lookup = {section.name: section for section in preview.sections}
    soma_sections = [section for section in preview.sections if section.section_type == "soma"]
    soma_fill_points = _soma_fill_points(preview, soma_sections)
    resolved_soma_color = soma_color or SECTION_COLORS.get("soma", "#1f2937")
    soma_line_color = _darken_hex_color(resolved_soma_color, factor=0.72)
    annotation_positions = _site_annotation_positions(
        [
            ("patch", recording_site),
            ("record", active_recording_site),
            ("fi", fi_site),
            ("output", output_site),
        ]
    )

    type_paths: dict[str, tuple[list[float], list[float], list[float]]] = {}
    for section in preview.sections:
        xs, ys, zs = type_paths.setdefault(section.section_type, ([], [], []))
        for point in section.points_3d:
            xs.append(point[0])
            ys.append(point[1])
            zs.append(point[2])
        xs.append(None)
        ys.append(None)
        zs.append(None)

    for section_type, (xs, ys, zs) in type_paths.items():
        line_color = soma_line_color if section_type == "soma" else SECTION_COLORS.get(section_type, "#64748b")
        figure.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                hoverinfo="skip",
                line={"color": line_color, "width": 5 if section_type == "soma" else 3},
                showlegend=False,
            )
        )

    if len(soma_fill_points) >= 4:
        soma_click_site = interpolate_section_site(soma_sections[0], 0.5) if soma_sections else MorphologySite()
        figure.add_trace(
            go.Mesh3d(
                x=[point[0] for point in soma_fill_points],
                y=[point[1] for point in soma_fill_points],
                z=[point[2] for point in soma_fill_points],
                color=resolved_soma_color,
                opacity=1.0,
                alphahull=0,
                customdata=_site_customdata(soma_click_site),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    marker_x: list[float] = []
    marker_y: list[float] = []
    marker_z: list[float] = []
    marker_color: list[str] = []
    customdata: list[list[object]] = []
    for section in preview.sections:
        for site in _sample_section_points(section):
            marker_x.append(site.x_um)
            marker_y.append(site.y_um)
            marker_z.append(site.z_um)
            marker_color.append(SECTION_COLORS.get(section.section_type, "#94a3b8"))
            customdata.append([site.section_name, float(site.section_x), site.x_um, site.y_um, site.z_um])

    figure.add_trace(
        go.Scatter3d(
            x=marker_x,
            y=marker_y,
            z=marker_z,
            mode="markers",
            customdata=customdata,
            hovertemplate=(
                "Section=%{customdata[0]}<br>x=%{customdata[1]:.2f}"
                "<br>X=%{customdata[2]:.1f} um<br>Y=%{customdata[3]:.1f} um<br>Z=%{customdata[4]:.1f} um<extra></extra>"
                if show_axes_and_coordinates
                else None
            ),
            # Keep sampled morphology markers clickable even when axes/coordinates are hidden.
            # `hoverinfo="skip"` suppresses Plotly click events, so use `none` instead.
            hoverinfo="none" if not show_axes_and_coordinates else None,
            marker={
                "size": 4 if show_axes_and_coordinates else 5,
                "opacity": 0.2 if show_axes_and_coordinates else 0.12,
                "color": marker_color,
            },
            showlegend=False,
        )
    )

    if recording_site is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[recording_site.x_um],
                y=[recording_site.y_um],
                z=[recording_site.z_um],
                mode="markers+text",
                customdata=_site_customdata(recording_site),
                text=["Patch"],
                textposition=annotation_positions.get("patch", "top center"),
                hoverinfo="none",
                marker={"size": 8, "color": "#dc2626"},
                showlegend=False,
            )
        )
    if active_recording_site is not None and not _site_matches(active_recording_site, recording_site):
        figure.add_trace(
            go.Scatter3d(
                x=[active_recording_site.x_um],
                y=[active_recording_site.y_um],
                z=[active_recording_site.z_um],
                mode="markers+text",
                customdata=_site_customdata(active_recording_site),
                text=["Record"],
                textposition=annotation_positions.get("record", "top center"),
                hoverinfo="none",
                marker={"size": 9, "color": "#111827", "symbol": "diamond"},
                showlegend=False,
            )
        )
    elif active_recording_site is not None and _site_matches(active_recording_site, recording_site):
        figure.add_trace(
            go.Scatter3d(
                x=[active_recording_site.x_um],
                y=[active_recording_site.y_um],
                z=[active_recording_site.z_um],
                mode="markers+text",
                customdata=_site_customdata(active_recording_site),
                text=["Record"],
                textposition=annotation_positions.get("record", "bottom center"),
                hoverinfo="none",
                marker={"size": 5, "color": "#111827"},
                showlegend=False,
            )
        )
    if fi_site is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[fi_site.x_um],
                y=[fi_site.y_um],
                z=[fi_site.z_um],
                mode="markers+text",
                customdata=_site_customdata(fi_site),
                text=["F-I"],
                textposition=annotation_positions.get("fi", "top center"),
                hoverinfo="none",
                marker={"size": 8, "color": "#7c3aed"},
                showlegend=False,
            )
        )
    if output_site is not None:
        figure.add_trace(
            go.Scatter3d(
                x=[output_site.x_um],
                y=[output_site.y_um],
                z=[output_site.z_um],
                mode="markers+text",
                customdata=_site_customdata(output_site),
                text=["Output"],
                textposition=annotation_positions.get("output", "top center"),
                hoverinfo="none",
                marker={"size": 8, "color": "#0ea5e9"},
                showlegend=False,
            )
        )

    if pulse_trains:
        train_sites = []
        for train in pulse_trains:
            section = section_lookup.get(train.section_name)
            train_sites.append(interpolate_section_site(section, train.section_x) if section else MorphologySite(train.section_name, train.section_x))
        figure.add_trace(
            go.Scatter3d(
                x=[site.x_um for site in train_sites],
                y=[site.y_um for site in train_sites],
                z=[site.z_um for site in train_sites],
                mode="markers+text",
                customdata=[
                    [site.section_name, float(site.section_x), site.x_um, site.y_um, site.z_um]
                    for site in train_sites
                ],
                text=[str(index + 1) for index in range(len(train_sites))],
                textposition="top center",
                hoverinfo="none",
                marker={
                    "size": [10 if selected_train_index == index else 7 for index in range(len(train_sites))],
                    "color": "#f59e0b",
                },
                showlegend=False,
            )
        )

    if incoming_connections:
        figure.add_trace(
            go.Scatter3d(
                x=[site.x_um for _, site, _ in incoming_connections],
                y=[site.y_um for _, site, _ in incoming_connections],
                z=[site.z_um for _, site, _ in incoming_connections],
                mode="markers+text",
                customdata=[
                    [site.section_name, float(site.section_x), site.x_um, site.y_um, site.z_um]
                    for _, site, _ in incoming_connections
                ],
                text=[label for label, _, _ in incoming_connections],
                textposition="bottom center",
                hoverinfo="none",
                marker={
                    "size": [10 if is_selected else 7 for _, _, is_selected in incoming_connections],
                    "color": ["#f97316" if is_selected else "#14b8a6" for _, _, is_selected in incoming_connections],
                },
                showlegend=False,
            )
        )

    figure.update_layout(
        template="plotly_white",
        height=640,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        scene={
            "aspectmode": "data",
            "dragmode": "turntable",
            "bgcolor": "#ffffff",
            "xaxis": {
                "title": "X (um)" if show_axes_and_coordinates else "",
                "visible": show_axes_and_coordinates,
                "showbackground": False,
                "showgrid": show_axes_and_coordinates,
                "showticklabels": show_axes_and_coordinates,
                "zeroline": False,
            },
            "yaxis": {
                "title": "Y (um)" if show_axes_and_coordinates else "",
                "visible": show_axes_and_coordinates,
                "showbackground": False,
                "showgrid": show_axes_and_coordinates,
                "showticklabels": show_axes_and_coordinates,
                "zeroline": False,
            },
            "zaxis": {
                "title": "Z (um)" if show_axes_and_coordinates else "",
                "visible": show_axes_and_coordinates,
                "showbackground": False,
                "showgrid": show_axes_and_coordinates,
                "showticklabels": show_axes_and_coordinates,
                "zeroline": False,
            },
        },
        showlegend=False,
    )
    return figure


def _site_from_click(click_data: dict | None) -> MorphologySite | None:
    if not click_data or not click_data.get("points"):
        return None
    point = click_data["points"][0]
    custom = point.get("customdata")
    if not custom or len(custom) < 5:
        return None
    return MorphologySite(
        section_name=str(custom[0]),
        section_x=float(custom[1]),
        x_um=float(custom[2]),
        y_um=float(custom[3]),
        z_um=float(custom[4]),
    )


def _build_circuit_elements(project: CircuitProject, selected_neuron_id: str | None, selected_connection_id: str | None) -> list[dict]:
    elements: list[dict] = []
    for neuron in project.neurons:
        classes = "selected-neuron" if neuron.id == selected_neuron_id else ""
        border_color = _darken_hex_color(neuron.color)
        elements.append(
            {
                "data": {
                    "id": neuron.id,
                    "label": neuron.label,
                    "color": neuron.color,
                    "borderColor": border_color,
                    "labelColor": _contrast_text_color(neuron.color),
                },
                "position": {"x": neuron.x, "y": neuron.y},
                "classes": classes,
            }
        )
    for connection in project.connections:
        classes = "selected-connection" if connection.id == selected_connection_id else ""
        elements.append(
            {
                "data": {
                    "id": connection.id,
                    "source": connection.source_id,
                    "target": connection.target_id,
                    "label": connection.label,
                },
                "classes": classes,
            }
        )
    return elements


def _circuit_stylesheet() -> list[dict]:
    return [
        {
            "selector": "node",
            "style": {
                "background-color": "data(color)",
                "border-color": "data(borderColor)",
                "label": "data(label)",
                "text-valign": "center",
                "text-halign": "center",
                "text-wrap": "wrap",
                "text-max-width": "46px",
                "color": "data(labelColor)",
                "font-size": "9px",
                "font-weight": 600,
                "width": "36px",
                "height": "36px",
                "border-width": 1.25,
            },
        },
        {
            "selector": "edge",
            "style": {
                "curve-style": "straight",
                "line-color": "#64748b",
                "target-arrow-shape": "triangle",
                "target-arrow-color": "#64748b",
                "width": 2,
            },
        },
        {
            "selector": ".selected-neuron",
            "style": {"border-width": 2.5},
        },
        {
            "selector": ".selected-connection",
            "style": {"line-color": "#dc2626", "target-arrow-color": "#dc2626", "width": 4},
        },
    ]


def _empty_current_result():
    empty = go.Figure()
    empty.update_layout(template="plotly_white", height=240, margin={"l": 60, "r": 18, "t": 42, "b": 58})
    return {
        "voltage": empty,
        "applied_trace": empty,
        "command_trace": empty,
        "ionic_currents": empty,
        "gating": empty,
        "conductances": empty,
    }


def _empty_fi_figure():
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        height=420,
        showlegend=False,
        xaxis={"title": "Injected current", "autorange": True, "fixedrange": False},
        yaxis={"title": "Spikes", "autorange": True, "fixedrange": False},
        margin={"l": 60, "r": 18, "t": 42, "b": 58},
    )
    return figure


def _figure_payload(figure):
    return figure.to_dict() if hasattr(figure, "to_dict") else figure


def _figure_map_payload(figures: dict[str, object], fallback: dict[str, object]) -> dict[str, object]:
    return {
        key: _figure_payload(figures.get(key, fallback[key]))
        for key in fallback
    }


def _clamp_panel_style(panel_key: str, selected_panels: set[str] | None) -> dict[str, str]:
    visible = panel_key == "voltage" or (selected_panels is not None and panel_key in selected_panels)
    return {"display": "block" if visible else "none"}


def _build_neuron_from_values(values: dict[str, object], unit: str) -> NeuronConfig:
    template = default_setup(CURRENT_CLAMP)[0]
    return NeuronConfig(
        duration_ms=_coerce_float(values.get("duration_ms"), template.duration_ms),
        dt_ms=_coerce_float(values.get("dt_ms"), template.dt_ms),
        v_rest_mV=_coerce_float(values.get("v_rest_mV"), template.v_rest_mV),
        holding_mV=_coerce_float(values.get("holding_mV"), template.holding_mV),
        current_injection_unit=unit,
        holding_current=_coerce_float(values.get("holding_current"), template.holding_current),
        cm_uF_cm2=_coerce_float(values.get("cm_uF_cm2"), template.cm_uF_cm2),
        gna_mS_cm2=_coerce_float(values.get("gna_mS_cm2"), template.gna_mS_cm2),
        gk_mS_cm2=_coerce_float(values.get("gk_mS_cm2"), template.gk_mS_cm2),
        gl_mS_cm2=_coerce_float(values.get("gl_mS_cm2"), template.gl_mS_cm2),
        ena_mV=_coerce_float(values.get("ena_mV"), template.ena_mV),
        ek_mV=_coerce_float(values.get("ek_mV"), template.ek_mV),
        eleak_mV=_coerce_float(values.get("eleak_mV"), template.eleak_mV),
        ecl_mV=_coerce_float(values.get("ecl_mV"), template.ecl_mV),
        gcl_mS_cm2=_coerce_float(values.get("gcl_mS_cm2"), template.gcl_mS_cm2),
        gshunt_mS_cm2=_coerce_float(values.get("gshunt_mS_cm2"), template.gshunt_mS_cm2),
    )


def _build_fi_from_values(values: dict[str, object], fallback: FISweepConfig | None = None) -> FISweepConfig:
    template = fallback or FISweepConfig()
    return FISweepConfig(
        start_current=_coerce_float(values.get("start_current"), template.start_current),
        end_current=_coerce_float(values.get("end_current"), template.end_current),
        step_current=_coerce_float(values.get("step_current"), template.step_current),
        pulse_start_ms=_coerce_float(values.get("pulse_start_ms"), template.pulse_start_ms),
        pulse_width_ms=_coerce_float(values.get("pulse_width_ms"), template.pulse_width_ms),
    )


def _make_sim_neuron_spec(neuron) -> CircuitNeuronSpec:
    return CircuitNeuronSpec(
        neuron_id=neuron.id,
        label=neuron.label,
        morphology_name=neuron.morphology_name,
        neuron=neuron.neuron_config,
        pulse_trains=list(neuron.pulse_trains),
        recording_site=neuron.recording_site,
        fi_site=neuron.fi_site,
        output_site=neuron.output_site,
    )


def _make_sim_connection_spec(connection) -> CircuitConnectionSpec:
    return CircuitConnectionSpec(
        connection_id=connection.id,
        source_id=connection.source_id,
        target_id=connection.target_id,
        target_site=connection.target_site,
        current_nA=connection.current_nA,
        pulse_width_ms=connection.pulse_width_ms,
        delay_ms=connection.delay_ms,
    )


def _connection_target_options(project: CircuitProject, selected_neuron_id: str | None) -> list[dict]:
    return [
        {"label": neuron.label, "value": neuron.id}
        for neuron in project.neurons
        if neuron.id != selected_neuron_id
    ]


def _selected_connection_text(project: CircuitProject, selected_connection_id: str | None) -> str:
    connection = _selected_connection(project, selected_connection_id)
    if connection is None:
        return "No incoming connection selected"
    source = project.neuron_by_id(connection.source_id)
    target = project.neuron_by_id(connection.target_id)
    source_label = source.label if source else connection.source_id
    target_label = target.label if target else connection.target_id
    return f"{source_label} -> {target_label} @ {_format_site(connection.target_site)}"


def _selected_train_text(neuron, selected_rows: list[int] | None) -> str:
    if neuron is None or not neuron.pulse_trains:
        return "No pulse trains"
    if not selected_rows:
        return "No pulse train selected"
    index = selected_rows[0]
    if not (0 <= index < len(neuron.pulse_trains)):
        return "No pulse train selected"
    train = neuron.pulse_trains[index]
    return f"{train.label} -> {train.section_name} @ {train.section_x:.2f}"


def _resolve_recording_site(
    project: CircuitProject,
    neuron,
    selected_connection_id: str | None,
    train_selected_rows: list[int] | None,
) -> tuple[MorphologySite, str, str]:
    default_site = default_recording_site(neuron.morphology_name)
    mode = neuron.recording_source_mode or "patch"
    if mode == "patch":
        if neuron.recording_site is not None:
            site = _materialize_site(neuron.recording_site, neuron.morphology_name)
            return site, "Dedicated patch", _format_site(site)
        return default_site, "Dedicated patch", f"{_format_site(default_site)} (soma fallback)"
    if mode == "train":
        if train_selected_rows:
            index = train_selected_rows[0]
            if 0 <= index < len(neuron.pulse_trains):
                train = neuron.pulse_trains[index]
                site = _materialize_site(
                    MorphologySite(section_name=train.section_name, section_x=train.section_x),
                    neuron.morphology_name,
                )
                return site, f"Stimulus input: {train.label}", f"{train.section_name} @ {train.section_x:.2f}"
        return default_site, "Stimulus input", f"{_format_site(default_site)} (no train selected)"
    if mode == "connection":
        connection = _selected_connection(project, selected_connection_id)
        if connection is not None and connection.target_id == neuron.id:
            site = _materialize_site(connection.target_site, neuron.morphology_name)
            return site, f"Incoming connection: {connection.label}", _format_site(site)
        return default_site, "Incoming connection", f"{_format_site(default_site)} (no connection selected)"
    if mode == "output":
        if neuron.output_site is not None:
            site = _materialize_site(neuron.output_site, neuron.morphology_name)
            return site, "Output site", _format_site(site)
        return default_site, "Output site", f"{_format_site(default_site)} (soma fallback)"
    return default_site, "Dedicated patch", _format_site(default_site)


def _morphology_options() -> list[dict]:
    return [{"label": name, "value": name} for name in list_available_swc_files()]


def _incoming_connection_options(project: CircuitProject, neuron_id: str) -> list[dict]:
    options: list[dict] = []
    for connection in project.connections:
        if connection.target_id != neuron_id:
            continue
        source = project.neuron_by_id(connection.source_id)
        source_label = source.label if source else connection.source_id
        options.append(
            {
                "label": f"From {source_label} ({connection.id})",
                "value": connection.id,
            }
        )
    return options


def _train_columns_for_unit(unit: str) -> list[dict]:
    columns = [dict(column) for column in TRAIN_COLUMNS]
    columns[5]["name"] = f"Amplitude ({unit})"
    return columns


def _current_display_label(unit: str) -> str:
    return "Holding current (nA)" if unit == CURRENT_UNIT_NA else "Holding current density (uA/cm2)"


def _fi_field_label(key: str, unit: str) -> str:
    base = {
        "start_current": "Start current",
        "end_current": "End current",
        "step_current": "Current increment",
        "pulse_start_ms": "Pulse start (ms)",
        "pulse_width_ms": "Pulse width (ms)",
    }[key]
    if key in {"start_current", "end_current", "step_current"}:
        return f"{base} ({unit})"
    return base


def _convert_value_between_units(
    value: float,
    old_unit: str,
    new_unit: str,
    site: MorphologySite,
    morphology_name: str,
) -> float:
    area_cm2 = estimate_site_segment_area_cm2(site, morphology_name)
    current_nA = current_value_to_nA(value, old_unit, area_cm2)
    return current_nA_to_value(current_nA, new_unit, area_cm2)


def _convert_neuron_current_unit(neuron, new_unit: str) -> None:
    old_unit = neuron.neuron_config.current_injection_unit
    if old_unit == new_unit:
        return
    morphology_name = neuron.morphology_name
    holding_site = neuron.recording_site or default_recording_site(morphology_name)
    fi_site = neuron.fi_site or default_recording_site(morphology_name)
    neuron.neuron_config.holding_current = _convert_value_between_units(
        neuron.neuron_config.holding_current,
        old_unit,
        new_unit,
        holding_site,
        morphology_name,
    )
    for train in neuron.pulse_trains:
        stim_site = MorphologySite(section_name=train.section_name, section_x=train.section_x)
        train.amplitude = _convert_value_between_units(
            train.amplitude,
            old_unit,
            new_unit,
            stim_site,
            morphology_name,
        )
    neuron.fi_config.start_current = _convert_value_between_units(
        neuron.fi_config.start_current,
        old_unit,
        new_unit,
        fi_site,
        morphology_name,
    )
    neuron.fi_config.end_current = _convert_value_between_units(
        neuron.fi_config.end_current,
        old_unit,
        new_unit,
        fi_site,
        morphology_name,
    )
    neuron.fi_config.step_current = _convert_value_between_units(
        neuron.fi_config.step_current,
        old_unit,
        new_unit,
        fi_site,
        morphology_name,
    )
    neuron.neuron_config.current_injection_unit = new_unit


def _reset_sites_for_morphology(neuron, project: CircuitProject) -> None:
    default_site = default_recording_site(neuron.morphology_name)
    neuron.recording_site = None
    neuron.fi_site = None
    neuron.output_site = None
    for train in neuron.pulse_trains:
        train.section_name = default_site.section_name
        train.section_x = default_site.section_x
    for connection in project.connections:
        if connection.target_id != neuron.id:
            continue
        connection.target_site = MorphologySite(
            section_name=default_site.section_name,
            section_x=default_site.section_x,
            x_um=default_site.x_um,
            y_um=default_site.y_um,
            z_um=default_site.z_um,
        )


def _incoming_connection_sites(project: CircuitProject, neuron_id: str, selected_connection_id: str | None) -> list[tuple[str, MorphologySite, bool]]:
    incoming: list[tuple[str, MorphologySite, bool]] = []
    for connection in project.connections:
        if connection.target_id != neuron_id:
            continue
        source = project.neuron_by_id(connection.source_id)
        source_label = source.label if source else connection.source_id
        incoming.append((source_label, connection.target_site, connection.id == selected_connection_id))
    return incoming


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_missing_dependency_app() -> Dash:
    app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], title="")
    app.layout = dbc.Container(
        fluid=True,
        className="container-fluid dash-shell",
        children=[
            dbc.Row(
                dbc.Col(
                    dbc.Alert(
                        [
                            html.H4("Missing dependency"),
                            html.P("`dash-cytoscape` is required for the circuit editor UI."),
                            html.P("Run `python3 bootstrap.py` again so the environment installs the updated requirements."),
                        ],
                        color="warning",
                        className="mt-4",
                    ),
                    lg=8,
                ),
                justify="center",
            )
        ],
    )
    return app


def create_app() -> Dash:
    if cyto is None:
        return _build_missing_dependency_app()

    project = CircuitProject.default()
    selected_neuron = project.neurons[0]
    preview = load_morphology_preview(selected_neuron.morphology_name)
    initial_recording_site, _, _ = _resolve_recording_site(project, selected_neuron, None, [])
    initial_clamp = simulate_current_clamp(
        selected_neuron.neuron_config,
        list(selected_neuron.pulse_trains),
        recording_site=initial_recording_site,
        selected_panels=set(PLOT_PANEL_DEFAULT_SELECTION),
        morphology_name=selected_neuron.morphology_name,
    )
    initial_clamp_figures = build_trace_panel_figures(initial_clamp, set(PLOT_PANEL_DEFAULT_SELECTION), theme="light")
    empty_trace_figures = _empty_current_result()
    empty_trace_payload = _figure_map_payload(empty_trace_figures, empty_trace_figures)
    empty_fi_payload = _figure_payload(_empty_fi_figure())

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.LUX],
        suppress_callback_exceptions=False,
        title="",
    )

    neuron_controls = [
        _number_field("neuron", key, label, getattr(selected_neuron.neuron_config, key), minimum, maximum, step)
        for key, label, minimum, maximum, step in NEURON_FIELDS
    ]
    fi_controls = [
        _number_field("fi", key, label, getattr(selected_neuron.fi_config, key), minimum, maximum, step)
        for key, label, minimum, maximum, step in FI_FIELDS
    ]

    app.layout = dbc.Container(
        fluid=True,
        className="container-fluid dash-shell",
        children=[
            dcc.Store(id="circuit-store", data=_project_to_dict(project)),
            dcc.Store(id="selected-neuron-id", data=selected_neuron.id),
            dcc.Store(id="selected-connection-id", data=None),
            dcc.Store(id="active-workspace", data="circuit"),
            dcc.Store(id="vclamp-figures-store", data=empty_trace_payload),
            dcc.Store(id="fi-figure-store", data=empty_fi_payload),
            dbc.Row(
                [
                    dbc.Col(
                        children=[
                            html.Div(
                                [
                                    dbc.Button("Circuit Workspace", id="open-circuit-workspace", color="primary", className="me-2"),
                                    dbc.Button("Neuron Workspace", id="open-neuron-workspace", color="secondary"),
                                ],
                                className="workspace-nav",
                            ),
                            dbc.Alert(id="status-banner", color="info", className="mt-3 mb-3"),
                        ]
                    )
                ]
            ),
            html.Div(
                id="circuit-workspace",
                className="workspace-page",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                lg=3,
                                children=[
                                    dbc.Card(
                                        className="panel-card mb-3",
                                        children=[
                                            dbc.CardBody(
                                                [
                                                    html.Div("Circuit", className="panel-section-title"),
                                                    html.Div("This workspace is for motif-level wiring only.", className="panel-help-text"),
                                                    html.Label("New neuron label", className="control-label"),
                                                    dcc.Input(id="new-neuron-label", value="Neuron", type="text", className="form-control label-input"),
                                                    html.Label("Selected neuron color", className="control-label mt-2"),
                                                    dcc.Input(id="new-neuron-color", type="color", value="#2563eb", className="color-picker-input"),
                                                    html.Label("Morphology", className="control-label mt-2"),
                                                    dcc.Dropdown(id="new-neuron-morphology", options=_morphology_options(), value=selected_neuron.morphology_name, clearable=False),
                                                    dbc.Button("Add neuron", id="add-neuron", color="primary", className="w-100 mt-3"),
                                                    html.Hr(),
                                                    html.Label("Connect selected neuron to", className="control-label"),
                                                    dcc.Dropdown(id="connect-target-neuron", options=_connection_target_options(project, selected_neuron.id), value=None, clearable=True),
                                                    dbc.Button("Create connection", id="connect-neurons", color="secondary", className="w-100 mt-2"),
                                                    dbc.Button("Delete selected element", id="delete-selected-element", color="secondary", className="w-100 mt-2"),
                                                    dbc.Button("Open selected neuron", id="open-selected-neuron", color="secondary", className="w-100 mt-2"),
                                                    html.Hr(),
                                                    html.Div("Selected connection", className="site-label"),
                                                    html.Div(id="selected-connection-text", className="site-value"),
                                                ]
                                            )
                                        ],
                                    )
                                ],
                            ),
                            dbc.Col(
                                lg=9,
                                children=[
                                    dbc.Card(
                                        className="panel-card mb-3",
                                        children=[
                                            dbc.CardBody(
                                                [
                                                    html.Div("Circuit motif editor", className="panel-section-title"),
                                                    html.Div(
                                                        "Drag neuron points to rearrange the motif. Select a neuron to open its detailed workspace, or select an edge to retarget its synaptic input site there.",
                                                        className="panel-help-text",
                                                    ),
                                                    cyto.Cytoscape(
                                                        id="circuit-graph",
                                                        layout={"name": "preset"},
                                                        elements=_build_circuit_elements(project, selected_neuron.id, None),
                                                        stylesheet=_circuit_stylesheet(),
                                                        style={"width": "100%", "height": "78vh"},
                                                        minZoom=0.2,
                                                        maxZoom=4.0,
                                                        userPanningEnabled=True,
                                                        userZoomingEnabled=True,
                                                        autoungrabify=False,
                                                        autolock=False,
                                                        boxSelectionEnabled=False,
                                                    ),
                                                ]
                                            )
                                        ],
                                    )
                                ],
                            ),
                        ],
                        className="g-3 py-3 align-items-start",
                    )
                ],
            ),
            html.Div(
                id="neuron-workspace",
                className="workspace-page",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                lg=4,
                                children=[
                                    dbc.Accordion(
                                        start_collapsed=False,
                                        always_open=True,
                                        children=[
                                            dbc.AccordionItem(
                                                title="Selection & display",
                                                children=[
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Div("Assignment mode", className="site-label"),
                                                                        dbc.RadioItems(
                                                                            id="assign-mode",
                                                                            options=[
                                                                                {"label": "Selected train", "value": "train"},
                                                                                {"label": "Recording patch", "value": "record"},
                                                                                {"label": "F-I site", "value": "fi"},
                                                                                {"label": "Neuron output", "value": "output"},
                                                                                {"label": "Selected connection input", "value": "connection_target"},
                                                                            ],
                                                                            value="train",
                                                                            className="compact-option-list",
                                                                        ),
                                                                    ],
                                                                    className="control-block",
                                                                ),
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Div("Recording source", className="site-label"),
                                                                        dbc.RadioItems(
                                                                            id="recording-source-mode",
                                                                            options=RECORDING_SOURCE_OPTIONS,
                                                                            value=selected_neuron.recording_source_mode,
                                                                            className="compact-option-list",
                                                                        ),
                                                                    ],
                                                                    className="control-block",
                                                                ),
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="g-2",
                                                    ),
                                                    html.Div(
                                                        [
                                                            html.Div("Incoming connection to edit", className="site-label"),
                                                            dcc.Dropdown(
                                                                id="selected-incoming-connection",
                                                                options=[],
                                                                value=None,
                                                                clearable=True,
                                                            ),
                                                        ],
                                                        className="control-block",
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        html.Div("Visible clamp traces", className="site-label"),
                                                                        dbc.Checklist(
                                                                            id="plot-panels",
                                                                            options=[{"label": label, "value": key} for key, label in PLOT_PANEL_LABELS.items()],
                                                                            value=PLOT_PANEL_DEFAULT_SELECTION,
                                                                            className="compact-option-list",
                                                                        ),
                                                                    ],
                                                                    className="control-block",
                                                                ),
                                                                md=6,
                                                            ),
                                                            dbc.Col(
                                                                html.Div(
                                                                    [
                                                                        dbc.Switch(
                                                                            id="isolate-selected-neuron",
                                                                            label="Isolate selected neuron (ignore circuit)",
                                                                            value=False,
                                                                            className="mb-3",
                                                                        ),
                                                                        dbc.Switch(
                                                                            id="live-update",
                                                                            label="Live update clamp plot",
                                                                            value=True,
                                                                            className="mb-3",
                                                                        ),
                                                                    ],
                                                                    className="control-block",
                                                                ),
                                                                md=6,
                                                            ),
                                                        ],
                                                        className="g-2",
                                                    ),
                                                    html.Div("Dedicated patch", className="site-label"),
                                                    html.Div(id="recording-site-text", className="site-value mb-2"),
                                                    html.Div("Active recording source", className="site-label"),
                                                    html.Div(id="active-recording-source-text", className="site-value mb-2"),
                                                    html.Div("F-I site", className="site-label"),
                                                    html.Div(id="fi-site-text", className="site-value mb-2"),
                                                    html.Div("Output site", className="site-label"),
                                                    html.Div(id="output-site-text", className="site-value mb-2"),
                                                    html.Div("Selected train", className="site-label"),
                                                    html.Div(id="selected-train-text", className="site-value"),
                                                ],
                                            ),
                                            dbc.AccordionItem(
                                                title="Selected neuron",
                                                children=[
                                                    html.Div(
                                                        [
                                                            html.Label("Neuron label", className="control-label"),
                                                            dcc.Input(id="selected-neuron-label", type="text", value=selected_neuron.label, className="form-control label-input"),
                                                            html.Label("Neuron color", className="control-label mt-2"),
                                                            dcc.Input(id="selected-neuron-color", type="color", value=selected_neuron.color, className="color-picker-input"),
                                                            html.Label("Morphology", className="control-label mt-2"),
                                                            dcc.Dropdown(id="selected-neuron-morphology", options=_morphology_options(), value=selected_neuron.morphology_name, clearable=False),
                                                            html.Label("Current injection unit", className="control-label mt-2"),
                                                            dcc.Dropdown(
                                                                id="current-unit",
                                                                options=[{"label": unit, "value": unit} for unit in CURRENT_INJECTION_UNITS],
                                                                value=selected_neuron.neuron_config.current_injection_unit,
                                                                clearable=False,
                                                            ),
                                                        ],
                                                        className="control-block",
                                                    ),
                                                    html.Div(neuron_controls, className="compact-control-list"),
                                                ],
                                            ),
                                            dbc.AccordionItem(
                                                title="Pulse trains",
                                                children=[
                                                    dash_table.DataTable(
                                                        id="train-table",
                                                        data=_serialize_trains(selected_neuron.pulse_trains),
                                                        columns=_train_columns_for_unit(selected_neuron.neuron_config.current_injection_unit),
                                                        editable=True,
                                                        row_selectable="single",
                                                        selected_rows=[],
                                                        style_table={"overflowX": "auto"},
                                                        style_cell={"fontFamily": "ui-sans-serif, system-ui"},
                                                        style_header={"fontWeight": "600"},
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dbc.Button("Add train", id="add-train", color="secondary", className="w-100")),
                                                            dbc.Col(dbc.Button("Delete train", id="delete-train", color="secondary", className="w-100")),
                                                        ],
                                                        className="g-2 mt-2",
                                                    ),
                                                ],
                                            ),
                                            dbc.AccordionItem(
                                                title="Voltage clamp",
                                                children=[
                                                    html.Div(
                                                        "Voltage clamp runs on the selected neuron in isolation at the selected patch site.",
                                                        className="panel-help-text",
                                                    ),
                                                    dash_table.DataTable(
                                                        id="voltage-train-table",
                                                        data=_serialize_trains(selected_neuron.voltage_trains),
                                                        columns=VOLTAGE_TRAIN_COLUMNS,
                                                        editable=True,
                                                        row_selectable="single",
                                                        selected_rows=[],
                                                        style_table={"overflowX": "auto"},
                                                        style_cell={"fontFamily": "ui-sans-serif, system-ui"},
                                                        style_header={"fontWeight": "600"},
                                                    ),
                                                    dbc.Row(
                                                        [
                                                            dbc.Col(dbc.Button("Add voltage step", id="add-voltage-train", color="secondary", className="w-100")),
                                                            dbc.Col(dbc.Button("Delete voltage step", id="delete-voltage-train", color="secondary", className="w-100")),
                                                        ],
                                                        className="g-2 mt-2",
                                                    ),
                                                ],
                                            ),
                                            dbc.AccordionItem(
                                                title="F-I sweep",
                                                children=[html.Div(fi_controls, className="compact-control-list")],
                                            ),
                                            dbc.AccordionItem(
                                                title="Actions",
                                                children=[
                                                    dbc.Button("Run clamp", id="run-clamp", color="primary", className="w-100 mb-2"),
                                                    dbc.Button("Run voltage clamp", id="run-vclamp", color="secondary", className="w-100 mb-2"),
                                                    dbc.Button("Run F-I sweep", id="run-fi", color="secondary", className="w-100 mb-2"),
                                                    dbc.Button("Reset circuit", id="reset-defaults", color="secondary", className="w-100"),
                                                ],
                                            ),
                                        ],
                                    )
                                ],
                            ),
                            dbc.Col(
                                lg=8,
                                children=[
                                    dbc.Card(
                                        className="panel-card mb-3 morphology-card",
                                        children=[
                                            dbc.CardBody(
                                                [
                                                    html.Div(id="selected-neuron-heading", className="panel-section-title"),
                                                    html.Div(
                                                        "Click a sampled membrane point to place the selected train, recording patch, F-I site, neuron output, or the selected incoming connection target.",
                                                        className="panel-help-text",
                                                    ),
                                                    dbc.Switch(
                                                        id="morphology-show-axes",
                                                        label="Show axes and selection coordinates",
                                                        value=False,
                                                        className="mb-2",
                                                    ),
                                                    dcc.Graph(
                                                        id="morphology-graph",
                                                        figure=build_morphology_figure(
                                                            preview,
                                                            selected_neuron.pulse_trains,
                                                            selected_neuron.recording_site,
                                                            selected_neuron.fi_site,
                                                            selected_neuron.output_site,
                                                            initial_recording_site if selected_neuron.recording_source_mode else None,
                                                            [],
                                                            None,
                                                            False,
                                                            selected_neuron.color,
                                                        ),
                                                        config={"displayModeBar": False},
                                                        style={"height": "620px"},
                                                    ),
                                                ]
                                            )
                                        ],
                                    ),
                                    dbc.Card(
                                        className="panel-card mb-3",
                                        children=[
                                            dbc.CardBody(
                                                [
                                                    dbc.Tabs(
                                                        [
                                                            dbc.Tab(
                                                                label="Current clamp",
                                                                tab_id="tab-clamp",
                                                                children=[
                                                                    html.Div(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(dcc.Graph(id="clamp-graph-voltage", figure=initial_clamp_figures.get("voltage", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "240px"}), id="clamp-panel-voltage-container", className="clamp-panel-block"),
                                                                                    html.Div(dcc.Graph(id="clamp-graph-applied_trace", figure=initial_clamp_figures.get("applied_trace", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="clamp-panel-applied_trace-container", className="clamp-panel-block", style=_clamp_panel_style("applied_trace", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="clamp-graph-command_trace", figure=initial_clamp_figures.get("command_trace", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="clamp-panel-command_trace-container", className="clamp-panel-block", style=_clamp_panel_style("command_trace", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="clamp-graph-ionic_currents", figure=initial_clamp_figures.get("ionic_currents", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="clamp-panel-ionic_currents-container", className="clamp-panel-block", style=_clamp_panel_style("ionic_currents", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="clamp-graph-gating", figure=initial_clamp_figures.get("gating", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="clamp-panel-gating-container", className="clamp-panel-block", style=_clamp_panel_style("gating", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="clamp-graph-conductances", figure=initial_clamp_figures.get("conductances", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="clamp-panel-conductances-container", className="clamp-panel-block", style=_clamp_panel_style("conductances", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                ],
                                                                                className="clamp-panels-stack",
                                                                            )
                                                                        ],
                                                                        className="tab-panel-body",
                                                                    )
                                                                ],
                                                            ),
                                                            dbc.Tab(
                                                                label="Voltage clamp",
                                                                tab_id="tab-vclamp",
                                                                children=[
                                                                    html.Div(
                                                                        [
                                                                            html.Div(
                                                                                [
                                                                                    html.Div(dcc.Graph(id="vclamp-graph-voltage", figure=empty_trace_figures.get("voltage", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "240px"}), id="vclamp-panel-voltage-container", className="clamp-panel-block"),
                                                                                    html.Div(dcc.Graph(id="vclamp-graph-applied_trace", figure=empty_trace_figures.get("applied_trace", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="vclamp-panel-applied_trace-container", className="clamp-panel-block", style=_clamp_panel_style("applied_trace", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="vclamp-graph-command_trace", figure=empty_trace_figures.get("command_trace", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="vclamp-panel-command_trace-container", className="clamp-panel-block", style=_clamp_panel_style("command_trace", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="vclamp-graph-ionic_currents", figure=empty_trace_figures.get("ionic_currents", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="vclamp-panel-ionic_currents-container", className="clamp-panel-block", style=_clamp_panel_style("ionic_currents", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="vclamp-graph-gating", figure=empty_trace_figures.get("gating", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="vclamp-panel-gating-container", className="clamp-panel-block", style=_clamp_panel_style("gating", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                    html.Div(dcc.Graph(id="vclamp-graph-conductances", figure=empty_trace_figures.get("conductances", go.Figure()), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "220px"}), id="vclamp-panel-conductances-container", className="clamp-panel-block", style=_clamp_panel_style("conductances", set(PLOT_PANEL_DEFAULT_SELECTION))),
                                                                                ],
                                                                                className="clamp-panels-stack",
                                                                            )
                                                                        ],
                                                                        className="tab-panel-body",
                                                                    )
                                                                ],
                                                            ),
                                                            dbc.Tab(
                                                                label="F-I",
                                                                tab_id="tab-fi",
                                                                children=[
                                                                    html.Div(
                                                                        [dcc.Graph(id="fi-graph", figure=_empty_fi_figure(), config={"displayModeBar": True}, animate=True, animation_options={"transition": {"duration": 180}, "frame": {"duration": 180, "redraw": True}}, style={"height": "420px"})],
                                                                        className="tab-panel-body",
                                                                    )
                                                                ],
                                                            ),
                                                        ],
                                                        id="results-tabs",
                                                        active_tab="tab-clamp",
                                                    )
                                                ]
                                            )
                                        ],
                                    ),
                                ],
                            ),
                        ],
                        className="g-3 py-3 align-items-start",
                    )
                ],
            ),
        ],
    )

    neuron_slider_ids = [_slider_id("neuron", key) for key, *_ in NEURON_FIELDS]
    fi_slider_ids = [_slider_id("fi", key) for key, *_ in FI_FIELDS]

    def register_value_sync(prefix: str, fields: list[tuple[str, str, float, float, float]]) -> None:
        for key, *_ in fields:
            slider_id = _slider_id(prefix, key)
            display_id = _display_id(prefix, key)

            @app.callback(
                Output(display_id, "children", allow_duplicate=True),
                Input(slider_id, "value"),
                prevent_initial_call=True,
            )
            def update_display_only(slider_value):
                if slider_value is None:
                    raise PreventUpdate
                return _format_numeric_value(float(slider_value))

    register_value_sync("neuron", NEURON_FIELDS)
    register_value_sync("fi", FI_FIELDS)

    @app.callback(
        Output("active-workspace", "data"),
        Input("open-circuit-workspace", "n_clicks"),
        Input("open-neuron-workspace", "n_clicks"),
        Input("open-selected-neuron", "n_clicks"),
        prevent_initial_call=True,
    )
    def switch_workspace(circuit_clicks, neuron_clicks, selected_neuron_clicks):
        triggered = dash.ctx.triggered_id
        if triggered == "open-circuit-workspace":
            return "circuit"
        if triggered in {"open-neuron-workspace", "open-selected-neuron"}:
            return "neuron"
        raise PreventUpdate

    @app.callback(
        Output("circuit-workspace", "style"),
        Output("neuron-workspace", "style"),
        Output("open-circuit-workspace", "color"),
        Output("open-neuron-workspace", "color"),
        Input("active-workspace", "data"),
    )
    def render_active_workspace(active_workspace):
        circuit_style = {"display": "block"} if active_workspace == "circuit" else {"display": "none"}
        neuron_style = {"display": "block"} if active_workspace == "neuron" else {"display": "none"}
        circuit_color = "primary" if active_workspace == "circuit" else "secondary"
        neuron_color = "primary" if active_workspace == "neuron" else "secondary"
        return circuit_style, neuron_style, circuit_color, neuron_color

    @app.callback(
        Output("selected-connection-id", "data", allow_duplicate=True),
        Input("selected-incoming-connection", "value"),
        prevent_initial_call=True,
    )
    def select_incoming_connection(connection_id):
        if not connection_id:
            return None
        return connection_id

    @app.callback(
        Output("circuit-graph", "elements"),
        Output("connect-target-neuron", "options"),
        Output("selected-connection-text", "children"),
        Output("selected-incoming-connection", "options"),
        Output("selected-incoming-connection", "value"),
        Output("recording-source-mode", "value"),
        Output("recording-site-text", "children"),
        Output("active-recording-source-text", "children"),
        Output("fi-site-text", "children"),
        Output("output-site-text", "children"),
        Output("selected-train-text", "children"),
        Output("selected-neuron-heading", "children"),
        Output("selected-neuron-label", "value"),
        Output("selected-neuron-color", "value"),
        Output("new-neuron-color", "value"),
        Output("selected-neuron-morphology", "value"),
        Output("current-unit", "value"),
        Output("train-table", "data"),
        Output("train-table", "columns"),
        Output("voltage-train-table", "data"),
        *[Output(_slider_id("neuron", key), "value") for key, *_ in NEURON_FIELDS],
        *[Output(_slider_id("fi", key), "value") for key, *_ in FI_FIELDS],
        Output("neuron-label-holding_current", "children"),
        Output("fi-label-start_current", "children"),
        Output("fi-label-end_current", "children"),
        Output("fi-label-step_current", "children"),
        Input("circuit-store", "data"),
        Input("selected-neuron-id", "data"),
        Input("selected-connection-id", "data"),
        Input("train-table", "selected_rows"),
    )
    def sync_selected_neuron_to_controls(project_data, selected_neuron_id, selected_connection_id, train_selected_rows):
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate
        unit = neuron.neuron_config.current_injection_unit
        incoming_connection_options = _incoming_connection_options(project, neuron.id)
        valid_connection_ids = {option["value"] for option in incoming_connection_options}
        selected_incoming_connection = selected_connection_id if selected_connection_id in valid_connection_ids else None
        _, recording_source_label, recording_source_text = _resolve_recording_site(
            project,
            neuron,
            selected_incoming_connection,
            train_selected_rows,
        )
        neuron_values = [getattr(neuron.neuron_config, key) for key, *_ in NEURON_FIELDS]
        fi_values = [getattr(neuron.fi_config, key) for key, *_ in FI_FIELDS]
        return (
            _build_circuit_elements(project, neuron.id, selected_incoming_connection),
            _connection_target_options(project, neuron.id),
            _selected_connection_text(project, selected_incoming_connection),
            incoming_connection_options,
            selected_incoming_connection,
            neuron.recording_source_mode,
            _format_site(neuron.recording_site),
            f"{recording_source_label}: {recording_source_text}",
            _format_site(neuron.fi_site),
            _format_site(neuron.output_site),
            _selected_train_text(neuron, train_selected_rows),
            neuron.label,
            neuron.label,
            neuron.color,
            neuron.color,
            neuron.morphology_name,
            unit,
            _serialize_trains(neuron.pulse_trains),
            _train_columns_for_unit(unit),
            _serialize_trains(neuron.voltage_trains),
            *neuron_values,
            *fi_values,
            _current_display_label(unit),
            _fi_field_label("start_current", unit),
            _fi_field_label("end_current", unit),
            _fi_field_label("step_current", unit),
        )

    @app.callback(
        Output("morphology-graph", "figure"),
        Input("circuit-store", "data"),
        Input("selected-neuron-id", "data"),
        Input("selected-connection-id", "data"),
        Input("train-table", "selected_rows"),
        Input("morphology-show-axes", "value"),
    )
    def sync_morphology_figure(project_data, selected_neuron_id, selected_connection_id, train_selected_rows, show_axes_and_coordinates):
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate
        selected_train_index = train_selected_rows[0] if train_selected_rows else None
        active_recording_site, _, _ = _resolve_recording_site(project, neuron, selected_connection_id, train_selected_rows)
        return build_morphology_figure(
            load_morphology_preview(neuron.morphology_name),
            neuron.pulse_trains,
            neuron.recording_site,
            neuron.fi_site,
            neuron.output_site,
            active_recording_site,
            _incoming_connection_sites(project, neuron.id, selected_connection_id),
            selected_train_index,
            bool(show_axes_and_coordinates),
            neuron.color,
        )

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("current-unit", "value"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        prevent_initial_call=True,
    )
    def change_current_unit(current_unit, project_data, selected_neuron_id):
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None or not current_unit or neuron.neuron_config.current_injection_unit == current_unit:
            raise PreventUpdate
        _convert_neuron_current_unit(neuron, current_unit)
        return _project_to_dict(project), f"Converted {neuron.label} current units to {current_unit}."

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("new-neuron-color", "value"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        prevent_initial_call=True,
    )
    def update_selected_neuron_color_from_circuit(color_value, project_data, selected_neuron_id):
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None or not color_value:
            raise PreventUpdate
        color_text = str(color_value)
        if neuron.color == color_text:
            raise PreventUpdate
        neuron.color = color_text
        return _project_to_dict(project), f"Updated {neuron.label} color."

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Input("circuit-graph", "elements"),
        State("circuit-store", "data"),
        prevent_initial_call=True,
    )
    def persist_circuit_positions(elements, project_data):
        project = _project_from_dict(project_data)
        changed = False
        position_by_id = {
            item["data"]["id"]: item.get("position", {})
            for item in elements or []
            if item.get("data", {}).get("id") in {neuron.id for neuron in project.neurons}
        }
        for neuron in project.neurons:
            position = position_by_id.get(neuron.id)
            if not position:
                continue
            x_value = float(position.get("x", neuron.x))
            y_value = float(position.get("y", neuron.y))
            if abs(neuron.x - x_value) > 1e-6 or abs(neuron.y - y_value) > 1e-6:
                neuron.x = x_value
                neuron.y = y_value
                changed = True
        if not changed:
            raise PreventUpdate
        return _project_to_dict(project)

    @app.callback(
        Output("selected-neuron-id", "data", allow_duplicate=True),
        Output("selected-connection-id", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("circuit-graph", "tapNodeData"),
        Input("circuit-graph", "tapEdgeData"),
        State("circuit-store", "data"),
        prevent_initial_call=True,
    )
    def select_circuit_element(tap_node_data, tap_edge_data, project_data):
        project = _project_from_dict(project_data)
        triggered_prop = dash.ctx.triggered[0]["prop_id"] if dash.ctx.triggered else ""
        if triggered_prop == "circuit-graph.tapNodeData" and tap_node_data:
            neuron = project.neuron_by_id(str(tap_node_data["id"]))
            if neuron is None:
                raise PreventUpdate
            return neuron.id, None, f"Selected {neuron.label}."
        if triggered_prop == "circuit-graph.tapEdgeData" and tap_edge_data:
            connection = project.connection_by_id(str(tap_edge_data["id"]))
            if connection is None:
                raise PreventUpdate
            return connection.target_id, connection.id, f"Selected {_selected_connection_text(project, connection.id)}."
        raise PreventUpdate

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Output("selected-neuron-id", "data", allow_duplicate=True),
        Output("selected-connection-id", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("add-neuron", "n_clicks"),
        Input("connect-neurons", "n_clicks"),
        Input("delete-selected-element", "n_clicks"),
        Input("reset-defaults", "n_clicks"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        State("selected-connection-id", "data"),
        State("new-neuron-label", "value"),
        State("new-neuron-color", "value"),
        State("new-neuron-morphology", "value"),
        State("connect-target-neuron", "value"),
        prevent_initial_call=True,
    )
    def mutate_circuit(add_clicks, connect_clicks, delete_clicks, reset_clicks, project_data, selected_neuron_id, selected_connection_id, new_label, new_color, new_morphology, connect_target):
        triggered = dash.ctx.triggered_id
        if triggered == "reset-defaults":
            project = CircuitProject.default()
            return _project_to_dict(project), project.neurons[0].id, None, "Reset circuit to the default single-neuron motif."

        project = _project_from_dict(project_data)
        selected_neuron = _selected_neuron(project, selected_neuron_id)
        if triggered == "add-neuron":
            index = len(project.neurons)
            neuron = project.add_neuron(
                label=str(new_label or f"Neuron {index + 1}"),
                color=str(new_color or "#2563eb"),
                x=120.0 + index * 80.0,
                y=120.0 + (index % 2) * 80.0,
                morphology_name=str(new_morphology or project.neurons[0].morphology_name),
            )
            return _project_to_dict(project), neuron.id, None, f"Added {neuron.label}."
        if triggered == "connect-neurons":
            if selected_neuron is None or not connect_target:
                raise PreventUpdate
            connection = project.add_connection(selected_neuron.id, str(connect_target))
            return _project_to_dict(project), connection.target_id, connection.id, f"Created {_selected_connection_text(project, connection.id)}."
        if triggered == "delete-selected-element":
            if selected_connection_id:
                project.remove_connection(selected_connection_id)
                return _project_to_dict(project), selected_neuron_id, None, "Deleted selected connection."
            if selected_neuron is None or len(project.neurons) == 1:
                return no_update, no_update, no_update, "Keep at least one neuron in the circuit."
            deleted_label = selected_neuron.label
            project.remove_neuron(selected_neuron.id)
            next_neuron = project.neurons[0]
            return _project_to_dict(project), next_neuron.id, None, f"Deleted {deleted_label}."
        raise PreventUpdate

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("selected-neuron-label", "value"),
        Input("selected-neuron-color", "value"),
        Input("selected-neuron-morphology", "value"),
        Input("recording-source-mode", "value"),
        Input("train-table", "data"),
        Input("voltage-train-table", "data"),
        *[Input(slider_id, "value") for slider_id in neuron_slider_ids],
        *[Input(slider_id, "value") for slider_id in fi_slider_ids],
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        prevent_initial_call=True,
    )
    def persist_selected_neuron(label_value, color_value, morphology_value, recording_source_mode, train_rows, voltage_train_rows, *values):
        project_data = values[-2]
        neuron_id = values[-1]
        neuron_value_list = values[: len(NEURON_FIELDS)]
        fi_value_list = values[len(NEURON_FIELDS) : len(NEURON_FIELDS) + len(FI_FIELDS)]
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, neuron_id)
        if neuron is None:
            raise PreventUpdate

        before = _project_to_dict(project)
        neuron.label = str(label_value or neuron.label)
        neuron.color = str(color_value or neuron.color)
        neuron.recording_source_mode = str(recording_source_mode or "patch")
        morphology_changed = bool(morphology_value and morphology_value != neuron.morphology_name)
        if morphology_changed:
            neuron.morphology_name = str(morphology_value)
            _reset_sites_for_morphology(neuron, project)

        neuron_values = {key: neuron_value_list[index] for index, (key, *_rest) in enumerate(NEURON_FIELDS)}
        fi_values = {key: fi_value_list[index] for index, (key, *_rest) in enumerate(FI_FIELDS)}
        current_unit = neuron.neuron_config.current_injection_unit
        neuron.neuron_config = _build_neuron_from_values(neuron_values, current_unit)
        neuron.fi_config = _build_fi_from_values(fi_values, neuron.fi_config)
        neuron.pulse_trains = _deserialize_trains(train_rows or [])
        neuron.voltage_trains = _deserialize_trains(voltage_train_rows or [])

        if morphology_changed:
            default_site = default_recording_site(neuron.morphology_name)
            for train in neuron.pulse_trains:
                train.section_name = default_site.section_name
                train.section_x = default_site.section_x

        after = _project_to_dict(project)
        if after == before:
            raise PreventUpdate
        return after, f"Updated {neuron.label}."

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Output("train-table", "selected_rows", allow_duplicate=True),
        Output("voltage-train-table", "selected_rows", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("add-train", "n_clicks"),
        Input("delete-train", "n_clicks"),
        Input("add-voltage-train", "n_clicks"),
        Input("delete-voltage-train", "n_clicks"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        State("train-table", "selected_rows"),
        State("voltage-train-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def mutate_train_tables(add_train_clicks, delete_train_clicks, add_voltage_clicks, delete_voltage_clicks, project_data, selected_neuron_id, train_selected_rows, voltage_selected_rows):
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate

        triggered = dash.ctx.triggered_id
        if triggered == "add-train":
            _, default_trains = default_setup(CURRENT_CLAMP, neuron.morphology_name)
            train = default_trains[0]
            train.label = f"Stim {len(neuron.pulse_trains) + 1}"
            stim_site = neuron.recording_site or default_recording_site(neuron.morphology_name)
            train.section_name = stim_site.section_name
            train.section_x = stim_site.section_x
            if neuron.neuron_config.current_injection_unit != CURRENT_UNIT_NA:
                area_cm2 = estimate_site_segment_area_cm2(stim_site, neuron.morphology_name)
                train.amplitude = current_nA_to_value(train.amplitude, neuron.neuron_config.current_injection_unit, area_cm2)
            neuron.pulse_trains.append(train)
            return _project_to_dict(project), [len(neuron.pulse_trains) - 1], (voltage_selected_rows or []), f"Added pulse train to {neuron.label}."
        if triggered == "delete-train":
            if not neuron.pulse_trains:
                raise PreventUpdate
            index = train_selected_rows[0] if train_selected_rows else len(neuron.pulse_trains) - 1
            index = max(0, min(index, len(neuron.pulse_trains) - 1))
            removed = neuron.pulse_trains.pop(index)
            next_selection = [max(0, min(index, len(neuron.pulse_trains) - 1))] if neuron.pulse_trains else []
            return _project_to_dict(project), next_selection, (voltage_selected_rows or []), f"Deleted {removed.label}."
        if triggered == "add-voltage-train":
            _, default_trains = default_setup(VOLTAGE_CLAMP, neuron.morphology_name)
            train = default_trains[0]
            train.label = f"V-step {len(neuron.voltage_trains) + 1}"
            neuron.voltage_trains.append(train)
            return _project_to_dict(project), (train_selected_rows or []), [len(neuron.voltage_trains) - 1], f"Added voltage step to {neuron.label}."
        if triggered == "delete-voltage-train":
            if not neuron.voltage_trains:
                raise PreventUpdate
            index = voltage_selected_rows[0] if voltage_selected_rows else len(neuron.voltage_trains) - 1
            index = max(0, min(index, len(neuron.voltage_trains) - 1))
            removed = neuron.voltage_trains.pop(index)
            next_selection = [max(0, min(index, len(neuron.voltage_trains) - 1))] if neuron.voltage_trains else []
            return _project_to_dict(project), (train_selected_rows or []), next_selection, f"Deleted {removed.label}."
        raise PreventUpdate

    @app.callback(
        Output("circuit-store", "data", allow_duplicate=True),
        Output("morphology-graph", "clickData", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("morphology-graph", "clickData"),
        State("assign-mode", "value"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        State("selected-connection-id", "data"),
        State("train-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def assign_morphology_site(click_data, assign_mode, project_data, selected_neuron_id, selected_connection_id, train_selected_rows):
        site = _site_from_click(click_data)
        if site is None:
            raise PreventUpdate
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate
        if assign_mode == "train":
            if not train_selected_rows:
                return no_update, None, "Select a pulse train row first."
            index = train_selected_rows[0]
            if not (0 <= index < len(neuron.pulse_trains)):
                return no_update, None, "Select a valid pulse train row first."
            neuron.pulse_trains[index].section_name = site.section_name
            neuron.pulse_trains[index].section_x = site.section_x
            return _project_to_dict(project), None, f"Assigned {neuron.pulse_trains[index].label} to {_format_site(site)}."
        if assign_mode == "record":
            neuron.recording_site = site
            return _project_to_dict(project), None, f"Recording patch moved to {_format_site(site)}."
        if assign_mode == "fi":
            neuron.fi_site = site
            return _project_to_dict(project), None, f"F-I site moved to {_format_site(site)}."
        if assign_mode == "output":
            neuron.output_site = site
            return _project_to_dict(project), None, f"Output site moved to {_format_site(site)}."
        if assign_mode == "connection_target":
            connection = _selected_connection(project, selected_connection_id)
            if connection is None:
                return no_update, None, "Select a connection first."
            connection.target_site = site
            return _project_to_dict(project), None, f"Connection target moved to {_format_site(site)}."
        raise PreventUpdate

    @app.callback(
        Output("clamp-panel-applied_trace-container", "style"),
        Output("clamp-panel-command_trace-container", "style"),
        Output("clamp-panel-ionic_currents-container", "style"),
        Output("clamp-panel-gating-container", "style"),
        Output("clamp-panel-conductances-container", "style"),
        Output("vclamp-panel-applied_trace-container", "style"),
        Output("vclamp-panel-command_trace-container", "style"),
        Output("vclamp-panel-ionic_currents-container", "style"),
        Output("vclamp-panel-gating-container", "style"),
        Output("vclamp-panel-conductances-container", "style"),
        Input("plot-panels", "value"),
    )
    def toggle_panel_visibility(plot_panels):
        selected_panels = set(plot_panels or [])
        return (
            _clamp_panel_style("applied_trace", selected_panels),
            _clamp_panel_style("command_trace", selected_panels),
            _clamp_panel_style("ionic_currents", selected_panels),
            _clamp_panel_style("gating", selected_panels),
            _clamp_panel_style("conductances", selected_panels),
            _clamp_panel_style("applied_trace", selected_panels),
            _clamp_panel_style("command_trace", selected_panels),
            _clamp_panel_style("ionic_currents", selected_panels),
            _clamp_panel_style("gating", selected_panels),
            _clamp_panel_style("conductances", selected_panels),
        )

    @app.callback(
        Output("clamp-graph-voltage", "figure"),
        Output("clamp-graph-applied_trace", "figure"),
        Output("clamp-graph-command_trace", "figure"),
        Output("clamp-graph-ionic_currents", "figure"),
        Output("clamp-graph-gating", "figure"),
        Output("clamp-graph-conductances", "figure"),
        Output("results-tabs", "active_tab", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("run-clamp", "n_clicks"),
        Input("circuit-store", "data"),
        Input("selected-neuron-id", "data"),
        Input("isolate-selected-neuron", "value"),
        Input("plot-panels", "value"),
        Input("live-update", "value"),
        State("selected-connection-id", "data"),
        State("train-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def update_current_clamp(run_clicks, project_data, selected_neuron_id, isolate_selected_neuron, plot_panels, live_update, selected_connection_id, train_selected_rows):
        if dash.ctx.triggered_id != "run-clamp" and not live_update:
            raise PreventUpdate
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate
        selected_panels = set(plot_panels or PLOT_PANEL_DEFAULT_SELECTION)
        effective_recording_site, recording_source_label, recording_source_text = _resolve_recording_site(
            project,
            neuron,
            selected_connection_id,
            train_selected_rows,
        )
        start_time = time.perf_counter()
        if isolate_selected_neuron:
            result = simulate_current_clamp(
                neuron.neuron_config,
                list(neuron.pulse_trains),
                recording_site=effective_recording_site,
                selected_panels=selected_panels,
                morphology_name=neuron.morphology_name,
            )
        else:
            result = simulate_circuit_current_clamp(
                [_make_sim_neuron_spec(item) for item in project.neurons],
                [_make_sim_connection_spec(item) for item in project.connections],
                selected_neuron_id=neuron.id,
                isolate_selected_neuron=False,
                selected_panels=selected_panels,
                recording_site=effective_recording_site,
            )
        figures = build_trace_panel_figures(result, selected_panels, theme="light")
        elapsed = time.perf_counter() - start_time
        return (
            figures.get("voltage", empty_trace_figures["voltage"]),
            figures.get("applied_trace", empty_trace_figures["applied_trace"]),
            figures.get("command_trace", empty_trace_figures["command_trace"]),
            figures.get("ionic_currents", empty_trace_figures["ionic_currents"]),
            figures.get("gating", empty_trace_figures["gating"]),
            figures.get("conductances", empty_trace_figures["conductances"]),
            "tab-clamp",
            f"Current clamp updated in {elapsed:.3f} s for {neuron.label}{' (isolated)' if isolate_selected_neuron else ' (in circuit)'} from {recording_source_label}: {recording_source_text}.",
        )

    @app.callback(
        Output("vclamp-figures-store", "data"),
        Output("results-tabs", "active_tab", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("run-vclamp", "n_clicks"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        State("plot-panels", "value"),
        State("selected-connection-id", "data"),
        State("train-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def run_voltage_clamp(n_clicks, project_data, selected_neuron_id, plot_panels, selected_connection_id, train_selected_rows):
        if not n_clicks:
            raise PreventUpdate
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate
        selected_panels = set(plot_panels or PLOT_PANEL_DEFAULT_SELECTION)
        trains = list(neuron.voltage_trains)
        used_default_step = False
        if not trains:
            _, trains = default_setup(VOLTAGE_CLAMP, neuron.morphology_name)
            used_default_step = True
        effective_recording_site, recording_source_label, recording_source_text = _resolve_recording_site(
            project,
            neuron,
            selected_connection_id,
            train_selected_rows,
        )
        start_time = time.perf_counter()
        result = simulate_voltage_clamp(
            neuron.neuron_config,
            trains,
            recording_site=effective_recording_site,
            morphology_name=neuron.morphology_name,
        )
        figures = build_trace_panel_figures(result, selected_panels, theme="light")
        elapsed = time.perf_counter() - start_time
        return (
            _figure_map_payload(figures, empty_trace_figures),
            "tab-vclamp",
            (
                f"Voltage clamp updated in {elapsed:.3f} s for {neuron.label} from "
                f"{recording_source_label}: {recording_source_text}."
                + (" Used a default soma voltage step because no voltage step is configured." if used_default_step else "")
            ),
        )

    @app.callback(
        Output("fi-figure-store", "data"),
        Output("results-tabs", "active_tab", allow_duplicate=True),
        Output("status-banner", "children", allow_duplicate=True),
        Input("run-fi", "n_clicks"),
        State("circuit-store", "data"),
        State("selected-neuron-id", "data"),
        State("selected-connection-id", "data"),
        State("train-table", "selected_rows"),
        prevent_initial_call=True,
    )
    def run_fi_sweep(n_clicks, project_data, selected_neuron_id, selected_connection_id, train_selected_rows):
        if not n_clicks:
            raise PreventUpdate
        project = _project_from_dict(project_data)
        neuron = _selected_neuron(project, selected_neuron_id)
        if neuron is None:
            raise PreventUpdate
        effective_recording_site, recording_source_label, recording_source_text = _resolve_recording_site(
            project,
            neuron,
            selected_connection_id,
            train_selected_rows,
        )
        start_time = time.perf_counter()
        fi_result = simulate_fi_sweep(
            neuron.neuron_config,
            neuron.fi_config,
            stimulation_site=neuron.fi_site,
            recording_site=effective_recording_site,
            morphology_name=neuron.morphology_name,
        )
        elapsed = time.perf_counter() - start_time
        return _figure_payload(build_fi_curve_figure(fi_result)), "tab-fi", f"F-I sweep completed in {elapsed:.3f} s for {neuron.label} from {recording_source_label}: {recording_source_text}."

    @app.callback(
        Output("vclamp-graph-voltage", "figure"),
        Output("vclamp-graph-applied_trace", "figure"),
        Output("vclamp-graph-command_trace", "figure"),
        Output("vclamp-graph-ionic_currents", "figure"),
        Output("vclamp-graph-gating", "figure"),
        Output("vclamp-graph-conductances", "figure"),
        Input("vclamp-figures-store", "data"),
        Input("results-tabs", "active_tab"),
    )
    def render_voltage_clamp_figures(figure_payload, active_tab):
        del active_tab
        figures = figure_payload or empty_trace_payload
        return (
            figures.get("voltage", empty_trace_payload["voltage"]),
            figures.get("applied_trace", empty_trace_payload["applied_trace"]),
            figures.get("command_trace", empty_trace_payload["command_trace"]),
            figures.get("ionic_currents", empty_trace_payload["ionic_currents"]),
            figures.get("gating", empty_trace_payload["gating"]),
            figures.get("conductances", empty_trace_payload["conductances"]),
        )

    @app.callback(
        Output("fi-graph", "figure"),
        Input("fi-figure-store", "data"),
        Input("results-tabs", "active_tab"),
    )
    def render_fi_figure(figure_payload, active_tab):
        del active_tab
        return figure_payload or empty_fi_payload

    return app


def run() -> int:
    app = create_app()
    port = _find_free_port()
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=port, debug=False)
    return 0
