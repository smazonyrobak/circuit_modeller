from __future__ import annotations

import contextlib
import copy
import concurrent.futures
import io
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from xml.etree import ElementTree as ET

VOLTAGE_CLAMP = "voltage_clamp"
CURRENT_CLAMP = "current_clamp"
CURRENT_UNIT_NA = "nA"
CURRENT_UNIT_DENSITY = "uA/cm2"
CURRENT_INJECTION_UNITS = (CURRENT_UNIT_NA, CURRENT_UNIT_DENSITY)


@dataclass(slots=True)
class NeuronConfig:
    duration_ms: float = 50.0
    dt_ms: float = 0.01
    v_rest_mV: float = -65.0
    holding_mV: float = -65.0
    current_injection_unit: str = CURRENT_UNIT_NA
    holding_current: float = 0.0
    cm_uF_cm2: float = 1.0
    gna_mS_cm2: float = 120.0
    gk_mS_cm2: float = 36.0
    gl_mS_cm2: float = 0.3
    ena_mV: float = 63.54
    ek_mV: float = -74.16
    eleak_mV: float = -54.3
    gcl_mS_cm2: float = 0.05
    gshunt_mS_cm2: float = 0.02
    ecl_mV: float = -70.35603740607618


@dataclass(slots=True)
class VoltagePulseTrain:
    label: str = "Pulse Train 1"
    start_ms: float = 20.0
    pulse_width_ms: float = 5.0
    interval_ms: float = 20.0
    pulse_count: int = 5
    amplitude: float = 15.0
    section_name: str = "soma_0"
    section_x: float = 0.5

    def active(self, time_ms: float) -> bool:
        if self.pulse_count <= 0 or self.pulse_width_ms <= 0:
            return False
        for index in range(self.pulse_count):
            start = self.start_ms + index * self.interval_ms
            end = start + self.pulse_width_ms
            if start <= time_ms < end:
                return True
        return False


@dataclass(slots=True)
class SimulationResult:
    mode: str = VOLTAGE_CLAMP
    times_ms: list[float] = field(default_factory=list)
    voltage_mV: list[float] = field(default_factory=list)
    current_trace: list[float] = field(default_factory=list)
    current_trace_label: str = "Clamp Current"
    current_trace_unit: str = "nA"
    command_trace: list[float] = field(default_factory=list)
    command_trace_label: str = "Command Voltage"
    command_trace_unit: str = "mV"
    ionic_currents_uA_cm2: dict[str, list[float]] = field(default_factory=dict)
    gating_variables: dict[str, list[float]] = field(default_factory=dict)
    conductances_mS_cm2: dict[str, list[float]] = field(default_factory=dict)
    eleak_mV: float = 0.0
    ecl_mV: float = 0.0


@dataclass(slots=True)
class IVSweepConfig:
    start_mV: float = -90.0
    end_mV: float = 20.0
    step_mV: float = 10.0
    pulse_start_ms: float = 20.0
    pulse_width_ms: float = 10.0


@dataclass(slots=True)
class IVSweepResult:
    command_voltage_mV: list[float] = field(default_factory=list)
    peak_inward_current_nA: list[float] = field(default_factory=list)
    peak_outward_current_nA: list[float] = field(default_factory=list)
    steady_current_nA: list[float] = field(default_factory=list)
    pulse_start_ms: float = 0.0
    pulse_width_ms: float = 0.0


@dataclass(slots=True)
class FISweepConfig:
    start_current: float = 0.0
    end_current: float = 2.0
    step_current: float = 0.2
    pulse_start_ms: float = 5.0
    pulse_width_ms: float = 25.0


@dataclass(slots=True)
class FISweepResult:
    current_values: list[float] = field(default_factory=list)
    current_unit: str = CURRENT_UNIT_NA
    spike_count: list[int] = field(default_factory=list)
    firing_rate_hz: list[float] = field(default_factory=list)
    pulse_start_ms: float = 0.0
    pulse_width_ms: float = 0.0


@dataclass(slots=True)
class MorphologySite:
    section_name: str = "soma_0"
    section_x: float = 0.5
    x_um: float = 0.0
    y_um: float = 0.0
    z_um: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.section_name} @ {self.section_x:.2f}"


@dataclass(frozen=True, slots=True)
class MorphologySection:
    name: str
    section_type: str
    points_3d: tuple[tuple[float, float, float, float], ...]
    cumulative_lengths_um: tuple[float, ...]
    total_length_um: float


@dataclass(frozen=True, slots=True)
class MorphologyPreview:
    sections: tuple[MorphologySection, ...]
    bounds_xyz: tuple[float, float, float, float, float, float]


@dataclass(slots=True)
class CircuitNeuronSpec:
    neuron_id: str
    label: str
    morphology_name: str
    neuron: NeuronConfig
    pulse_trains: list[VoltagePulseTrain]
    recording_site: MorphologySite
    fi_site: MorphologySite
    output_site: MorphologySite


@dataclass(slots=True)
class CircuitConnectionSpec:
    connection_id: str
    source_id: str
    target_id: str
    target_site: MorphologySite
    current_nA: float = 1.0
    pulse_width_ms: float = 1.5
    delay_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class TutorialCellSpec:
    soma_diam_um: float
    soma_length_um: float
    ra_ohm_cm: float
    cm_uF_cm2: float
    gna_mS_cm2: float
    gk_mS_cm2: float
    gl_mS_cm2: float
    ena_mV: float
    ek_mV: float
    el_mV: float
    init_v_mV: float


@dataclass(frozen=True, slots=True)
class TutorialClampSpec:
    delay_ms: float
    duration_ms: float
    conditioning_voltage_mV: float
    testing_voltage_mV: float
    return_voltage_mV: float
    series_resistance_ohm: float


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SWC_FILENAME = "pyramidal_neuron.swc"
PYRAMIDAL_SWC_SOURCE = PROJECT_ROOT / DEFAULT_SWC_FILENAME
TUTORIAL_SOURCE_DIR = PROJECT_ROOT / "hodgkin_huxley_tutorial" / "Tutorial" / "Source"
HHCELL_SOURCE = TUTORIAL_SOURCE_DIR / "hhcell.cell.nml"
VCLAMP_SOURCE = TUTORIAL_SOURCE_DIR / "vclamp.xml"
NEUROML_NS = {"nml": "http://www.neuroml.org/schema/neuroml2"}
LEMS_NS = {"lems": "http://www.neuroml.org/lems/0.7.4"}
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

FALLBACK_TUTORIAL_CELL = TutorialCellSpec(
    soma_diam_um=17.841242,
    soma_length_um=17.841242,
    ra_ohm_cm=30.0,
    cm_uF_cm2=1.0,
    gna_mS_cm2=120.0,
    gk_mS_cm2=36.0,
    gl_mS_cm2=0.3,
    ena_mV=50.0,
    ek_mV=-77.0,
    el_mV=-54.387,
    init_v_mV=-65.0,
)
FALLBACK_TUTORIAL_CLAMP = TutorialClampSpec(
    delay_ms=10.0,
    duration_ms=30.0,
    conditioning_voltage_mV=-63.77,
    testing_voltage_mV=10.0,
    return_voltage_mV=-63.77,
    series_resistance_ohm=1e4,
)

DEFAULT_CM_UF_CM2 = 1.0
DEFAULT_GNA_MS_CM2 = 120.0
DEFAULT_GK_MS_CM2 = 36.0
DEFAULT_GL_MS_CM2 = 0.3
DEFAULT_ENA_MV = 63.54
DEFAULT_EK_MV = -74.16
DEFAULT_ELEAK_MV = -54.3
DEFAULT_GCL_MS_CM2 = 0.05
DEFAULT_GSHUNT_MS_CM2 = 0.02
DEFAULT_CLI_MM = 7.0
DEFAULT_CLO_MM = 130.0
DEFAULT_ECL_MV = -70.35603740607618
DEFAULT_VREST_MV = -65.0
DEFAULT_DT_MS = 0.01
DEFAULT_DURATION_MS = 50.0
DEFAULT_CURRENT_STEP_START_MS = 5.0
DEFAULT_CURRENT_STEP_WIDTH_MS = 1.0
DEFAULT_CURRENT_STEP_NA = 1.0
DEFAULT_CURRENT_STEP_DENSITY_UA_CM2 = 60.0
DEFAULT_VCLAMP_HOLDING_MV = DEFAULT_VREST_MV
DEFAULT_VCLAMP_TEST_MV = 10.0
DEFAULT_VCLAMP_START_MS = 10.0
DEFAULT_VCLAMP_WIDTH_MS = 30.0
DEFAULT_FI_START_NA = 0.0
DEFAULT_FI_END_NA = 2.0
DEFAULT_FI_STEP_NA = 0.2
DEFAULT_FI_START_UA_CM2 = 0.0
DEFAULT_FI_END_UA_CM2 = 200.0
DEFAULT_FI_STEP_UA_CM2 = 20.0
DEFAULT_CELSIUS_C = 6.3
TARGET_SEGMENT_LENGTH_UM = 40.0
GAS_CONSTANT_J_MOL_K = 8.314462618
FARADAY_C_MOL = 96485.33212
FI_PARALLEL_MIN_SWEEP_POINTS = 4
FI_MAX_WORKERS = 4
DEFAULT_CONNECTION_CURRENT_NA = 1.0
DEFAULT_CONNECTION_PULSE_MS = 1.5


def _parse_number(raw_value: str) -> float:
    match = NUMBER_RE.search(raw_value)
    if match is None:
        raise ValueError(f"Could not parse numeric value from {raw_value!r}.")
    value = float(match.group(0))
    unit = raw_value[match.end():].strip()
    if unit == "kohm_cm":
        return value * 1000.0
    return value


def _find_required(element: ET.Element, query: str, namespaces: dict[str, str]) -> ET.Element:
    found = element.find(query, namespaces)
    if found is None:
        raise ValueError(f"Missing required XML element: {query}")
    return found


@lru_cache(maxsize=1)
def load_tutorial_cell_spec() -> TutorialCellSpec:
    if not HHCELL_SOURCE.exists():
        return FALLBACK_TUTORIAL_CELL

    root = ET.parse(HHCELL_SOURCE).getroot()
    soma_segment = _find_required(root, ".//nml:segment[@id='0']", NEUROML_NS)
    proximal = _find_required(soma_segment, "nml:proximal", NEUROML_NS)

    membrane = _find_required(root, ".//nml:membraneProperties", NEUROML_NS)
    intracellular = _find_required(root, ".//nml:intracellularProperties", NEUROML_NS)
    leak = _find_required(membrane, "nml:channelDensity[@id='leak']", NEUROML_NS)
    na = _find_required(membrane, "nml:channelDensity[@id='naChans']", NEUROML_NS)
    k = _find_required(membrane, "nml:channelDensity[@id='kChans']", NEUROML_NS)
    capacitance = _find_required(membrane, "nml:specificCapacitance", NEUROML_NS)
    init_v = _find_required(membrane, "nml:initMembPotential", NEUROML_NS)
    resistivity = _find_required(intracellular, "nml:resistivity", NEUROML_NS)

    diameter_um = _parse_number(proximal.attrib["diameter"])

    return TutorialCellSpec(
        soma_diam_um=diameter_um,
        # NeuroML encodes the single soma as a sphere; use the equivalent NEURON cylinder
        # that preserves the 1000 um2 surface area used throughout the tutorial.
        soma_length_um=diameter_um,
        ra_ohm_cm=_parse_number(resistivity.attrib["value"]),
        cm_uF_cm2=_parse_number(capacitance.attrib["value"]),
        gna_mS_cm2=_parse_number(na.attrib["condDensity"]),
        gk_mS_cm2=_parse_number(k.attrib["condDensity"]),
        gl_mS_cm2=_parse_number(leak.attrib["condDensity"]),
        ena_mV=_parse_number(na.attrib["erev"]),
        ek_mV=_parse_number(k.attrib["erev"]),
        el_mV=_parse_number(leak.attrib["erev"]),
        init_v_mV=_parse_number(init_v.attrib["value"]),
    )


@lru_cache(maxsize=1)
def load_tutorial_clamp_spec() -> TutorialClampSpec:
    if not VCLAMP_SOURCE.exists():
        return FALLBACK_TUTORIAL_CLAMP

    root = ET.parse(VCLAMP_SOURCE).getroot()
    clamp = _find_required(root, ".//lems:voltageClamp2[@id='vClamp']", LEMS_NS)
    return TutorialClampSpec(
        delay_ms=_parse_number(clamp.attrib["delay"]),
        duration_ms=_parse_number(clamp.attrib["duration"]),
        conditioning_voltage_mV=_parse_number(clamp.attrib["conditioningVoltage"]),
        testing_voltage_mV=_parse_number(clamp.attrib["testingVoltage"]),
        return_voltage_mV=_parse_number(clamp.attrib["returnVoltage"]),
        series_resistance_ohm=_parse_number(clamp.attrib["simpleSeriesResistance"]),
    )


def soma_area_cm2(tutorial_cell: TutorialCellSpec) -> float:
    soma_area_um2 = math.pi * tutorial_cell.soma_diam_um * tutorial_cell.soma_length_um
    return soma_area_um2 * 1e-8


def density_uA_cm2_to_nA(current_density_uA_cm2: float, area_cm2: float) -> float:
    return current_density_uA_cm2 * area_cm2 * 1000.0


def current_nA_to_density_uA_cm2(current_nA: float, area_cm2: float) -> float:
    if area_cm2 <= 0:
        raise ValueError("Soma area must be positive.")
    return current_nA / (1000.0 * area_cm2)


def _validate_current_unit(unit: str) -> str:
    if unit not in CURRENT_INJECTION_UNITS:
        raise ValueError(f"Unsupported current injection unit: {unit!r}")
    return unit


def current_value_to_nA(value: float, unit: str, area_cm2: float) -> float:
    unit = _validate_current_unit(unit)
    if unit == CURRENT_UNIT_NA:
        return value
    return density_uA_cm2_to_nA(value, area_cm2)


def current_nA_to_value(current_nA: float, unit: str, area_cm2: float) -> float:
    unit = _validate_current_unit(unit)
    if unit == CURRENT_UNIT_NA:
        return current_nA
    return current_nA_to_density_uA_cm2(current_nA, area_cm2)


def chloride_reversal_mV(cli_mM: float, clo_mM: float, temperature_c: float = DEFAULT_CELSIUS_C) -> float:
    if cli_mM <= 0 or clo_mM <= 0:
        raise ValueError("Chloride concentrations must be positive.")
    temperature_K = temperature_c + 273.15
    return -((GAS_CONSTANT_J_MOL_K * temperature_K) / FARADAY_C_MOL) * 1000.0 * math.log(clo_mM / cli_mM)


def _configure_runtime_env() -> None:
    cache_root = Path(tempfile.gettempdir()) / "netpyne_modeler_runtime"
    mpl_dir = cache_root / "mpl"
    xdg_cache_dir = cache_root / "xdg_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def _import_backend():
    _configure_runtime_env()
    try:
        from netpyne import specs, sim
        from neuron import h
    except ImportError as exc:
        raise RuntimeError(
            "NetPyNe/NEURON is not installed in .venv. Run "
            "`python3 bootstrap.py` to bootstrap the environment."
        ) from exc
    return specs, sim, h


def list_available_swc_files(directory: Path | None = None) -> list[str]:
    search_dir = directory or PROJECT_ROOT
    return sorted(path.name for path in search_dir.glob("*.swc"))


def resolve_morphology_path(morphology_name: str | Path | None = None) -> Path:
    if morphology_name is None or str(morphology_name).strip() == "":
        candidate = PYRAMIDAL_SWC_SOURCE
    else:
        raw = Path(morphology_name)
        candidate = raw if raw.is_absolute() else PROJECT_ROOT / raw.name
    if not candidate.exists():
        raise FileNotFoundError(f"Morphology file not found: {candidate}")
    return candidate.resolve()


def _sanitized_swc_path(morphology_name: str | Path | None = None) -> Path:
    source_path = resolve_morphology_path(morphology_name)
    cache_root = Path(tempfile.gettempdir()) / "netpyne_modeler_runtime"
    cache_root.mkdir(parents=True, exist_ok=True)
    sanitized_path = cache_root / f"{source_path.stem}_clean.swc"
    source_text = source_path.read_text(encoding="utf-8")
    sanitized_lines = [line for line in source_text.splitlines() if line.strip()]
    sanitized_text = "\n".join(sanitized_lines) + "\n"
    if not sanitized_path.exists() or sanitized_path.read_text(encoding="utf-8") != sanitized_text:
        sanitized_path.write_text(sanitized_text, encoding="utf-8")
    return sanitized_path


def _section_type_from_name(section_name: str) -> str:
    return section_name.split("_", 1)[0]


def _segment_count_for_length_um(length_um: float) -> int:
    if length_um <= 0:
        return 1
    segment_count = max(1, int(math.ceil(length_um / TARGET_SEGMENT_LENGTH_UM)))
    if segment_count % 2 == 0:
        segment_count += 1
    return segment_count


def _set_section_compartmentalization(section_rule: dict) -> None:
    geom = section_rule.setdefault("geom", {})
    length_um = float(geom.get("L", 10.0))
    # Keep the exact SWC geometry for rendering and site placement, but use a
    # coarser electrical discretization so live updates stay responsive.
    geom["nseg"] = _segment_count_for_length_um(length_um)


def _apply_biophysics_to_rule(cell_rule: dict, neuron: NeuronConfig) -> float:
    ecl_mV = neuron.ecl_mV
    for section_name, section_rule in cell_rule["secs"].items():
        geom = section_rule.setdefault("geom", {})
        geom["Ra"] = load_tutorial_cell_spec().ra_ohm_cm
        geom["cm"] = neuron.cm_uF_cm2
        _set_section_compartmentalization(section_rule)

        section_type = _section_type_from_name(section_name)
        mechs: dict[str, dict[str, float]] = {}

        if section_type in {"soma", "axon"}:
            mechs["hh"] = {
                "gnabar": neuron.gna_mS_cm2 / 1000.0,
                "gkbar": neuron.gk_mS_cm2 / 1000.0,
                "gl": neuron.gl_mS_cm2 / 1000.0,
                "el": neuron.eleak_mV,
            }
        else:
            # Dendrites use the same conductance densities for now so the
            # morphology remains fully active until section-specific channel maps are added.
            mechs["hh"] = {
                "gnabar": neuron.gna_mS_cm2 / 1000.0,
                "gkbar": neuron.gk_mS_cm2 / 1000.0,
                "gl": neuron.gl_mS_cm2 / 1000.0,
                "el": neuron.eleak_mV,
            }

        mechs["pas"] = {
            "g": (neuron.gcl_mS_cm2 + neuron.gshunt_mS_cm2) / 1000.0,
            "e": ecl_mV,
        }
        section_rule["mechs"] = mechs
    return ecl_mV


@lru_cache(maxsize=8)
def _load_cell_rule_base(morphology_name: str | None = None) -> dict:
    specs, _, _ = _import_backend()
    net_params = specs.NetParams()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rule = net_params.importCellParams(
            label="PYR_MORPH",
            fileName=str(_sanitized_swc_path(morphology_name)),
            cellName="",
            conds={"cellType": "PYR", "cellModel": "MorphHH"},
        )
    if hasattr(rule, "todict"):
        return rule.todict()
    return dict(rule)


def _clone_morphology_cell_rule(neuron: NeuronConfig, morphology_name: str | None = None) -> tuple[dict, float]:
    cell_rule = copy.deepcopy(_load_cell_rule_base(morphology_name))
    ecl_mV = _apply_biophysics_to_rule(cell_rule, neuron)
    return cell_rule, ecl_mV


def _clone_pyramidal_cell_rule(neuron: NeuronConfig) -> tuple[dict, float]:
    return _clone_morphology_cell_rule(neuron, DEFAULT_SWC_FILENAME)


def _build_morphology_section(section_name: str, section_rule: dict) -> MorphologySection:
    points = tuple(
        (float(pt[0]), float(pt[1]), float(pt[2]), float(pt[3]))
        for pt in section_rule.get("geom", {}).get("pt3d", [])
    )
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        distance = math.dist(first[:3], second[:3])
        cumulative.append(cumulative[-1] + distance)
    total_length = cumulative[-1] if cumulative else float(section_rule.get("geom", {}).get("L", 0.0))
    return MorphologySection(
        name=section_name,
        section_type=_section_type_from_name(section_name),
        points_3d=points,
        cumulative_lengths_um=tuple(cumulative),
        total_length_um=total_length,
    )


@lru_cache(maxsize=8)
def load_morphology_preview(morphology_name: str | None = None) -> MorphologyPreview:
    cell_rule = _load_cell_rule_base(morphology_name)
    sections = tuple(
        _build_morphology_section(section_name, section_rule)
        for section_name, section_rule in cell_rule["secs"].items()
        if len(section_rule.get("geom", {}).get("pt3d", [])) >= 2
    )
    xs = [point[0] for section in sections for point in section.points_3d]
    ys = [point[1] for section in sections for point in section.points_3d]
    zs = [point[2] for section in sections for point in section.points_3d]
    bounds = (
        min(xs),
        max(xs),
        min(ys),
        max(ys),
        min(zs),
        max(zs),
    )
    return MorphologyPreview(sections=sections, bounds_xyz=bounds)


def load_pyramidal_morphology_preview() -> MorphologyPreview:
    return load_morphology_preview(DEFAULT_SWC_FILENAME)


def interpolate_section_site(section: MorphologySection, section_x: float) -> MorphologySite:
    if not section.points_3d:
        return MorphologySite(section_name=section.name, section_x=section_x)
    if section.total_length_um <= 0:
        x_um, y_um, z_um, _ = section.points_3d[0]
        return MorphologySite(section_name=section.name, section_x=section_x, x_um=x_um, y_um=y_um, z_um=z_um)

    clamped_x = min(1.0, max(0.0, section_x))
    target_distance = clamped_x * section.total_length_um
    cumulative = section.cumulative_lengths_um
    for index in range(len(cumulative) - 1):
        start_distance = cumulative[index]
        end_distance = cumulative[index + 1]
        if target_distance <= end_distance:
            first = section.points_3d[index]
            second = section.points_3d[index + 1]
            segment_length = max(end_distance - start_distance, 1e-9)
            fraction = (target_distance - start_distance) / segment_length
            x_um = first[0] + fraction * (second[0] - first[0])
            y_um = first[1] + fraction * (second[1] - first[1])
            z_um = first[2] + fraction * (second[2] - first[2])
            return MorphologySite(
                section_name=section.name,
                section_x=clamped_x,
                x_um=x_um,
                y_um=y_um,
                z_um=z_um,
            )
    last = section.points_3d[-1]
    return MorphologySite(section_name=section.name, section_x=clamped_x, x_um=last[0], y_um=last[1], z_um=last[2])


def default_recording_site(morphology_name: str | None = None) -> MorphologySite:
    preview = load_morphology_preview(morphology_name)
    soma_section = next((section for section in preview.sections if section.section_type == "soma"), preview.sections[0])
    return interpolate_section_site(soma_section, 0.5)


def estimate_site_segment_area_cm2(site: MorphologySite, morphology_name: str | None = None) -> float:
    preview = load_morphology_preview(morphology_name)
    section = next((item for item in preview.sections if item.name == site.section_name), None)
    base_rule = _load_cell_rule_base(morphology_name)
    section_rule = base_rule["secs"].get(site.section_name)
    if section_rule is None:
        raise ValueError(f"Unknown morphology section: {site.section_name!r}")

    geom = section_rule.get("geom", {})
    length_um = float(geom.get("L", 10.0))
    segment_count = _segment_count_for_length_um(length_um)
    segment_length_um = length_um / max(1, segment_count)

    if section is None or not section.points_3d:
        diameter_um = float(geom.get("diam", 1.0))
        return math.pi * diameter_um * segment_length_um * 1e-8

    clamped_x = min(1.0, max(0.0, site.section_x))
    if section.total_length_um <= 0:
        diameter_um = float(section.points_3d[0][3])
        return math.pi * diameter_um * segment_length_um * 1e-8

    target_distance = clamped_x * section.total_length_um
    cumulative = section.cumulative_lengths_um
    for index in range(len(cumulative) - 1):
        start_distance = cumulative[index]
        end_distance = cumulative[index + 1]
        if target_distance <= end_distance:
            first = section.points_3d[index]
            second = section.points_3d[index + 1]
            path_length = max(end_distance - start_distance, 1e-9)
            fraction = (target_distance - start_distance) / path_length
            diameter_um = first[3] + fraction * (second[3] - first[3])
            return math.pi * diameter_um * segment_length_um * 1e-8

    diameter_um = float(section.points_3d[-1][3])
    return math.pi * diameter_um * segment_length_um * 1e-8


def command_delta(time_ms: float, trains: list[VoltagePulseTrain]) -> float:
    delta = 0.0
    for train in trains:
        if train.active(time_ms):
            delta += train.amplitude
    return delta


def _validate_simulation_inputs(neuron: NeuronConfig, trains: list[VoltagePulseTrain]) -> int:
    if neuron.duration_ms <= 0:
        raise ValueError("Simulation duration must be positive.")
    if neuron.dt_ms <= 0:
        raise ValueError("Time step must be positive.")
    if neuron.gl_mS_cm2 < 0:
        raise ValueError("Leak conductance must be non-negative.")
    if neuron.gcl_mS_cm2 < 0:
        raise ValueError("Chloride conductance must be non-negative.")
    if neuron.gshunt_mS_cm2 < 0:
        raise ValueError("Shunting conductance must be non-negative.")
    _validate_current_unit(neuron.current_injection_unit)
    for train in trains:
        if train.interval_ms <= 0:
            raise ValueError(f"{train.label}: interval must be positive.")
        if train.pulse_width_ms <= 0:
            raise ValueError(f"{train.label}: pulse width must be positive.")
        if train.pulse_count <= 0:
            raise ValueError(f"{train.label}: pulse count must be positive.")

    step_count = int(round(neuron.duration_ms / neuron.dt_ms))
    if step_count < 2:
        raise ValueError("Simulation duration is too short for the chosen time step.")
    return step_count


def _build_single_cell(specs, neuron: NeuronConfig, tutorial_cell: TutorialCellSpec):
    ecl_mV = neuron.ecl_mV
    net_params = specs.NetParams()
    net_params.cellParams["VC_HH"] = {
        "conds": {"cellType": "VCELL", "cellModel": "HH"},
        "secs": {
            "soma": {
                "geom": {
                    "diam": tutorial_cell.soma_diam_um,
                    "L": tutorial_cell.soma_length_um,
                    "Ra": tutorial_cell.ra_ohm_cm,
                    "cm": neuron.cm_uF_cm2,
                },
                "mechs": {
                    "hh": {
                        "gnabar": neuron.gna_mS_cm2 / 1000.0,
                        "gkbar": neuron.gk_mS_cm2 / 1000.0,
                        "gl": neuron.gl_mS_cm2 / 1000.0,
                        "el": neuron.eleak_mV,
                    },
                    "pas": {
                        "g": (neuron.gcl_mS_cm2 + neuron.gshunt_mS_cm2) / 1000.0,
                        "e": ecl_mV,
                    },
                },
                "vinit": neuron.v_rest_mV,
            }
        },
    }
    net_params.popParams["cell"] = {
        "cellType": "VCELL",
        "cellModel": "HH",
        "numCells": 1,
    }
    return net_params, ecl_mV


def _build_sim_config(specs, neuron: NeuronConfig):
    sim_config = specs.SimConfig()
    sim_config.duration = neuron.duration_ms
    sim_config.dt = neuron.dt_ms
    sim_config.verbose = False
    sim_config.analysis = {}
    sim_config.createNEURONObj = True
    sim_config.hParams["celsius"] = DEFAULT_CELSIUS_C
    sim_config.hParams["v_init"] = neuron.v_rest_mV
    return sim_config


def _record_hh_state(h, segment) -> dict[str, object]:
    return {
        "ina_mA_cm2": h.Vector().record(segment._ref_ina),
        "ik_mA_cm2": h.Vector().record(segment._ref_ik),
        "m": h.Vector().record(segment._ref_m_hh),
        "h": h.Vector().record(segment._ref_h_hh),
        "n": h.Vector().record(segment._ref_n_hh),
        "gna_S_cm2": h.Vector().record(segment._ref_gna_hh),
        "gk_S_cm2": h.Vector().record(segment._ref_gk_hh),
        "gl_S_cm2": h.Vector().record(segment._ref_gl_hh),
    }


def _record_passive_state(h, segment) -> dict[str, object]:
    return {
        "ipas_mA_cm2": h.Vector().record(segment._ref_i_pas),
        "gpas_S_cm2": h.Vector().record(segment._ref_g_pas),
    }


def _vector_to_floats(vector, recorded_len: int, scale: float = 1.0) -> list[float]:
    return [float(vector[index]) * scale for index in range(recorded_len)]


def _extract_membrane_traces(
    hh_vectors: dict[str, object],
    passive_vectors: dict[str, object],
    voltage_mV: list[float],
    neuron: NeuronConfig,
    ecl_mV: float,
    recorded_len: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    gcl_values = [neuron.gcl_mS_cm2] * recorded_len
    gshunt_values = [neuron.gshunt_mS_cm2] * recorded_len
    conductances = {
        "g_Na": _vector_to_floats(hh_vectors["gna_S_cm2"], recorded_len, scale=1000.0),
        "g_K": _vector_to_floats(hh_vectors["gk_S_cm2"], recorded_len, scale=1000.0),
        "g_leak": _vector_to_floats(hh_vectors["gl_S_cm2"], recorded_len, scale=1000.0),
        "g_Cl": gcl_values,
        "g_shunt": gshunt_values,
        "g_inhib_total": _vector_to_floats(passive_vectors["gpas_S_cm2"], recorded_len, scale=1000.0),
    }
    ionic_currents = {
        "I_Na": _vector_to_floats(hh_vectors["ina_mA_cm2"], recorded_len, scale=1000.0),
        "I_K": _vector_to_floats(hh_vectors["ik_mA_cm2"], recorded_len, scale=1000.0),
        "I_leak": [
            conductance * (voltage - neuron.eleak_mV)
            for conductance, voltage in zip(conductances["g_leak"], voltage_mV)
        ],
        "I_Cl": [
            conductance * (voltage - ecl_mV)
            for conductance, voltage in zip(gcl_values, voltage_mV)
        ],
        "I_shunt": [
            conductance * (voltage - ecl_mV)
            for conductance, voltage in zip(gshunt_values, voltage_mV)
        ],
        "I_inhib_total": _vector_to_floats(passive_vectors["ipas_mA_cm2"], recorded_len, scale=1000.0),
    }
    gating_variables = {
        "m": _vector_to_floats(hh_vectors["m"], recorded_len),
        "h": _vector_to_floats(hh_vectors["h"], recorded_len),
        "n": _vector_to_floats(hh_vectors["n"], recorded_len),
    }
    return ionic_currents, gating_variables, conductances


def _build_morphology_cell(specs, neuron: NeuronConfig, morphology_name: str | None = None) -> tuple[object, float]:
    cell_rule, ecl_mV = _clone_morphology_cell_rule(neuron, morphology_name)
    net_params = specs.NetParams()
    net_params.cellParams["PYR_MORPH"] = {
        "conds": {"cellType": "PYR", "cellModel": "MorphHH"},
        "secs": cell_rule["secs"],
        "secLists": cell_rule.get("secLists", {}),
        "globals": cell_rule.get("globals", {}),
    }
    net_params.popParams["cell"] = {
        "cellType": "PYR",
        "cellModel": "MorphHH",
        "numCells": 1,
    }
    return net_params, ecl_mV


def _build_circuit_net_params(specs, neurons: list[CircuitNeuronSpec]) -> tuple[object, dict[str, float]]:
    net_params = specs.NetParams()
    ecl_by_id: dict[str, float] = {}
    for neuron_spec in neurons:
        cell_rule, ecl_mV = _clone_morphology_cell_rule(neuron_spec.neuron, neuron_spec.morphology_name)
        cell_type = f"CELL_{neuron_spec.neuron_id}"
        net_params.cellParams[cell_type] = {
            "conds": {"cellType": cell_type, "cellModel": "MorphHH"},
            "secs": cell_rule["secs"],
            "secLists": cell_rule.get("secLists", {}),
            "globals": cell_rule.get("globals", {}),
        }
        net_params.popParams[neuron_spec.neuron_id] = {
            "cellType": cell_type,
            "cellModel": "MorphHH",
            "numCells": 1,
        }
        ecl_by_id[neuron_spec.neuron_id] = ecl_mV
    return net_params, ecl_by_id


def _resolve_site(cell, site: MorphologySite | None):
    resolved = site or default_recording_site()
    if resolved.section_name not in cell.secs:
        raise ValueError(f"Selected morphology section {resolved.section_name!r} does not exist in the cell.")
    section_x = min(1.0, max(0.0, resolved.section_x))
    return cell.secs[resolved.section_name]["hObj"], section_x


def _apply_section_reversal_potentials(cell, neuron: NeuronConfig) -> None:
    for sec_data in cell.secs.values():
        section = sec_data.get("hObj")
        if section is None:
            continue
        section.ena = neuron.ena_mV
        section.ek = neuron.ek_mV


def _attach_current_clamp_train(h, sec, loc: float, times_ms: list[float], current_nA_values: list[float]):
    clamp = h.IClamp(sec(loc))
    clamp.delay = 0.0
    clamp.dur = 1e9
    time_vec = h.Vector(times_ms)
    command_vec = h.Vector(current_nA_values)
    command_vec.play(clamp._ref_amp, time_vec)
    return clamp, time_vec, command_vec


def _simple_synaptic_weight_uS(current_nA: float, resting_mV: float, reversal_mV: float = 0.0) -> float:
    driving_force_mV = max(1.0, abs(resting_mV - reversal_mV))
    return current_nA / driving_force_mV


def _command_delta_for_train(time_ms: float, train: VoltagePulseTrain) -> float:
    return train.amplitude if train.active(time_ms) else 0.0


def simulate_voltage_clamp(
    neuron: NeuronConfig,
    trains: list[VoltagePulseTrain],
    recording_site: MorphologySite | None = None,
    morphology_name: str | None = None,
) -> SimulationResult:
    step_count = _validate_simulation_inputs(neuron, trains)

    specs, sim, h = _import_backend()

    tutorial_cell = load_tutorial_cell_spec()
    tutorial_clamp = load_tutorial_clamp_spec()
    ecl_mV = neuron.ecl_mV
    times_ms = [index * neuron.dt_ms for index in range(step_count + 1)]
    command_voltage = [neuron.holding_mV + command_delta(time_ms, trains) for time_ms in times_ms]

    if morphology_name:
        net_params, ecl_mV = _build_morphology_cell(specs, neuron, morphology_name)
    else:
        net_params, ecl_mV = _build_single_cell(specs, neuron, tutorial_cell)
    sim_config = _build_sim_config(specs, neuron)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell = sim.net.cells[0]
        _apply_section_reversal_potentials(cell, neuron)
        if morphology_name:
            sec, sec_x = _resolve_site(cell, recording_site or default_recording_site(morphology_name))
            segment = sec(sec_x)
        else:
            sec = cell.secs["soma"]["hObj"]
            segment = sec(0.5)

        clamp = h.SEClamp(sec(0.5))
        clamp.dur1 = 1e9
        clamp.rs = tutorial_clamp.series_resistance_ohm / 1e6
        clamp.amp1 = neuron.holding_mV

        t_vec = h.Vector(times_ms)
        command_vec = h.Vector(command_voltage)
        command_vec.play(clamp._ref_amp1, t_vec)

        voltage_rec = h.Vector().record(segment._ref_v)
        current_rec = h.Vector().record(clamp._ref_i)
        time_rec = h.Vector().record(h._ref_t)
        hh_vectors = _record_hh_state(h, segment)
        passive_vectors = _record_passive_state(h, segment)

        sim.simulate()

    # Trim to the common recorded length in case NEURON includes one extra sample.
    recorded_len = min(len(time_rec), len(voltage_rec), len(current_rec), len(command_voltage))
    voltage_trace = [float(voltage_rec[index]) for index in range(recorded_len)]
    ionic_currents, gating_variables, conductances = _extract_membrane_traces(
        hh_vectors,
        passive_vectors,
        voltage_trace,
        neuron,
        ecl_mV,
        recorded_len,
    )
    return SimulationResult(
        mode=VOLTAGE_CLAMP,
        times_ms=[float(time_rec[index]) for index in range(recorded_len)],
        voltage_mV=voltage_trace,
        current_trace=[float(current_rec[index]) for index in range(recorded_len)],
        current_trace_label="Clamp Current",
        current_trace_unit="nA",
        command_trace=command_voltage[:recorded_len],
        command_trace_label="Command Voltage",
        command_trace_unit="mV",
        ionic_currents_uA_cm2=ionic_currents,
        gating_variables=gating_variables,
        conductances_mS_cm2=conductances,
        eleak_mV=neuron.eleak_mV,
        ecl_mV=ecl_mV,
    )


def simulate_current_clamp(
    neuron: NeuronConfig,
    trains: list[VoltagePulseTrain],
    recording_site: MorphologySite | None = None,
    selected_panels: set[str] | None = None,
    morphology_name: str | None = None,
) -> SimulationResult:
    step_count = _validate_simulation_inputs(neuron, trains)

    specs, sim, h = _import_backend()
    current_unit = neuron.current_injection_unit

    times_ms = [index * neuron.dt_ms for index in range(step_count + 1)]
    configured_command = [neuron.holding_current + command_delta(time_ms, trains) for time_ms in times_ms]
    record_membrane_state = selected_panels is None or any(
        panel in selected_panels for panel in ("ionic_currents", "gating", "conductances")
    )

    net_params, ecl_mV = _build_morphology_cell(specs, neuron, morphology_name)
    sim_config = _build_sim_config(specs, neuron)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell = sim.net.cells[0]
        _apply_section_reversal_potentials(cell, neuron)
        record_sec, record_x = _resolve_site(cell, recording_site or default_recording_site(morphology_name))
        segment = record_sec(record_x)

        holding_clamp = h.IClamp(record_sec(record_x))
        holding_clamp.delay = 0.0
        holding_clamp.dur = 1e9
        holding_area_cm2 = segment.area() * 1e-8

        t_vec = h.Vector(times_ms)
        holding_command_nA = [
            current_value_to_nA(neuron.holding_current, current_unit, holding_area_cm2)
            for _ in times_ms
        ]
        command_vec = h.Vector(holding_command_nA)
        command_vec.play(holding_clamp._ref_amp, t_vec)

        total_applied_current_nA = holding_command_nA.copy()

        train_clamps = []
        for train in trains:
            if train.section_name not in cell.secs:
                raise ValueError(f"{train.label}: selected section {train.section_name!r} does not exist.")
            stim_sec = cell.secs[train.section_name]["hObj"]
            stim_x = min(1.0, max(0.0, train.section_x))
            stim_segment = stim_sec(stim_x)
            stim_area_cm2 = stim_segment.area() * 1e-8

            stim_clamp = h.IClamp(stim_sec(stim_x))
            stim_clamp.delay = 0.0
            stim_clamp.dur = 1e9
            stim_t_vec = h.Vector(times_ms)
            stim_command_nA = [
                current_value_to_nA(_command_delta_for_train(time_ms, train), current_unit, stim_area_cm2)
                for time_ms in times_ms
            ]
            stim_command_vec = h.Vector(stim_command_nA)
            stim_command_vec.play(stim_clamp._ref_amp, stim_t_vec)
            train_clamps.append((stim_clamp, stim_t_vec, stim_command_vec))
            total_applied_current_nA = [
                total_applied_current_nA[index] + stim_command_nA[index]
                for index in range(len(total_applied_current_nA))
            ]

        voltage_rec = h.Vector().record(segment._ref_v)
        time_rec = h.Vector().record(h._ref_t)
        hh_vectors = _record_hh_state(h, segment) if record_membrane_state else {}
        passive_vectors = _record_passive_state(h, segment) if record_membrane_state else {}

        sim.simulate()

    recorded_len = min(len(time_rec), len(voltage_rec), len(total_applied_current_nA), len(configured_command))
    current_trace = total_applied_current_nA[:recorded_len]
    voltage_trace = [float(voltage_rec[index]) for index in range(recorded_len)]
    if record_membrane_state:
        ionic_currents, gating_variables, conductances = _extract_membrane_traces(
            hh_vectors,
            passive_vectors,
            voltage_trace,
            neuron,
            ecl_mV,
            recorded_len,
        )
    else:
        ionic_currents, gating_variables, conductances = {}, {}, {}

    return SimulationResult(
        mode=CURRENT_CLAMP,
        times_ms=[float(time_rec[index]) for index in range(recorded_len)],
        voltage_mV=voltage_trace,
        current_trace=current_trace,
        current_trace_label="Total Applied Current",
        current_trace_unit="nA",
        command_trace=configured_command[:recorded_len],
        command_trace_label=(
            "Configured Command Current" if current_unit == CURRENT_UNIT_NA else "Configured Command Density"
        ),
        command_trace_unit=current_unit,
        ionic_currents_uA_cm2=ionic_currents,
        gating_variables=gating_variables,
        conductances_mS_cm2=conductances,
        eleak_mV=neuron.eleak_mV,
        ecl_mV=ecl_mV,
    )


def simulate_circuit_current_clamp(
    neurons: list[CircuitNeuronSpec],
    connections: list[CircuitConnectionSpec],
    selected_neuron_id: str,
    isolate_selected_neuron: bool = False,
    selected_panels: set[str] | None = None,
    recording_site: MorphologySite | None = None,
) -> SimulationResult:
    if not neurons:
        raise ValueError("Circuit does not contain any neurons.")

    neuron_by_id = {item.neuron_id: item for item in neurons}
    selected_spec = neuron_by_id.get(selected_neuron_id)
    if selected_spec is None:
        raise ValueError(f"Selected neuron {selected_neuron_id!r} does not exist.")

    if isolate_selected_neuron:
        return simulate_current_clamp(
            selected_spec.neuron,
            list(selected_spec.pulse_trains),
            recording_site=recording_site or selected_spec.recording_site,
            selected_panels=selected_panels,
            morphology_name=selected_spec.morphology_name,
        )

    master_neuron = selected_spec.neuron
    step_count = _validate_simulation_inputs(master_neuron, list(selected_spec.pulse_trains))
    specs, sim, h = _import_backend()

    times_ms = [index * master_neuron.dt_ms for index in range(step_count + 1)]
    configured_command = [
        master_neuron.holding_current + command_delta(time_ms, selected_spec.pulse_trains)
        for time_ms in times_ms
    ]
    record_membrane_state = selected_panels is None or any(
        panel in selected_panels for panel in ("ionic_currents", "gating", "conductances")
    )

    net_params, ecl_by_id = _build_circuit_net_params(specs, neurons)
    sim_config = _build_sim_config(specs, master_neuron)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell_by_id = {str(cell.tags["pop"]): cell for cell in sim.net.cells}
        for neuron_spec in neurons:
            cell = cell_by_id[neuron_spec.neuron_id]
            _apply_section_reversal_potentials(cell, neuron_spec.neuron)

        selected_cell = cell_by_id[selected_neuron_id]
        effective_recording_site = recording_site or selected_spec.recording_site
        record_sec, record_x = _resolve_site(selected_cell, effective_recording_site)
        record_segment = record_sec(record_x)

        clamp_refs = []
        total_selected_current_nA = [0.0 for _ in times_ms]
        incoming_syn_current_vectors = []

        for neuron_spec in neurons:
            cell = cell_by_id[neuron_spec.neuron_id]
            hold_site = neuron_spec.recording_site or default_recording_site(neuron_spec.morphology_name)
            if neuron_spec.neuron_id == selected_neuron_id and recording_site is not None:
                hold_site = recording_site
            hold_sec, hold_x = _resolve_site(cell, hold_site)
            hold_segment = hold_sec(hold_x)
            holding_area_cm2 = hold_segment.area() * 1e-8
            holding_command_nA = [
                current_value_to_nA(neuron_spec.neuron.holding_current, neuron_spec.neuron.current_injection_unit, holding_area_cm2)
                for _ in times_ms
            ]
            if any(abs(value) > 1e-12 for value in holding_command_nA):
                clamp_refs.append(_attach_current_clamp_train(h, hold_sec, hold_x, times_ms, holding_command_nA))
            if neuron_spec.neuron_id == selected_neuron_id:
                total_selected_current_nA = [
                    total_selected_current_nA[index] + holding_command_nA[index]
                    for index in range(len(total_selected_current_nA))
                ]

            for train in neuron_spec.pulse_trains:
                if train.section_name not in cell.secs:
                    raise ValueError(f"{train.label}: selected section {train.section_name!r} does not exist.")
                stim_sec = cell.secs[train.section_name]["hObj"]
                stim_x = min(1.0, max(0.0, train.section_x))
                stim_segment = stim_sec(stim_x)
                stim_area_cm2 = stim_segment.area() * 1e-8
                stim_command_nA = [
                    current_value_to_nA(_command_delta_for_train(time_ms, train), neuron_spec.neuron.current_injection_unit, stim_area_cm2)
                    for time_ms in times_ms
                ]
                clamp_refs.append(_attach_current_clamp_train(h, stim_sec, stim_x, times_ms, stim_command_nA))
                if neuron_spec.neuron_id == selected_neuron_id:
                    total_selected_current_nA = [
                        total_selected_current_nA[index] + stim_command_nA[index]
                        for index in range(len(total_selected_current_nA))
                    ]

        synapse_refs = []
        for connection in connections:
            source_spec = neuron_by_id.get(connection.source_id)
            target_spec = neuron_by_id.get(connection.target_id)
            if source_spec is None or target_spec is None:
                continue

            source_cell = cell_by_id[connection.source_id]
            target_cell = cell_by_id[connection.target_id]
            pre_sec, pre_x = _resolve_site(source_cell, source_spec.output_site)
            post_sec, post_x = _resolve_site(target_cell, connection.target_site)

            syn = h.Exp2Syn(post_sec(post_x))
            syn.e = 0.0
            syn.tau1 = max(0.1, min(0.35, connection.pulse_width_ms / 6.0))
            syn.tau2 = max(syn.tau1 + 0.1, connection.pulse_width_ms)

            netcon = h.NetCon(pre_sec(pre_x)._ref_v, syn, sec=pre_sec)
            netcon.threshold = 0.0
            netcon.delay = max(0.0, connection.delay_ms)
            netcon.weight[0] = _simple_synaptic_weight_uS(connection.current_nA, target_spec.neuron.v_rest_mV)
            synapse_refs.append((syn, netcon))

            if connection.target_id == selected_neuron_id:
                incoming_syn_current_vectors.append(h.Vector().record(syn._ref_i))

        voltage_rec = h.Vector().record(record_segment._ref_v)
        time_rec = h.Vector().record(h._ref_t)
        hh_vectors = _record_hh_state(h, record_segment) if record_membrane_state else {}
        passive_vectors = _record_passive_state(h, record_segment) if record_membrane_state else {}

        sim.simulate()

    recorded_len = min(len(time_rec), len(voltage_rec), len(total_selected_current_nA), len(configured_command))
    synaptic_current_nA = [0.0 for _ in range(recorded_len)]
    for syn_vector in incoming_syn_current_vectors:
        for index in range(recorded_len):
            synaptic_current_nA[index] += -float(syn_vector[index])

    current_trace = [
        total_selected_current_nA[index] + synaptic_current_nA[index]
        for index in range(recorded_len)
    ]
    voltage_trace = [float(voltage_rec[index]) for index in range(recorded_len)]
    if record_membrane_state:
        ionic_currents, gating_variables, conductances = _extract_membrane_traces(
            hh_vectors,
            passive_vectors,
            voltage_trace,
            selected_spec.neuron,
            ecl_by_id[selected_neuron_id],
            recorded_len,
        )
    else:
        ionic_currents, gating_variables, conductances = {}, {}, {}

    return SimulationResult(
        mode=CURRENT_CLAMP,
        times_ms=[float(time_rec[index]) for index in range(recorded_len)],
        voltage_mV=voltage_trace,
        current_trace=current_trace,
        current_trace_label="Total Applied Current",
        current_trace_unit="nA",
        command_trace=configured_command[:recorded_len],
        command_trace_label=(
            "Configured Command Current"
            if selected_spec.neuron.current_injection_unit == CURRENT_UNIT_NA
            else "Configured Command Density"
        ),
        command_trace_unit=selected_spec.neuron.current_injection_unit,
        ionic_currents_uA_cm2=ionic_currents,
        gating_variables=gating_variables,
        conductances_mS_cm2=conductances,
        eleak_mV=selected_spec.neuron.eleak_mV,
        ecl_mV=ecl_by_id[selected_neuron_id],
    )


def _sweep_range(start_value: float, end_value: float, step_value: float, label: str) -> list[float]:
    if step_value == 0:
        raise ValueError(f"{label} step size must be non-zero.")
    direction = 1 if end_value >= start_value else -1
    if direction * step_value < 0:
        raise ValueError(f"{label} step size sign does not match the sweep direction.")

    values: list[float] = []
    current = start_value
    epsilon = abs(step_value) * 1e-6
    if direction > 0:
        while current <= end_value + epsilon:
            values.append(round(current, 6))
            current += step_value
    else:
        while current >= end_value - epsilon:
            values.append(round(current, 6))
            current += step_value
    return values


def _count_spikes_in_window(
    times_ms: list[float],
    voltage_mV: list[float],
    window_start_ms: float,
    window_end_ms: float,
    threshold_mV: float = 0.0,
) -> int:
    spike_count = 0
    previous_voltage = None
    for time_ms, current_voltage in zip(times_ms, voltage_mV):
        if not (window_start_ms <= time_ms < window_end_ms):
            previous_voltage = current_voltage
            continue
        if previous_voltage is not None and previous_voltage < threshold_mV <= current_voltage:
            spike_count += 1
        previous_voltage = current_voltage
    return spike_count


def _simulate_fi_step(args: tuple[NeuronConfig, FISweepConfig, MorphologySite | None, MorphologySite | None, float]) -> tuple[int, float]:
    neuron, config, stimulation_site, recording_site, current_step, morphology_name = args
    pulse_end_ms = config.pulse_start_ms + config.pulse_width_ms
    pulse_duration_s = config.pulse_width_ms / 1000.0
    current_unit = neuron.current_injection_unit
    stim_site = stimulation_site or default_recording_site(morphology_name)
    pulse = VoltagePulseTrain(
        label=f"Step to {current_step:.3g} {current_unit}",
        start_ms=config.pulse_start_ms,
        pulse_width_ms=config.pulse_width_ms,
        interval_ms=max(config.pulse_width_ms + 1.0, config.pulse_width_ms),
        pulse_count=1,
        amplitude=current_step,
        section_name=stim_site.section_name,
        section_x=stim_site.section_x,
    )
    result = simulate_current_clamp(neuron, [pulse], recording_site=recording_site, morphology_name=morphology_name)
    spike_count = _count_spikes_in_window(
        result.times_ms,
        result.voltage_mV,
        config.pulse_start_ms,
        pulse_end_ms,
    )
    firing_rate = spike_count / pulse_duration_s if pulse_duration_s > 0 else 0.0
    return spike_count, firing_rate


def simulate_iv_sweep(
    neuron: NeuronConfig,
    config: IVSweepConfig,
    morphology_name: str | None = None,
) -> IVSweepResult:
    if config.pulse_width_ms <= 0:
        raise ValueError("I-V pulse width must be positive.")
    if config.pulse_start_ms < 0:
        raise ValueError("I-V pulse start must be non-negative.")

    command_voltages = _sweep_range(config.start_mV, config.end_mV, config.step_mV, label="I-V")
    peak_inward: list[float] = []
    peak_outward: list[float] = []
    steady_state: list[float] = []

    for command_voltage in command_voltages:
        amplitude = command_voltage - neuron.holding_mV
        pulse = VoltagePulseTrain(
            label=f"Step to {command_voltage:.1f} mV",
            start_ms=config.pulse_start_ms,
            pulse_width_ms=config.pulse_width_ms,
            interval_ms=max(config.pulse_width_ms + 1.0, config.pulse_width_ms),
            pulse_count=1,
            amplitude=amplitude,
        )
        result = simulate_voltage_clamp(neuron, [pulse], morphology_name=morphology_name)

        in_pulse_currents = [
            current
            for time_ms, current in zip(result.times_ms, result.current_trace)
            if config.pulse_start_ms <= time_ms < config.pulse_start_ms + config.pulse_width_ms
        ]
        if not in_pulse_currents:
            raise RuntimeError("No current samples were captured inside the I-V pulse window.")

        steady_window_count = max(1, int(round(len(in_pulse_currents) * 0.2)))
        peak_inward.append(min(in_pulse_currents))
        peak_outward.append(max(in_pulse_currents))
        steady_state.append(sum(in_pulse_currents[-steady_window_count:]) / steady_window_count)

    return IVSweepResult(
        command_voltage_mV=command_voltages,
        peak_inward_current_nA=peak_inward,
        peak_outward_current_nA=peak_outward,
        steady_current_nA=steady_state,
        pulse_start_ms=config.pulse_start_ms,
        pulse_width_ms=config.pulse_width_ms,
    )


def simulate_fi_sweep(
    neuron: NeuronConfig,
    config: FISweepConfig,
    stimulation_site: MorphologySite | None = None,
    recording_site: MorphologySite | None = None,
    morphology_name: str | None = None,
) -> FISweepResult:
    if config.pulse_width_ms <= 0:
        raise ValueError("F-I pulse width must be positive.")
    if config.pulse_start_ms < 0:
        raise ValueError("F-I pulse start must be non-negative.")

    current_steps = _sweep_range(config.start_current, config.end_current, config.step_current, label="F-I")
    spike_counts: list[int] = []
    firing_rates: list[float] = []
    current_unit = neuron.current_injection_unit
    step_args = [(neuron, config, stimulation_site, recording_site, current_step, morphology_name) for current_step in current_steps]

    max_workers = min(len(current_steps), FI_MAX_WORKERS, os.cpu_count() or 1)
    if len(current_steps) >= FI_PARALLEL_MIN_SWEEP_POINTS and max_workers > 1:
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_simulate_fi_step, step_args))
        except Exception:
            results = [_simulate_fi_step(step_arg) for step_arg in step_args]
    else:
        results = [_simulate_fi_step(step_arg) for step_arg in step_args]

    for spike_count, firing_rate in results:
        spike_counts.append(spike_count)
        firing_rates.append(firing_rate)

    return FISweepResult(
        current_values=current_steps,
        current_unit=current_unit,
        spike_count=spike_counts,
        firing_rate_hz=firing_rates,
        pulse_start_ms=config.pulse_start_ms,
        pulse_width_ms=config.pulse_width_ms,
    )


def default_setup(mode: str = VOLTAGE_CLAMP, morphology_name: str | None = None) -> tuple[NeuronConfig, list[VoltagePulseTrain]]:
    default_site = default_recording_site(morphology_name)
    neuron = NeuronConfig(
        duration_ms=DEFAULT_DURATION_MS,
        dt_ms=DEFAULT_DT_MS,
        v_rest_mV=DEFAULT_VREST_MV,
        holding_mV=DEFAULT_VCLAMP_HOLDING_MV,
        current_injection_unit=CURRENT_UNIT_NA,
        holding_current=0.0,
        cm_uF_cm2=DEFAULT_CM_UF_CM2,
        gna_mS_cm2=DEFAULT_GNA_MS_CM2,
        gk_mS_cm2=DEFAULT_GK_MS_CM2,
        gl_mS_cm2=DEFAULT_GL_MS_CM2,
        ena_mV=DEFAULT_ENA_MV,
        ek_mV=DEFAULT_EK_MV,
        eleak_mV=DEFAULT_ELEAK_MV,
        gcl_mS_cm2=DEFAULT_GCL_MS_CM2,
        gshunt_mS_cm2=DEFAULT_GSHUNT_MS_CM2,
        ecl_mV=DEFAULT_ECL_MV,
    )
    if mode == CURRENT_CLAMP:
        return (
            neuron,
            [
                VoltagePulseTrain(
                    label="Example AP Step",
                    start_ms=DEFAULT_CURRENT_STEP_START_MS,
                    pulse_width_ms=DEFAULT_CURRENT_STEP_WIDTH_MS,
                    interval_ms=DEFAULT_CURRENT_STEP_WIDTH_MS + 5.0,
                    pulse_count=1,
                    amplitude=DEFAULT_CURRENT_STEP_NA,
                    section_name=default_site.section_name,
                    section_x=default_site.section_x,
                )
            ],
        )
    return (
        neuron,
        [
                VoltagePulseTrain(
                    label="Voltage Step",
                    start_ms=DEFAULT_VCLAMP_START_MS,
                    pulse_width_ms=DEFAULT_VCLAMP_WIDTH_MS,
                    interval_ms=DEFAULT_VCLAMP_WIDTH_MS + 5.0,
                    pulse_count=1,
                    amplitude=DEFAULT_VCLAMP_TEST_MV - DEFAULT_VCLAMP_HOLDING_MV,
                    section_name=default_site.section_name,
                    section_x=default_site.section_x,
                )
            ],
    )
