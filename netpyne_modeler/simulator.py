from __future__ import annotations

import contextlib
import copy
import concurrent.futures
import hashlib
import io
import math
import os
import re
import shutil
import subprocess
import sys
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
    v_rest_mV: float = -72.0
    holding_mV: float = -72.0
    current_injection_unit: str = CURRENT_UNIT_NA
    holding_current: float = 0.0
    cm_uF_cm2: float = 1.0
    gna_mS_cm2: float = 600.0
    gk_mS_cm2: float = 120.0
    gl_mS_cm2: float = 0.03
    ena_mV: float = 50.0
    ek_mV: float = -85.0
    eleak_mV: float = -90.0
    gcl_mS_cm2: float = 0.0
    gshunt_mS_cm2: float = 0.0
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


@dataclass(frozen=True, slots=True)
class RegionBiophysics:
    region_name: str
    cm_uF_cm2: float
    g_pas_base_mS_cm2: float
    gcl_mS_cm2: float
    gshunt_mS_cm2: float
    e_pas_mV: float
    ecl_mV: float
    gna_mS_cm2: float
    gk_mS_cm2: float
    hh_gl_mS_cm2: float
    optional_mechanisms: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()


@dataclass(slots=True)
class PythonIhSegment:
    segment: object
    clamp: object
    area_cm2: float
    gbar_S_cm2: float
    ehcn_mV: float
    m: float = 0.0
    gIh_S_cm2: float = 0.0
    ih_mA_cm2: float = 0.0
    record: bool = False


@dataclass(slots=True)
class PythonIhController:
    h: object
    segments: list[PythonIhSegment]
    currents_mA_cm2: list[float] = field(default_factory=list)
    conductances_S_cm2: list[float] = field(default_factory=list)
    gating_m: list[float] = field(default_factory=list)


@dataclass(slots=True)
class PythonKASegment:
    segment: object
    clamp: object
    area_cm2: float
    gbar_S_cm2: float
    ek_mV: float
    m: float = 0.0
    h: float = 1.0
    gKA_S_cm2: float = 0.0
    ika_mA_cm2: float = 0.0
    record: bool = False


@dataclass(slots=True)
class PythonKAController:
    h: object
    segments: list[PythonKASegment]
    currents_mA_cm2: list[float] = field(default_factory=list)
    conductances_S_cm2: list[float] = field(default_factory=list)
    gating_m: list[float] = field(default_factory=list)
    gating_h: list[float] = field(default_factory=list)


@dataclass(slots=True)
class PythonCaLVASegment:
    segment: object
    clamp: object
    area_cm2: float
    gbar_S_cm2: float
    cao_mM: float
    cai_rest_mM: float
    decay_ms: float
    influx_scale_mM_per_ms_per_mA_cm2: float
    cai_mM: float
    m: float = 0.0
    h: float = 1.0
    gCaLVA_S_cm2: float = 0.0
    ica_mA_cm2: float = 0.0
    eca_mV: float = 120.0
    record: bool = False


@dataclass(slots=True)
class PythonCaLVAController:
    h: object
    segments: list[PythonCaLVASegment]
    currents_mA_cm2: list[float] = field(default_factory=list)
    conductances_S_cm2: list[float] = field(default_factory=list)
    gating_m: list[float] = field(default_factory=list)
    gating_h: list[float] = field(default_factory=list)
    cai_mM: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SegmentStateSnapshot:
    section_name: str
    segment_x: float
    values: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class ControllerStateSnapshot:
    section_name: str
    segment_x: float
    values: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class SteadyStateInitializationSnapshot:
    segment_states: tuple[SegmentStateSnapshot, ...]
    ih_states: tuple[ControllerStateSnapshot, ...] = ()
    ka_states: tuple[ControllerStateSnapshot, ...] = ()
    calva_states: tuple[ControllerStateSnapshot, ...] = ()


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

DEFAULT_AXIAL_RESISTANCE_OHM_CM = 100.0
DEFAULT_SOMA_AXON_CM_UF_CM2 = 1.0
DEFAULT_DENDRITE_CM_UF_CM2 = 2.0
DEFAULT_CM_UF_CM2 = DEFAULT_SOMA_AXON_CM_UF_CM2
DEFAULT_GNA_MS_CM2 = 600.0
DEFAULT_GK_MS_CM2 = 120.0
DEFAULT_GL_MS_CM2 = 0.03
DEFAULT_ENA_MV = 50.0
DEFAULT_EK_MV = -85.0
DEFAULT_ELEAK_MV = -90.0
DEFAULT_GCL_MS_CM2 = 0.0
DEFAULT_GSHUNT_MS_CM2 = 0.0
DEFAULT_CLI_MM = 7.0
DEFAULT_CLO_MM = 130.0
DEFAULT_ECL_MV = -70.35603740607618
DEFAULT_VREST_MV = -72.0
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
DEFAULT_CELSIUS_C = 34.0
TARGET_SEGMENT_LENGTH_UM = 40.0
GAS_CONSTANT_J_MOL_K = 8.314462618
FARADAY_C_MOL = 96485.33212
FI_PARALLEL_MIN_SWEEP_POINTS = 4
FI_MAX_WORKERS = 4
FI_PROCESS_POOL_ENV = "NETPYNE_MODELER_ENABLE_FI_PROCESS_POOL"
DEFAULT_CONNECTION_CURRENT_NA = 1.0
DEFAULT_CONNECTION_PULSE_MS = 1.5
REGION_AIS = "ais"
REGION_DISTAL_AXON = "distal_axon"
REGION_SOMA = "soma"
REGION_BASAL_DEND = "basal_dend"
REGION_APICAL_DEND = "apical_dend"
REGION_APICAL_HOTZONE = "apical_hotzone"
REGION_TUFT = "tuft"
AIS_LENGTH_UM = 35.0
AIS_GNABAR_S_CM2 = 2.4
AIS_GKBAR_S_CM2 = 0.216
DISTAL_AXON_GNABAR_S_CM2 = 1.2
DISTAL_AXON_GKBAR_S_CM2 = 0.156
SOMA_GNABAR_S_CM2 = 0.60
SOMA_GKBAR_S_CM2 = 0.12
BASAL_DEND_GNABAR_S_CM2 = 0.035
BASAL_DEND_GKBAR_S_CM2 = 0.0144
APICAL_DEND_GNABAR_S_CM2 = 0.070
APICAL_DEND_GKBAR_S_CM2 = 0.024
APICAL_HOTZONE_START_UM = 500.0
APICAL_HOTZONE_END_UM = 850.0
APICAL_TUFT_START_UM = APICAL_HOTZONE_END_UM
APICAL_HOTZONE_GNABAR_S_CM2 = 0.098
APICAL_HOTZONE_GKBAR_S_CM2 = 0.0264
BASAL_GP_AS_SCALE = 2.0
APICAL_GP_AS_SCALE = 2.0
AXON_GP_AS_SCALE = 1.0
SOMA_GP_AS_SCALE = 1.0
APICAL_IH_PROXIMAL_S_CM2 = 5.0e-5
APICAL_IH_DISTAL_S_CM2 = 6.0e-4
APICAL_IH_GRADIENT_END_UM = 1000.0
IH_REVERSAL_MV = -45.0
DEND_KA_PROXIMAL_S_CM2 = 1.0e-3
DEND_KA_DISTAL_S_CM2 = 3.0e-3
BASAL_KA_GRADIENT_END_UM = 400.0
APICAL_KA_PROXIMAL_S_CM2 = 8.0e-3
APICAL_KA_DISTAL_S_CM2 = 3.0e-2
APICAL_KA_GRADIENT_END_UM = 1000.0
APICAL_HOTZONE_CALVA_S_CM2 = 1.5e-3
APICAL_HOTZONE_CALVA_CAI_REST_MM = 5.0e-5
APICAL_HOTZONE_CALVA_CAO_MM = 2.0
APICAL_HOTZONE_CALVA_DECAY_MS = 40.0
APICAL_HOTZONE_CALVA_INFLUX_SCALE = 1.5e-4
APICAL_CA_HVA_S_CM2 = 2.0e-4
APICAL_HOTZONE_CA_HVA_S_CM2 = 3.8e-3
CAHVA_M_VHALF_MV = -55.0
CAHVA_M_SLOPE_MV = 6.0
CAHVA_M_TAU_MS = 1.0
CAHVA_H_VHALF_MV = -60.0
CAHVA_H_SLOPE_MV = -7.0
CAHVA_H_TAU_MS = 80.0
AIS_NAP_S_CM2 = 8.0e-3
DISTAL_AXON_NAP_S_CM2 = 2.0e-3
SOMA_NAP_S_CM2 = 4.0e-4
NAP_VHALF_MV = -55.0
NAP_SLOPE_MV = 4.5
NAP_TAU_MS = 1.0
SOMA_IM_S_CM2 = 8.0e-5
APICAL_IM_S_CM2 = 7.0e-4
IM_VHALF_MV = -35.0
IM_SLOPE_MV = 10.0
IM_TAU_MS = 40.0
HOTZONE_SK_E2_S_CM2 = 8.0e-4
HOTZONE_SK_E2_KD_MM = 3.5e-4
HOTZONE_SK_E2_HILL_POWER = 4.0
HOTZONE_SK_E2_TAU_MS = 5.0
HOTZONE_CADYNAMICS_GAMMA = 6.4e-4
CALVA_M_VHALF_MV = -57.0
CALVA_M_SLOPE_MV = 6.2
CALVA_M_TAU_BASE_MS = 0.5
CALVA_M_TAU_SCALE_MS = 2.0
CALVA_M_TAU_VHALF_MV = -40.0
CALVA_M_TAU_SLOPE_MV = 10.0
CALVA_H_VHALF_MV = -81.0
CALVA_H_SLOPE_MV = -4.0
CALVA_H_TAU_BASE_MS = 8.0
CALVA_H_TAU_SCALE_MS = 25.0
CALVA_H_TAU_VHALF_MV = -50.0
CALVA_H_TAU_SLOPE_MV = 7.0
OPTIONAL_MECHANISM_NAMES = (
    "Ih",
    "KA",
    "Im",
    "NaTa_t",
    "NaTs2_t",
    "Nap_Et2",
    "K_Tst",
    "K_Pst",
    "SKv3_1",
    "Ca_HVA",
    "Ca_LVAst",
    "CaDynamics_E2",
    "SK_E2",
)
STEADY_STATE_INITIALIZATION_DURATION_MS = 1000.0
STEADY_STATE_SEGMENT_ATTRIBUTES = (
    "v",
    "m_hh",
    "h_hh",
    "n_hh",
    "m_Ih",
    "m_KA",
    "h_KA",
    "m_Ca_LVAst",
    "h_Ca_LVAst",
    "m_Ca_HVA",
    "h_Ca_HVA",
    "m_Nap_Et2",
    "p_Im",
    "z_SK_E2",
    "cai",
)
STEADY_STATE_MODEL_SIGNATURE_NAMES = (
    "DEFAULT_AXIAL_RESISTANCE_OHM_CM",
    "DEFAULT_SOMA_AXON_CM_UF_CM2",
    "DEFAULT_DENDRITE_CM_UF_CM2",
    "DEFAULT_CELSIUS_C",
    "AIS_LENGTH_UM",
    "AIS_GNABAR_S_CM2",
    "AIS_GKBAR_S_CM2",
    "DISTAL_AXON_GNABAR_S_CM2",
    "DISTAL_AXON_GKBAR_S_CM2",
    "SOMA_GNABAR_S_CM2",
    "SOMA_GKBAR_S_CM2",
    "BASAL_DEND_GNABAR_S_CM2",
    "BASAL_DEND_GKBAR_S_CM2",
    "APICAL_DEND_GNABAR_S_CM2",
    "APICAL_DEND_GKBAR_S_CM2",
    "APICAL_HOTZONE_GNABAR_S_CM2",
    "APICAL_HOTZONE_GKBAR_S_CM2",
    "BASAL_GP_AS_SCALE",
    "APICAL_GP_AS_SCALE",
    "AXON_GP_AS_SCALE",
    "SOMA_GP_AS_SCALE",
    "APICAL_IH_PROXIMAL_S_CM2",
    "APICAL_IH_DISTAL_S_CM2",
    "APICAL_IH_GRADIENT_END_UM",
    "IH_REVERSAL_MV",
    "DEND_KA_PROXIMAL_S_CM2",
    "DEND_KA_DISTAL_S_CM2",
    "BASAL_KA_GRADIENT_END_UM",
    "APICAL_KA_PROXIMAL_S_CM2",
    "APICAL_KA_DISTAL_S_CM2",
    "APICAL_KA_GRADIENT_END_UM",
    "APICAL_HOTZONE_CALVA_S_CM2",
    "APICAL_HOTZONE_CALVA_CAI_REST_MM",
    "APICAL_HOTZONE_CALVA_CAO_MM",
    "APICAL_HOTZONE_CALVA_DECAY_MS",
    "APICAL_HOTZONE_CALVA_INFLUX_SCALE",
    "APICAL_CA_HVA_S_CM2",
    "APICAL_HOTZONE_CA_HVA_S_CM2",
    "CAHVA_M_VHALF_MV",
    "CAHVA_M_SLOPE_MV",
    "CAHVA_M_TAU_MS",
    "CAHVA_H_VHALF_MV",
    "CAHVA_H_SLOPE_MV",
    "CAHVA_H_TAU_MS",
    "AIS_NAP_S_CM2",
    "DISTAL_AXON_NAP_S_CM2",
    "SOMA_NAP_S_CM2",
    "NAP_VHALF_MV",
    "NAP_SLOPE_MV",
    "NAP_TAU_MS",
    "SOMA_IM_S_CM2",
    "APICAL_IM_S_CM2",
    "IM_VHALF_MV",
    "IM_SLOPE_MV",
    "IM_TAU_MS",
    "HOTZONE_SK_E2_S_CM2",
    "HOTZONE_SK_E2_KD_MM",
    "HOTZONE_SK_E2_HILL_POWER",
    "HOTZONE_SK_E2_TAU_MS",
    "HOTZONE_CADYNAMICS_GAMMA",
    "CALVA_M_VHALF_MV",
    "CALVA_M_SLOPE_MV",
    "CALVA_M_TAU_BASE_MS",
    "CALVA_M_TAU_SCALE_MS",
    "CALVA_M_TAU_VHALF_MV",
    "CALVA_M_TAU_SLOPE_MV",
    "CALVA_H_VHALF_MV",
    "CALVA_H_SLOPE_MV",
    "CALVA_H_TAU_BASE_MS",
    "CALVA_H_TAU_SCALE_MS",
    "CALVA_H_TAU_VHALF_MV",
    "CALVA_H_TAU_SLOPE_MV",
)
PROJECT_MOD_DIR = PROJECT_ROOT / "mod"
ENABLE_OPTIONAL_NMODL_LOADING = True


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


def _mechanism_build_root() -> Path:
    build_root = Path(tempfile.gettempdir()) / "netpyne_modeler_runtime" / "mechanisms"
    build_root.mkdir(parents=True, exist_ok=True)
    return build_root


def _symlinked_venv_root() -> Path | None:
    # Keep the virtualenv path itself instead of resolving through the interpreter
    # symlink into the base Python install; the mechanism toolchain/resources live
    # under this environment's site-packages tree.
    venv_root = Path(sys.executable).parent.parent
    if " " not in str(venv_root):
        return venv_root
    link_root = Path(tempfile.gettempdir()) / "netpyne_modeler_venv_link"
    if link_root.exists() or link_root.is_symlink():
        if link_root.resolve() == venv_root:
            return link_root
        link_root.unlink()
    link_root.symlink_to(venv_root, target_is_directory=True)
    return link_root


def _source_tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    digest.update(b"moddir-v3")
    digest.update(sys.executable.encode("utf-8"))
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _compiled_mechanism_library(build_dir: Path) -> Path | None:
    preferred_order = (".dylib", ".so", ".dll")
    candidates = sorted(build_dir.rglob("libnrnmech*")) + sorted(build_dir.rglob("nrnmech*"))
    for suffix in preferred_order:
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix == suffix:
                return candidate
    return None


def _compile_optional_mechanisms() -> Path | None:
    if not PROJECT_MOD_DIR.exists() or not any(PROJECT_MOD_DIR.glob("*.mod")):
        return None
    build_root = _mechanism_build_root()
    source_hash = _source_tree_hash(PROJECT_MOD_DIR)
    build_dir = build_root / source_hash
    library_path = _compiled_mechanism_library(build_dir)
    if library_path is not None:
        return library_path

    venv_root = _symlinked_venv_root()
    preferred_venv_root = venv_root if venv_root is not None else Path(sys.executable).parent.parent
    preferred_nrn_home = preferred_venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "neuron" / ".data"
    preferred_site_packages = preferred_venv_root / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    preferred_compile = preferred_nrn_home / "bin" / "nrnivmodl"
    preferred_nmodl = preferred_nrn_home / "bin" / "nmodl"
    if preferred_compile.exists():
        compile_cmd = [str(preferred_compile)]
        if preferred_nmodl.exists():
            compile_cmd.extend(["-nmodl", str(preferred_nmodl)])
    else:
        compile_executable = shutil.which("nrnivmodl")
        compile_cmd = [compile_executable] if compile_executable else None
    if compile_cmd is None:
        return None
    compile_env = os.environ.copy()
    tmp_dir = build_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    compile_env["TMPDIR"] = str(tmp_dir)
    compile_env["NRNHOME"] = str(preferred_nrn_home)
    compile_env["NMODLHOME"] = str(preferred_nrn_home)
    python_path_parts = [str(preferred_site_packages)]
    existing_python_path = compile_env.get("PYTHONPATH", "").strip()
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    compile_env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    try:
        from find_libpython import find_libpython
    except ImportError:
        find_libpython = None
    if find_libpython is not None:
        try:
            compile_env["NMODL_PYLIB"] = find_libpython()
        except Exception:
            pass

    working_dir = Path(tempfile.mkdtemp(prefix=f"mechbuild_{source_hash}_"))
    source_dir = working_dir / "mod"
    source_dir.mkdir(parents=True, exist_ok=True)
    for mod_file in PROJECT_MOD_DIR.glob("*.mod"):
        shutil.copy2(mod_file, source_dir / mod_file.name)

    tmp_dir = working_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    compile_env["TMPDIR"] = str(tmp_dir)

    mod_inputs = [str(path) for path in sorted(source_dir.glob("*.mod"))]
    compile_result = subprocess.run(
        [*compile_cmd, *mod_inputs],
        cwd=working_dir,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=compile_env,
    )
    built_library_path = _compiled_mechanism_library(working_dir)
    if built_library_path is not None:
        target_dir = build_dir / "arm64"
        target_dir.mkdir(parents=True, exist_ok=True)
        cached_library_path = target_dir / built_library_path.name
        shutil.copy2(built_library_path, cached_library_path)
        return cached_library_path
    stdout_tail = compile_result.stdout[-4000:] if compile_result.stdout else ""
    stderr_tail = compile_result.stderr[-4000:] if compile_result.stderr else ""
    raise RuntimeError(
        "Optional mechanism compilation failed.\n"
        f"Command: {' '.join([*compile_cmd, *mod_inputs])}\n"
        f"Return code: {compile_result.returncode}\n"
        f"stdout tail:\n{stdout_tail}\n"
        f"stderr tail:\n{stderr_tail}"
    )


@lru_cache(maxsize=1)
def _ensure_optional_mechanisms_loaded() -> tuple[str, ...]:
    if not ENABLE_OPTIONAL_NMODL_LOADING:
        return ()
    _configure_runtime_env()
    try:
        from neuron import h
    except ImportError:
        return ()

    library_path = None
    try:
        library_path = _compile_optional_mechanisms()
    except Exception:
        library_path = None
    if library_path is not None:
        try:
            h.nrn_load_dll(str(library_path))
        except Exception:
            pass

    available: list[str] = []
    temp_sec = h.Section(name="mech_probe")
    for mechanism_name in OPTIONAL_MECHANISM_NAMES:
        try:
            temp_sec.insert(mechanism_name)
        except Exception:
            continue
        available.append(mechanism_name)
    return tuple(sorted(set(available)))


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
    _ensure_optional_mechanisms_loaded()
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


def _normalize_section_name(raw_name: str) -> str:
    base_name = raw_name.split(".")[-1]
    return re.sub(r"\[(\d+)\]", r"_\1", base_name)


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


def _section_base_cm(section_type: str) -> float:
    if section_type in {"dend", "apic"}:
        return DEFAULT_DENDRITE_CM_UF_CM2
    return DEFAULT_SOMA_AXON_CM_UF_CM2


def _segment_distance_um(h, soma_section, sec, x: float) -> float:
    h.distance(0.0, 0.5, sec=soma_section)
    return float(h.distance(x, sec=sec))


def _region_name_for_segment(section_type: str, distance_um: float) -> str:
    if section_type == "axon":
        return REGION_AIS if distance_um <= AIS_LENGTH_UM else REGION_DISTAL_AXON
    if section_type == "soma":
        return REGION_SOMA
    if section_type == "apic":
        if distance_um > APICAL_TUFT_START_UM:
            return REGION_TUFT
        if APICAL_HOTZONE_START_UM <= distance_um <= APICAL_HOTZONE_END_UM:
            return REGION_APICAL_HOTZONE
        return REGION_APICAL_DEND
    return REGION_BASAL_DEND


def _supports_hotzone_calcium_dynamics(available_mechanisms: tuple[str, ...]) -> bool:
    return "CaDynamics_E2" in set(available_mechanisms)


def _effective_optional_mechanisms(available_mechanisms: tuple[str, ...]) -> tuple[str, ...]:
    if not available_mechanisms:
        return ()
    return tuple(sorted(set(available_mechanisms)))


def _optional_mechanism_zero_settings(mechanism_name: str) -> tuple[tuple[str, float], ...]:
    zero_settings = {
        "Ih": (("gIhbar_Ih", 0.0), ("ehcn_Ih", IH_REVERSAL_MV)),
        "KA": (("gKAbar_KA", 0.0),),
        "Im": (("gImbar_Im", 0.0),),
        "Nap_Et2": (("gNap_Et2bar_Nap_Et2", 0.0),),
        "NaTa_t": (("gNaTa_tbar_NaTa_t", 0.0),),
        "NaTs2_t": (("gNaTs2_tbar_NaTs2_t", 0.0),),
        "K_Tst": (("gK_Tstbar_K_Tst", 0.0),),
        "K_Pst": (("gK_Pstbar_K_Pst", 0.0),),
        "SKv3_1": (("gSKv3_1bar_SKv3_1", 0.0),),
        "Ca_HVA": (("gCa_HVAbar_Ca_HVA", 0.0),),
        "Ca_LVAst": (("gCa_LVAstbar_Ca_LVAst", 0.0),),
        "CaDynamics_E2": (
            ("gamma_CaDynamics_E2", 0.0),
            ("decay_CaDynamics_E2", APICAL_HOTZONE_CALVA_DECAY_MS),
            ("minCai_CaDynamics_E2", 1.0e-4),
        ),
        "SK_E2": (("gSK_E2bar_SK_E2", 0.0),),
    }
    return zero_settings.get(mechanism_name, ())


def _apply_segment_parameter_pairs(segment, parameter_pairs: tuple[tuple[str, float], ...]) -> None:
    for parameter_name, value in parameter_pairs:
        if hasattr(segment, parameter_name):
            setattr(segment, parameter_name, value)


def _configure_optional_mechanism_globals(h) -> None:
    global_settings = {
        "vhalf_m_Ca_LVAst": CALVA_M_VHALF_MV,
        "k_m_Ca_LVAst": CALVA_M_SLOPE_MV,
        "vhalf_h_Ca_LVAst": CALVA_H_VHALF_MV,
        "k_h_Ca_LVAst": CALVA_H_SLOPE_MV,
        "tau_m_base_Ca_LVAst": CALVA_M_TAU_BASE_MS,
        "tau_m_scale_Ca_LVAst": CALVA_M_TAU_SCALE_MS,
        "tau_m_vhalf_Ca_LVAst": CALVA_M_TAU_VHALF_MV,
        "tau_m_slope_Ca_LVAst": CALVA_M_TAU_SLOPE_MV,
        "tau_h_base_Ca_LVAst": CALVA_H_TAU_BASE_MS,
        "tau_h_scale_Ca_LVAst": CALVA_H_TAU_SCALE_MS,
        "tau_h_vhalf_Ca_LVAst": CALVA_H_TAU_VHALF_MV,
        "tau_h_slope_Ca_LVAst": CALVA_H_TAU_SLOPE_MV,
        "vhalf_m_Ca_HVA": CAHVA_M_VHALF_MV,
        "k_m_Ca_HVA": CAHVA_M_SLOPE_MV,
        "tau_m_Ca_HVA": CAHVA_M_TAU_MS,
        "vhalf_h_Ca_HVA": CAHVA_H_VHALF_MV,
        "k_h_Ca_HVA": CAHVA_H_SLOPE_MV,
        "tau_h_Ca_HVA": CAHVA_H_TAU_MS,
        "vhalf_Nap_Et2": NAP_VHALF_MV,
        "k_Nap_Et2": NAP_SLOPE_MV,
        "tau_Nap_Et2": NAP_TAU_MS,
        "vhalf_Im": IM_VHALF_MV,
        "k_Im": IM_SLOPE_MV,
        "tau_Im": IM_TAU_MS,
        "Kd_SK_E2": HOTZONE_SK_E2_KD_MM,
        "hill_power_SK_E2": HOTZONE_SK_E2_HILL_POWER,
        "tau_SK_E2": HOTZONE_SK_E2_TAU_MS,
    }
    for parameter_name, value in global_settings.items():
        if hasattr(h, parameter_name):
            setattr(h, parameter_name, value)


def _optional_mechanism_settings(
    region_name: str,
    distance_um: float,
    available_mechanisms: tuple[str, ...],
) -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    if not available_mechanisms:
        return ()
    available = set(available_mechanisms)
    mechanism_map: dict[str, dict[str, float]] = {}

    if "Ih" in available and region_name in {REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        normalized_distance = max(0.0, min(1.0, distance_um / APICAL_IH_GRADIENT_END_UM))
        g_ih_bar = APICAL_IH_PROXIMAL_S_CM2 + (APICAL_IH_DISTAL_S_CM2 - APICAL_IH_PROXIMAL_S_CM2) * normalized_distance
        mechanism_map["Ih"] = {
            "gIhbar_Ih": g_ih_bar,
            "ehcn_Ih": IH_REVERSAL_MV,
        }

    if "KA" in available and region_name in {REGION_BASAL_DEND, REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        gka_bar = _ka_settings_for_region(region_name, distance_um)
        if gka_bar is not None:
            mechanism_map["KA"] = {"gKAbar_KA": gka_bar}

    if "Im" in available and region_name in {REGION_SOMA, REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        mechanism_map["Im"] = {
            "gImbar_Im": SOMA_IM_S_CM2 if region_name == REGION_SOMA else APICAL_IM_S_CM2
        }

    if "Nap_Et2" in available and region_name in {REGION_AIS, REGION_DISTAL_AXON, REGION_SOMA}:
        mechanism_map["Nap_Et2"] = {
            "gNap_Et2bar_Nap_Et2": (
                AIS_NAP_S_CM2 if region_name == REGION_AIS
                else DISTAL_AXON_NAP_S_CM2 if region_name == REGION_DISTAL_AXON
                else SOMA_NAP_S_CM2
            )
        }

    if "NaTa_t" in available and region_name in {REGION_AIS, REGION_DISTAL_AXON}:
        mechanism_map["NaTa_t"] = {"gNaTa_tbar_NaTa_t": 3.5 if region_name == REGION_AIS else 1.5}

    if "NaTs2_t" in available and region_name in {REGION_SOMA, REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT, REGION_BASAL_DEND}:
        gbar = 0.9 if region_name == REGION_SOMA else 0.02 if region_name in {REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT} else 0.005
        mechanism_map["NaTs2_t"] = {"gNaTs2_tbar_NaTs2_t": gbar}

    if "K_Tst" in available and region_name in {REGION_AIS, REGION_DISTAL_AXON}:
        mechanism_map["K_Tst"] = {"gK_Tstbar_K_Tst": 0.07 if region_name == REGION_AIS else 0.05}

    if "K_Pst" in available and region_name in {REGION_AIS, REGION_DISTAL_AXON}:
        mechanism_map["K_Pst"] = {"gK_Pstbar_K_Pst": 0.18 if region_name == REGION_AIS else 0.12}

    if "SKv3_1" in available and region_name in {REGION_AIS, REGION_DISTAL_AXON, REGION_SOMA, REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        if region_name == REGION_AIS:
            gbar = 0.45
        elif region_name == REGION_DISTAL_AXON:
            gbar = 0.25
        elif region_name == REGION_SOMA:
            gbar = 0.34
        else:
            gbar = 0.0018
        mechanism_map["SKv3_1"] = {"gSKv3_1bar_SKv3_1": gbar}

    if "Ca_HVA" in available and region_name in {REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        mechanism_map["Ca_HVA"] = {
            "gCa_HVAbar_Ca_HVA": APICAL_HOTZONE_CA_HVA_S_CM2 if region_name == REGION_APICAL_HOTZONE else APICAL_CA_HVA_S_CM2
        }

    if region_name == REGION_APICAL_HOTZONE:
        if "Ca_LVAst" in available:
            mechanism_map["Ca_LVAst"] = {"gCa_LVAstbar_Ca_LVAst": APICAL_HOTZONE_CALVA_S_CM2}
        if _supports_hotzone_calcium_dynamics(available_mechanisms):
            mechanism_map["CaDynamics_E2"] = {
                "gamma_CaDynamics_E2": HOTZONE_CADYNAMICS_GAMMA,
                "decay_CaDynamics_E2": APICAL_HOTZONE_CALVA_DECAY_MS,
                "minCai_CaDynamics_E2": APICAL_HOTZONE_CALVA_CAI_REST_MM,
            }
        if {"SK_E2", "CaDynamics_E2"} <= available:
            mechanism_map["SK_E2"] = {"gSK_E2bar_SK_E2": HOTZONE_SK_E2_S_CM2}

    return tuple(
        (mechanism_name, tuple(sorted(parameters.items())))
        for mechanism_name, parameters in sorted(mechanism_map.items())
    )


def _ih_settings_for_region(region_name: str, distance_um: float) -> tuple[float, float] | None:
    if region_name not in {REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        return None
    normalized_distance = max(0.0, min(1.0, distance_um / APICAL_IH_GRADIENT_END_UM))
    g_ih_bar = APICAL_IH_PROXIMAL_S_CM2 + (APICAL_IH_DISTAL_S_CM2 - APICAL_IH_PROXIMAL_S_CM2) * normalized_distance
    return g_ih_bar, IH_REVERSAL_MV


def _ka_settings_for_region(region_name: str, distance_um: float) -> float | None:
    if region_name == REGION_BASAL_DEND:
        normalized_distance = max(0.0, min(1.0, distance_um / BASAL_KA_GRADIENT_END_UM))
        return DEND_KA_PROXIMAL_S_CM2 + (DEND_KA_DISTAL_S_CM2 - DEND_KA_PROXIMAL_S_CM2) * normalized_distance
    if region_name in {REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        normalized_distance = max(0.0, min(1.0, distance_um / APICAL_KA_GRADIENT_END_UM))
        return APICAL_KA_PROXIMAL_S_CM2 + (APICAL_KA_DISTAL_S_CM2 - APICAL_KA_PROXIMAL_S_CM2) * normalized_distance
    return None


def _calva_settings_for_region(region_name: str) -> tuple[float, float, float, float] | None:
    if region_name != REGION_APICAL_HOTZONE:
        return None
    return (
        APICAL_HOTZONE_CALVA_S_CM2,
        APICAL_HOTZONE_CALVA_CAI_REST_MM,
        APICAL_HOTZONE_CALVA_CAO_MM,
        APICAL_HOTZONE_CALVA_DECAY_MS,
    )


def _segment_region_biophysics(
    neuron: NeuronConfig,
    section_type: str,
    distance_um: float,
    available_mechanisms: tuple[str, ...] = (),
) -> RegionBiophysics:
    base_g_pas = neuron.gl_mS_cm2
    gcl = neuron.gcl_mS_cm2
    gshunt = neuron.gshunt_mS_cm2
    region_name = _region_name_for_segment(section_type, distance_um)

    if region_name == REGION_AIS:
        gna = AIS_GNABAR_S_CM2 * 1000.0
        gk = AIS_GKBAR_S_CM2 * 1000.0
        cm = DEFAULT_SOMA_AXON_CM_UF_CM2
        g_pas = base_g_pas * AXON_GP_AS_SCALE
    elif region_name == REGION_DISTAL_AXON:
        gna = DISTAL_AXON_GNABAR_S_CM2 * 1000.0
        gk = DISTAL_AXON_GKBAR_S_CM2 * 1000.0
        cm = DEFAULT_SOMA_AXON_CM_UF_CM2
        g_pas = base_g_pas * AXON_GP_AS_SCALE
    elif region_name == REGION_SOMA:
        gna = SOMA_GNABAR_S_CM2 * 1000.0
        gk = SOMA_GKBAR_S_CM2 * 1000.0
        cm = DEFAULT_SOMA_AXON_CM_UF_CM2
        g_pas = base_g_pas * SOMA_GP_AS_SCALE
    elif region_name in {REGION_APICAL_DEND, REGION_APICAL_HOTZONE, REGION_TUFT}:
        gna = (APICAL_HOTZONE_GNABAR_S_CM2 if region_name == REGION_APICAL_HOTZONE else APICAL_DEND_GNABAR_S_CM2) * 1000.0
        gk = (APICAL_HOTZONE_GKBAR_S_CM2 if region_name == REGION_APICAL_HOTZONE else APICAL_DEND_GKBAR_S_CM2) * 1000.0
        cm = DEFAULT_DENDRITE_CM_UF_CM2
        g_pas = base_g_pas * APICAL_GP_AS_SCALE
    else:
        gna = BASAL_DEND_GNABAR_S_CM2 * 1000.0
        gk = BASAL_DEND_GKBAR_S_CM2 * 1000.0
        cm = DEFAULT_DENDRITE_CM_UF_CM2
        g_pas = base_g_pas * BASAL_GP_AS_SCALE

    return RegionBiophysics(
        region_name=region_name,
        cm_uF_cm2=cm,
        g_pas_base_mS_cm2=g_pas,
        gcl_mS_cm2=gcl,
        gshunt_mS_cm2=gshunt,
        e_pas_mV=neuron.eleak_mV,
        ecl_mV=neuron.ecl_mV,
        gna_mS_cm2=gna,
        gk_mS_cm2=gk,
        hh_gl_mS_cm2=0.0,
        optional_mechanisms=_optional_mechanism_settings(region_name, distance_um, available_mechanisms),
    )


def _effective_pas_reversal_mV(profile: RegionBiophysics) -> float:
    total_g = profile.g_pas_base_mS_cm2 + profile.gcl_mS_cm2 + profile.gshunt_mS_cm2
    if total_g <= 0:
        return profile.e_pas_mV
    weighted_sum = (
        profile.g_pas_base_mS_cm2 * profile.e_pas_mV
        + profile.gcl_mS_cm2 * profile.ecl_mV
        + profile.gshunt_mS_cm2 * profile.ecl_mV
    )
    return weighted_sum / total_g


def _apply_biophysics_to_rule(cell_rule: dict, neuron: NeuronConfig) -> float:
    ecl_mV = neuron.ecl_mV
    for section_name, section_rule in cell_rule["secs"].items():
        geom = section_rule.setdefault("geom", {})
        section_type = _section_type_from_name(section_name)
        base_profile = _segment_region_biophysics(neuron, section_type, 0.0, ())
        geom["Ra"] = DEFAULT_AXIAL_RESISTANCE_OHM_CM
        geom["cm"] = _section_base_cm(section_type)
        _set_section_compartmentalization(section_rule)

        mechs: dict[str, dict[str, float]] = {}
        mechs["hh"] = {
            "gnabar": max(base_profile.gna_mS_cm2, 1.0) / 1000.0,
            "gkbar": max(base_profile.gk_mS_cm2, 1.0) / 1000.0,
            "gl": 0.0,
            "el": neuron.eleak_mV,
        }
        mechs["pas"] = {
            "g": max(neuron.gl_mS_cm2 + neuron.gcl_mS_cm2 + neuron.gshunt_mS_cm2, 1e-9) / 1000.0,
            "e": neuron.eleak_mV,
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
    base_profile = _segment_region_biophysics(neuron, "soma", 0.0)
    net_params = specs.NetParams()
    net_params.cellParams["VC_HH"] = {
        "conds": {"cellType": "VCELL", "cellModel": "HH"},
        "secs": {
            "soma": {
                "geom": {
                    "diam": tutorial_cell.soma_diam_um,
                    "L": tutorial_cell.soma_length_um,
                    "Ra": DEFAULT_AXIAL_RESISTANCE_OHM_CM,
                    "cm": base_profile.cm_uF_cm2,
                },
                "mechs": {
                    "hh": {
                        "gnabar": base_profile.gna_mS_cm2 / 1000.0,
                        "gkbar": base_profile.gk_mS_cm2 / 1000.0,
                        "gl": 0.0,
                        "el": base_profile.e_pas_mV,
                    },
                    "pas": {
                        "g": (base_profile.g_pas_base_mS_cm2 + base_profile.gcl_mS_cm2 + base_profile.gshunt_mS_cm2) / 1000.0,
                        "e": _effective_pas_reversal_mV(base_profile),
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


def _record_if_available(h, segment, attr_name: str):
    if not hasattr(segment, attr_name):
        return None
    return h.Vector().record(getattr(segment, attr_name))


def _record_hh_state(h, segment) -> dict[str, object]:
    state = {
        "ina_mA_cm2": h.Vector().record(segment._ref_ina),
        "ik_mA_cm2": h.Vector().record(segment._ref_ik),
        "m": h.Vector().record(segment._ref_m_hh),
        "h": h.Vector().record(segment._ref_h_hh),
        "n": h.Vector().record(segment._ref_n_hh),
        "gna_S_cm2": h.Vector().record(segment._ref_gna_hh),
        "gk_S_cm2": h.Vector().record(segment._ref_gk_hh),
    }
    optional_refs = {
        "ica_mA_cm2": "_ref_ica",
        "cai_mM": "_ref_cai",
        "ih_mA_cm2": "_ref_ihcn_Ih",
        "m_Ih": "_ref_m_Ih",
        "gIh_S_cm2": "_ref_gIh_Ih",
        "ika_mA_cm2": "_ref_ika_KA",
        "m_KA": "_ref_m_KA",
        "h_KA": "_ref_h_KA",
        "gKA_S_cm2": "_ref_gKA_KA",
        "ina_nap_mA_cm2": "_ref_ina_nap_Nap_Et2",
        "m_NaP": "_ref_m_Nap_Et2",
        "gNaP_S_cm2": "_ref_gNap_Et2_Nap_Et2",
        "ik_im_mA_cm2": "_ref_ik_im_Im",
        "p_Im": "_ref_p_Im",
        "gIm_S_cm2": "_ref_gIm_Im",
        "ica_lva_mA_cm2": "_ref_ica_lva_Ca_LVAst",
        "m_Ca_LVA": "_ref_m_Ca_LVAst",
        "h_Ca_LVA": "_ref_h_Ca_LVAst",
        "gCa_LVA_S_cm2": "_ref_gCa_LVA_Ca_LVAst",
        "ica_hva_mA_cm2": "_ref_ica_hva_Ca_HVA",
        "m_Ca_HVA": "_ref_m_Ca_HVA",
        "h_Ca_HVA": "_ref_h_Ca_HVA",
        "gCa_HVA_S_cm2": "_ref_gCa_HVA_Ca_HVA",
        "isk_mA_cm2": "_ref_ik_sk_SK_E2",
        "z_SK": "_ref_z_SK_E2",
        "gSK_S_cm2": "_ref_gSK_E2_SK_E2",
    }
    for key, attr_name in optional_refs.items():
        vector = _record_if_available(h, segment, attr_name)
        if vector is not None:
            state[key] = vector
    return state


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
    profile: RegionBiophysics,
    recorded_len: int,
    python_ih_controller: PythonIhController | None = None,
    python_ka_controller: PythonKAController | None = None,
    python_calva_controller: PythonCaLVAController | None = None,
) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    g_leak_values = [profile.g_pas_base_mS_cm2] * recorded_len
    gcl_values = [profile.gcl_mS_cm2] * recorded_len
    gshunt_values = [profile.gshunt_mS_cm2] * recorded_len
    gpas_total_values = [profile.g_pas_base_mS_cm2 + profile.gcl_mS_cm2 + profile.gshunt_mS_cm2] * recorded_len
    conductances = {
        "g_Na": _vector_to_floats(hh_vectors["gna_S_cm2"], recorded_len, scale=1000.0),
        "g_K": _vector_to_floats(hh_vectors["gk_S_cm2"], recorded_len, scale=1000.0),
        "g_leak": g_leak_values,
        "g_Cl": gcl_values,
        "g_shunt": gshunt_values,
        "g_inhib_total": gpas_total_values,
    }
    if "gIh_S_cm2" in hh_vectors:
        conductances["g_Ih"] = _vector_to_floats(hh_vectors["gIh_S_cm2"], recorded_len, scale=1000.0)
    elif python_ih_controller is not None and python_ih_controller.conductances_S_cm2:
        conductances["g_Ih"] = [float(value) * 1000.0 for value in python_ih_controller.conductances_S_cm2[:recorded_len]]
    if "gKA_S_cm2" in hh_vectors:
        conductances["g_KA"] = _vector_to_floats(hh_vectors["gKA_S_cm2"], recorded_len, scale=1000.0)
    if python_ka_controller is not None and python_ka_controller.conductances_S_cm2:
        conductances["g_KA"] = [float(value) * 1000.0 for value in python_ka_controller.conductances_S_cm2[:recorded_len]]
    if "gNaP_S_cm2" in hh_vectors:
        conductances["g_NaP"] = _vector_to_floats(hh_vectors["gNaP_S_cm2"], recorded_len, scale=1000.0)
    if "gIm_S_cm2" in hh_vectors:
        conductances["g_Im"] = _vector_to_floats(hh_vectors["gIm_S_cm2"], recorded_len, scale=1000.0)
    if "gCa_LVA_S_cm2" in hh_vectors:
        conductances["g_Ca_LVA"] = _vector_to_floats(hh_vectors["gCa_LVA_S_cm2"], recorded_len, scale=1000.0)
    if python_calva_controller is not None and python_calva_controller.conductances_S_cm2:
        conductances["g_Ca_LVA"] = [float(value) * 1000.0 for value in python_calva_controller.conductances_S_cm2[:recorded_len]]
    if "gCa_HVA_S_cm2" in hh_vectors:
        conductances["g_Ca_HVA"] = _vector_to_floats(hh_vectors["gCa_HVA_S_cm2"], recorded_len, scale=1000.0)
    if "gSK_S_cm2" in hh_vectors:
        conductances["g_SK"] = _vector_to_floats(hh_vectors["gSK_S_cm2"], recorded_len, scale=1000.0)
    for mechanism_name, parameter_pairs in profile.optional_mechanisms:
        for parameter_name, value in parameter_pairs:
            if parameter_name.startswith("g") and parameter_name.endswith(f"_{mechanism_name}"):
                conductance_key = {
                    "gIhbar_Ih": "g_Ih",
                    "gKAbar_KA": "g_KA",
                    "gNap_Et2bar_Nap_Et2": "g_NaP",
                    "gImbar_Im": "g_Im",
                    "gCa_LVAstbar_Ca_LVAst": "g_Ca_LVA",
                    "gCa_HVAbar_Ca_HVA": "g_Ca_HVA",
                    "gSK_E2bar_SK_E2": "g_SK",
                }.get(parameter_name, parameter_name.split("_", 1)[0])
                if conductance_key in conductances:
                    continue
                conductances[conductance_key] = [value * 1000.0] * recorded_len
    ionic_currents = {
        "I_Na": _vector_to_floats(hh_vectors["ina_mA_cm2"], recorded_len, scale=1000.0),
        "I_K": _vector_to_floats(hh_vectors["ik_mA_cm2"], recorded_len, scale=1000.0),
        "I_leak": [
            conductance * (voltage - profile.e_pas_mV)
            for conductance, voltage in zip(g_leak_values, voltage_mV)
        ],
        "I_Cl": [
            conductance * (voltage - profile.ecl_mV)
            for conductance, voltage in zip(gcl_values, voltage_mV)
        ],
        "I_shunt": [
            conductance * (voltage - profile.ecl_mV)
            for conductance, voltage in zip(gshunt_values, voltage_mV)
        ],
        "I_inhib_total": _vector_to_floats(passive_vectors["ipas_mA_cm2"], recorded_len, scale=1000.0),
    }
    if "ica_mA_cm2" in hh_vectors:
        ionic_currents["I_Ca"] = _vector_to_floats(hh_vectors["ica_mA_cm2"], recorded_len, scale=1000.0)
    if "ih_mA_cm2" in hh_vectors:
        ionic_currents["I_h"] = _vector_to_floats(hh_vectors["ih_mA_cm2"], recorded_len, scale=1000.0)
    elif python_ih_controller is not None and python_ih_controller.currents_mA_cm2:
        ionic_currents["I_h"] = [float(value) * 1000.0 for value in python_ih_controller.currents_mA_cm2[:recorded_len]]
    if "ika_mA_cm2" in hh_vectors:
        ionic_currents["I_KA"] = _vector_to_floats(hh_vectors["ika_mA_cm2"], recorded_len, scale=1000.0)
    if python_ka_controller is not None and python_ka_controller.currents_mA_cm2:
        ionic_currents["I_KA"] = [float(value) * 1000.0 for value in python_ka_controller.currents_mA_cm2[:recorded_len]]
    if "ina_nap_mA_cm2" in hh_vectors:
        ionic_currents["I_NaP"] = _vector_to_floats(hh_vectors["ina_nap_mA_cm2"], recorded_len, scale=1000.0)
    if "ik_im_mA_cm2" in hh_vectors:
        ionic_currents["I_Im"] = _vector_to_floats(hh_vectors["ik_im_mA_cm2"], recorded_len, scale=1000.0)
    if "ica_lva_mA_cm2" in hh_vectors:
        ionic_currents["I_Ca_LVA"] = _vector_to_floats(hh_vectors["ica_lva_mA_cm2"], recorded_len, scale=1000.0)
    if python_calva_controller is not None and python_calva_controller.currents_mA_cm2:
        ionic_currents["I_Ca_LVA"] = [float(value) * 1000.0 for value in python_calva_controller.currents_mA_cm2[:recorded_len]]
    if "ica_hva_mA_cm2" in hh_vectors:
        ionic_currents["I_Ca_HVA"] = _vector_to_floats(hh_vectors["ica_hva_mA_cm2"], recorded_len, scale=1000.0)
    if "isk_mA_cm2" in hh_vectors:
        ionic_currents["I_SK"] = _vector_to_floats(hh_vectors["isk_mA_cm2"], recorded_len, scale=1000.0)
    gating_variables = {
        "m": _vector_to_floats(hh_vectors["m"], recorded_len),
        "h": _vector_to_floats(hh_vectors["h"], recorded_len),
        "n": _vector_to_floats(hh_vectors["n"], recorded_len),
    }
    if "m_Ih" in hh_vectors:
        gating_variables["m_Ih"] = _vector_to_floats(hh_vectors["m_Ih"], recorded_len)
    elif python_ih_controller is not None and python_ih_controller.gating_m:
        gating_variables["m_Ih"] = [float(value) for value in python_ih_controller.gating_m[:recorded_len]]
    if "m_KA" in hh_vectors:
        gating_variables["m_KA"] = _vector_to_floats(hh_vectors["m_KA"], recorded_len)
    if "h_KA" in hh_vectors:
        gating_variables["h_KA"] = _vector_to_floats(hh_vectors["h_KA"], recorded_len)
    if "m_NaP" in hh_vectors:
        gating_variables["m_NaP"] = _vector_to_floats(hh_vectors["m_NaP"], recorded_len)
    if "p_Im" in hh_vectors:
        gating_variables["p_Im"] = _vector_to_floats(hh_vectors["p_Im"], recorded_len)
    if python_ka_controller is not None and python_ka_controller.gating_m:
        gating_variables["m_KA"] = [float(value) for value in python_ka_controller.gating_m[:recorded_len]]
    if python_ka_controller is not None and python_ka_controller.gating_h:
        gating_variables["h_KA"] = [float(value) for value in python_ka_controller.gating_h[:recorded_len]]
    if "m_Ca_LVA" in hh_vectors:
        gating_variables["m_Ca_LVA"] = _vector_to_floats(hh_vectors["m_Ca_LVA"], recorded_len)
    if "h_Ca_LVA" in hh_vectors:
        gating_variables["h_Ca_LVA"] = _vector_to_floats(hh_vectors["h_Ca_LVA"], recorded_len)
    if "m_Ca_HVA" in hh_vectors:
        gating_variables["m_Ca_HVA"] = _vector_to_floats(hh_vectors["m_Ca_HVA"], recorded_len)
    if "h_Ca_HVA" in hh_vectors:
        gating_variables["h_Ca_HVA"] = _vector_to_floats(hh_vectors["h_Ca_HVA"], recorded_len)
    if "z_SK" in hh_vectors:
        gating_variables["z_SK"] = _vector_to_floats(hh_vectors["z_SK"], recorded_len)
    if python_calva_controller is not None and python_calva_controller.gating_m:
        gating_variables["m_Ca_LVA"] = [float(value) for value in python_calva_controller.gating_m[:recorded_len]]
    if python_calva_controller is not None and python_calva_controller.gating_h:
        gating_variables["h_Ca_LVA"] = [float(value) for value in python_calva_controller.gating_h[:recorded_len]]
    if python_calva_controller is not None and python_calva_controller.cai_mM:
        gating_variables["cai_mM"] = [float(value) for value in python_calva_controller.cai_mM[:recorded_len]]
    if "cai_mM" in hh_vectors:
        gating_variables["cai_mM"] = _vector_to_floats(hh_vectors["cai_mM"], recorded_len)
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


def _insert_section_mechanisms(section, mechanism_names: tuple[str, ...]) -> None:
    for mechanism_name in mechanism_names:
        try:
            section.insert(mechanism_name)
        except Exception:
            continue


def _soma_section(cell):
    for section_name, sec_data in cell.secs.items():
        if _section_type_from_name(section_name) == "soma":
            return sec_data["hObj"]
    return next(iter(cell.secs.values()))["hObj"]


def _apply_region_biophysics(cell, neuron: NeuronConfig, h, available_mechanisms: tuple[str, ...]) -> None:
    _configure_optional_mechanism_globals(h)
    soma_section = _soma_section(cell)
    h.distance(0.0, 0.5, sec=soma_section)
    for section_name, sec_data in cell.secs.items():
        section = sec_data.get("hObj")
        if section is None:
            continue
        section_type = _section_type_from_name(section_name)
        segment_profiles: list[tuple[object, RegionBiophysics]] = []
        section_mechanisms: set[str] = set()
        for segment in section:
            distance_um = float(h.distance(segment.x, sec=section))
            profile = _segment_region_biophysics(neuron, section_type, distance_um, available_mechanisms)
            segment_profiles.append((segment, profile))
            section_mechanisms.update(mechanism_name for mechanism_name, _ in profile.optional_mechanisms)

        inserted_mechanisms = tuple(sorted(section_mechanisms))
        _insert_section_mechanisms(section, inserted_mechanisms)
        section.Ra = DEFAULT_AXIAL_RESISTANCE_OHM_CM
        section.ena = neuron.ena_mV
        section.ek = neuron.ek_mV

        for segment, profile in segment_profiles:
            total_passive_g = profile.g_pas_base_mS_cm2 + profile.gcl_mS_cm2 + profile.gshunt_mS_cm2

            segment.cm = profile.cm_uF_cm2
            segment.gnabar_hh = profile.gna_mS_cm2 / 1000.0
            segment.gkbar_hh = profile.gk_mS_cm2 / 1000.0
            segment.gl_hh = profile.hh_gl_mS_cm2 / 1000.0
            segment.el_hh = profile.e_pas_mV
            segment.g_pas = max(total_passive_g, 1e-9) / 1000.0
            segment.e_pas = _effective_pas_reversal_mV(profile)

            for mechanism_name in inserted_mechanisms:
                _apply_segment_parameter_pairs(segment, _optional_mechanism_zero_settings(mechanism_name))
            for mechanism_name, parameter_pairs in profile.optional_mechanisms:
                _apply_segment_parameter_pairs(segment, parameter_pairs)
            if hasattr(segment, "cai"):
                if profile.region_name == REGION_APICAL_HOTZONE:
                    segment.cai = APICAL_HOTZONE_CALVA_CAI_REST_MM
                else:
                    segment.cai = 1.0e-4
            if hasattr(segment, "cao"):
                segment.cao = APICAL_HOTZONE_CALVA_CAO_MM if profile.region_name == REGION_APICAL_HOTZONE else 2.0


def _segment_profile_at_site(cell, site: MorphologySite | None, neuron: NeuronConfig, h, available_mechanisms: tuple[str, ...], morphology_name: str | None = None) -> RegionBiophysics:
    if morphology_name is None and site is None:
        site = MorphologySite(section_name="soma", section_x=0.5)
    section, section_x = _resolve_site(cell, site or default_recording_site(morphology_name))
    soma_section = _soma_section(cell)
    distance_um = _segment_distance_um(h, soma_section, section, section_x)
    return _segment_region_biophysics(neuron, _section_type_from_name(section.name().split(".")[-1]), distance_um, available_mechanisms)


def _segment_cache_key(section_name: str, segment_x: float) -> tuple[str, float]:
    return section_name, round(float(segment_x), 6)


def _neuron_config_cache_key(neuron: NeuronConfig) -> tuple[object, ...]:
    return tuple(getattr(neuron, field_name) for field_name in NeuronConfig.__dataclass_fields__)


def _neuron_config_from_cache_key(cache_key: tuple[object, ...]) -> NeuronConfig:
    return NeuronConfig(**dict(zip(NeuronConfig.__dataclass_fields__, cache_key)))


def _steady_state_model_signature() -> tuple[object, ...]:
    return tuple(globals()[name] for name in STEADY_STATE_MODEL_SIGNATURE_NAMES)


def _segment_state_pairs(segment) -> tuple[tuple[str, float], ...]:
    values: list[tuple[str, float]] = []
    for attr_name in STEADY_STATE_SEGMENT_ATTRIBUTES:
        if hasattr(segment, attr_name):
            values.append((attr_name, float(getattr(segment, attr_name))))
    return tuple(values)


def _snapshot_cell_segment_states(cell) -> tuple[SegmentStateSnapshot, ...]:
    states: list[SegmentStateSnapshot] = []
    for section_name, sec_data in cell.secs.items():
        section = sec_data.get("hObj")
        if section is None:
            continue
        for segment in section:
            states.append(
                SegmentStateSnapshot(
                    section_name=section_name,
                    segment_x=float(segment.x),
                    values=_segment_state_pairs(segment),
                )
            )
    return tuple(states)


def _controller_segment_state_pairs(controller_segment, attr_names: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    return tuple((attr_name, float(getattr(controller_segment, attr_name))) for attr_name in attr_names)


def _snapshot_controller_states(controller, attr_names: tuple[str, ...]) -> tuple[ControllerStateSnapshot, ...]:
    if controller is None:
        return ()
    states: list[ControllerStateSnapshot] = []
    for segment_state in controller.segments:
        section_name = _normalize_section_name(segment_state.segment.sec.name())
        states.append(
            ControllerStateSnapshot(
                section_name=section_name,
                segment_x=float(segment_state.segment.x),
                values=_controller_segment_state_pairs(segment_state, attr_names),
            )
        )
    return tuple(states)


def _snapshot_steady_state_initialization(
    cell,
    python_ih_controller: PythonIhController | None = None,
    python_ka_controller: PythonKAController | None = None,
    python_calva_controller: PythonCaLVAController | None = None,
) -> SteadyStateInitializationSnapshot:
    return SteadyStateInitializationSnapshot(
        segment_states=_snapshot_cell_segment_states(cell),
        ih_states=_snapshot_controller_states(python_ih_controller, ("m",)),
        ka_states=_snapshot_controller_states(python_ka_controller, ("m", "h")),
        calva_states=_snapshot_controller_states(python_calva_controller, ("m", "h", "cai_mM")),
    )


def _apply_segment_state_pairs(target, values: tuple[tuple[str, float], ...]) -> None:
    for attr_name, value in values:
        if attr_name == "v":
            target.v = value
        elif hasattr(target, attr_name):
            setattr(target, attr_name, value)


def _apply_cell_segment_state_snapshot(cell, segment_states: tuple[SegmentStateSnapshot, ...]) -> None:
    segment_lookup: dict[tuple[str, float], object] = {}
    for section_name, sec_data in cell.secs.items():
        section = sec_data.get("hObj")
        if section is None:
            continue
        for segment in section:
            segment_lookup[_segment_cache_key(section_name, segment.x)] = segment
    for snapshot in segment_states:
        segment = segment_lookup.get(_segment_cache_key(snapshot.section_name, snapshot.segment_x))
        if segment is None:
            continue
        _apply_segment_state_pairs(segment, snapshot.values)


def _apply_controller_state_snapshot(controller, snapshot_states: tuple[ControllerStateSnapshot, ...]) -> None:
    if controller is None:
        return
    controller_lookup: dict[tuple[str, float], object] = {}
    for segment_state in controller.segments:
        section_name = _normalize_section_name(segment_state.segment.sec.name())
        controller_lookup[_segment_cache_key(section_name, segment_state.segment.x)] = segment_state
    for snapshot in snapshot_states:
        segment_state = controller_lookup.get(_segment_cache_key(snapshot.section_name, snapshot.segment_x))
        if segment_state is None:
            continue
        for attr_name, value in snapshot.values:
            setattr(segment_state, attr_name, value)


def _refresh_python_ih_segment(segment_state: PythonIhSegment) -> None:
    v_mV = float(segment_state.segment.v)
    segment_state.gIh_S_cm2 = segment_state.gbar_S_cm2 * segment_state.m
    segment_state.ih_mA_cm2 = segment_state.gIh_S_cm2 * (v_mV - segment_state.ehcn_mV)
    segment_state.clamp.amp = -segment_state.ih_mA_cm2 * segment_state.area_cm2 * 1e6


def _refresh_python_ka_segment(segment_state: PythonKASegment) -> None:
    v_mV = float(segment_state.segment.v)
    segment_state.gKA_S_cm2 = segment_state.gbar_S_cm2 * (segment_state.m ** 4) * segment_state.h
    segment_state.ika_mA_cm2 = segment_state.gKA_S_cm2 * (v_mV - segment_state.ek_mV)
    segment_state.clamp.amp = -segment_state.ika_mA_cm2 * segment_state.area_cm2 * 1e6


def _refresh_python_calva_segment(segment_state: PythonCaLVASegment) -> None:
    v_mV = float(segment_state.segment.v)
    segment_state.eca_mV = _eca_from_concentrations(segment_state.cai_mM, segment_state.cao_mM)
    segment_state.gCaLVA_S_cm2 = segment_state.gbar_S_cm2 * (segment_state.m ** 2) * segment_state.h
    segment_state.ica_mA_cm2 = segment_state.gCaLVA_S_cm2 * (v_mV - segment_state.eca_mV)
    segment_state.clamp.amp = -segment_state.ica_mA_cm2 * segment_state.area_cm2 * 1e6


def _apply_steady_state_initialization_snapshot(
    cell,
    snapshot: SteadyStateInitializationSnapshot,
    python_ih_controller: PythonIhController | None = None,
    python_ka_controller: PythonKAController | None = None,
    python_calva_controller: PythonCaLVAController | None = None,
) -> None:
    _apply_cell_segment_state_snapshot(cell, snapshot.segment_states)
    _apply_controller_state_snapshot(python_ih_controller, snapshot.ih_states)
    _apply_controller_state_snapshot(python_ka_controller, snapshot.ka_states)
    _apply_controller_state_snapshot(python_calva_controller, snapshot.calva_states)


@lru_cache(maxsize=32)
def _steady_state_initialization_snapshot_cached(
    neuron_key: tuple[object, ...],
    morphology_path: str,
    available_mechanisms: tuple[str, ...],
    model_signature: tuple[object, ...],
) -> SteadyStateInitializationSnapshot:
    del model_signature
    neuron = _neuron_config_from_cache_key(neuron_key)
    specs, sim_backend, h = _import_backend()
    net_params, _ = _build_morphology_cell(specs, neuron, morphology_path)
    sim_config = _build_sim_config(specs, neuron)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim_backend.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell = sim_backend.net.cells[0]
        _apply_section_reversal_potentials(cell, neuron)
        _apply_region_biophysics(cell, neuron, h, available_mechanisms)
        python_ih_controller = None if "Ih" in available_mechanisms else _build_python_ih_controller(cell, neuron, h)
        python_ka_controller = None if "KA" in available_mechanisms else _build_python_ka_controller(cell, neuron, h)
        python_calva_controller = None if "Ca_LVAst" in available_mechanisms else _build_python_calva_controller(cell, h)
        _simulate_with_python_currents(
            h,
            STEADY_STATE_INITIALIZATION_DURATION_MS,
            neuron.dt_ms,
            neuron.v_rest_mV,
            [python_ih_controller] if python_ih_controller else [],
            [python_ka_controller] if python_ka_controller else [],
            [python_calva_controller] if python_calva_controller else [],
        )
    return _snapshot_steady_state_initialization(
        cell,
        python_ih_controller=python_ih_controller,
        python_ka_controller=python_ka_controller,
        python_calva_controller=python_calva_controller,
    )


def _steady_state_initialization_snapshot(
    neuron: NeuronConfig,
    morphology_name: str | None,
    available_mechanisms: tuple[str, ...],
) -> SteadyStateInitializationSnapshot:
    morphology_path = str(resolve_morphology_path(morphology_name))
    return _steady_state_initialization_snapshot_cached(
        _neuron_config_cache_key(neuron),
        morphology_path,
        available_mechanisms,
        _steady_state_model_signature(),
    )


def _steady_state_initializer(
    cell,
    snapshot: SteadyStateInitializationSnapshot,
    python_ih_controller: PythonIhController | None = None,
    python_ka_controller: PythonKAController | None = None,
    python_calva_controller: PythonCaLVAController | None = None,
):
    def initialize_state() -> None:
        _apply_steady_state_initialization_snapshot(
            cell,
            snapshot,
            python_ih_controller=python_ih_controller,
            python_ka_controller=python_ka_controller,
            python_calva_controller=python_calva_controller,
        )

    return initialize_state


def _ih_rates(v_mV: float) -> tuple[float, float]:
    minf = 1.0 / (1.0 + math.exp((v_mV + 90.0) / 10.0))
    mtau_ms = 5.0 + 40.0 / (1.0 + math.exp((v_mV + 75.0) / 12.0))
    return minf, mtau_ms


def _update_python_ih_segment(segment_state: PythonIhSegment, dt_ms: float | None = None) -> None:
    v_mV = float(segment_state.segment.v)
    if dt_ms is None:
        minf, _ = _ih_rates(v_mV)
        segment_state.m = minf
    else:
        minf, mtau_ms = _ih_rates(v_mV)
        segment_state.m = minf + (segment_state.m - minf) * math.exp(-dt_ms / max(mtau_ms, 1e-6))
    segment_state.gIh_S_cm2 = segment_state.gbar_S_cm2 * segment_state.m
    segment_state.ih_mA_cm2 = segment_state.gIh_S_cm2 * (v_mV - segment_state.ehcn_mV)
    segment_state.clamp.amp = -segment_state.ih_mA_cm2 * segment_state.area_cm2 * 1e6


def _ka_rates(v_mV: float) -> tuple[float, float, float, float]:
    m_inf = 1.0 / (1.0 + math.exp(-(v_mV + 30.0) / 8.5))
    h_inf = 1.0 / (1.0 + math.exp((v_mV + 58.0) / 6.0))
    tau_m_ms = 0.2 + 2.0 / (1.0 + math.exp((v_mV + 25.0) / 10.0))
    tau_h_ms = 5.0 + 20.0 / (1.0 + math.exp((v_mV + 50.0) / 8.0))
    return m_inf, h_inf, tau_m_ms, tau_h_ms


def _calva_rates(v_mV: float) -> tuple[float, float, float, float]:
    m_inf = 1.0 / (1.0 + math.exp(-(v_mV - CALVA_M_VHALF_MV) / CALVA_M_SLOPE_MV))
    h_inf = 1.0 / (1.0 + math.exp(-(v_mV - CALVA_H_VHALF_MV) / CALVA_H_SLOPE_MV))
    tau_m_ms = CALVA_M_TAU_BASE_MS + CALVA_M_TAU_SCALE_MS / (
        1.0 + math.exp((v_mV - CALVA_M_TAU_VHALF_MV) / CALVA_M_TAU_SLOPE_MV)
    )
    tau_h_ms = CALVA_H_TAU_BASE_MS + CALVA_H_TAU_SCALE_MS / (
        1.0 + math.exp((v_mV - CALVA_H_TAU_VHALF_MV) / CALVA_H_TAU_SLOPE_MV)
    )
    return m_inf, h_inf, tau_m_ms, tau_h_ms


def _eca_from_concentrations(cai_mM: float, cao_mM: float, temperature_c: float = DEFAULT_CELSIUS_C) -> float:
    clamped_cai = max(cai_mM, 1e-7)
    temperature_K = temperature_c + 273.15
    return ((GAS_CONSTANT_J_MOL_K * temperature_K) / (2.0 * FARADAY_C_MOL)) * 1000.0 * math.log(cao_mM / clamped_cai)


def _update_python_ka_segment(segment_state: PythonKASegment, dt_ms: float | None = None) -> None:
    v_mV = float(segment_state.segment.v)
    m_inf, h_inf, tau_m_ms, tau_h_ms = _ka_rates(v_mV)
    if dt_ms is None:
        segment_state.m = m_inf
        segment_state.h = h_inf
    else:
        segment_state.m = m_inf + (segment_state.m - m_inf) * math.exp(-dt_ms / max(tau_m_ms, 1e-6))
        segment_state.h = h_inf + (segment_state.h - h_inf) * math.exp(-dt_ms / max(tau_h_ms, 1e-6))
    segment_state.gKA_S_cm2 = segment_state.gbar_S_cm2 * (segment_state.m ** 4) * segment_state.h
    segment_state.ika_mA_cm2 = segment_state.gKA_S_cm2 * (v_mV - segment_state.ek_mV)
    segment_state.clamp.amp = -segment_state.ika_mA_cm2 * segment_state.area_cm2 * 1e6


def _update_python_calva_segment(segment_state: PythonCaLVASegment, dt_ms: float | None = None) -> None:
    v_mV = float(segment_state.segment.v)
    m_inf, h_inf, tau_m_ms, tau_h_ms = _calva_rates(v_mV)
    if dt_ms is None:
        segment_state.m = m_inf
        segment_state.h = h_inf
    else:
        segment_state.m = m_inf + (segment_state.m - m_inf) * math.exp(-dt_ms / max(tau_m_ms, 1e-6))
        segment_state.h = h_inf + (segment_state.h - h_inf) * math.exp(-dt_ms / max(tau_h_ms, 1e-6))
    segment_state.eca_mV = _eca_from_concentrations(segment_state.cai_mM, segment_state.cao_mM)
    segment_state.gCaLVA_S_cm2 = segment_state.gbar_S_cm2 * (segment_state.m ** 2) * segment_state.h
    segment_state.ica_mA_cm2 = segment_state.gCaLVA_S_cm2 * (v_mV - segment_state.eca_mV)
    if dt_ms is not None:
        inward_current = max(0.0, -segment_state.ica_mA_cm2)
        decay_term = (segment_state.cai_mM - segment_state.cai_rest_mM) / max(segment_state.decay_ms, 1e-6)
        segment_state.cai_mM += dt_ms * (segment_state.influx_scale_mM_per_ms_per_mA_cm2 * inward_current - decay_term)
        segment_state.cai_mM = max(segment_state.cai_rest_mM * 0.5, segment_state.cai_mM)
        segment_state.eca_mV = _eca_from_concentrations(segment_state.cai_mM, segment_state.cao_mM)
        segment_state.ica_mA_cm2 = segment_state.gCaLVA_S_cm2 * (v_mV - segment_state.eca_mV)
    segment_state.clamp.amp = -segment_state.ica_mA_cm2 * segment_state.area_cm2 * 1e6


def _snapshot_python_ih(controller: PythonIhController) -> None:
    recorded = next((item for item in controller.segments if item.record), None)
    if recorded is None:
        return
    controller.currents_mA_cm2.append(recorded.ih_mA_cm2)
    controller.conductances_S_cm2.append(recorded.gIh_S_cm2)
    controller.gating_m.append(recorded.m)


def _snapshot_python_ka(controller: PythonKAController) -> None:
    recorded = next((item for item in controller.segments if item.record), None)
    if recorded is None:
        return
    controller.currents_mA_cm2.append(recorded.ika_mA_cm2)
    controller.conductances_S_cm2.append(recorded.gKA_S_cm2)
    controller.gating_m.append(recorded.m)
    controller.gating_h.append(recorded.h)


def _snapshot_python_calva(controller: PythonCaLVAController) -> None:
    recorded = next((item for item in controller.segments if item.record), None)
    if recorded is None:
        return
    controller.currents_mA_cm2.append(recorded.ica_mA_cm2)
    controller.conductances_S_cm2.append(recorded.gCaLVA_S_cm2)
    controller.gating_m.append(recorded.m)
    controller.gating_h.append(recorded.h)
    controller.cai_mM.append(recorded.cai_mM)


def _build_python_ih_controller(cell, neuron: NeuronConfig, h, record_section=None, record_segment=None) -> PythonIhController | None:
    segments: list[PythonIhSegment] = []
    soma_section = _soma_section(cell)
    h.distance(0.0, 0.5, sec=soma_section)
    record_section_name = record_section.name() if record_section is not None else None
    record_target_x = float(record_segment.x) if record_segment is not None else None
    record_candidates: list[PythonIhSegment] = []
    for section_name, sec_data in cell.secs.items():
        section = sec_data.get("hObj")
        if section is None:
            continue
        section_type = _section_type_from_name(section_name)
        for segment in section:
            distance_um = float(h.distance(segment.x, sec=section))
            ih_settings = _ih_settings_for_region(_region_name_for_segment(section_type, distance_um), distance_um)
            if ih_settings is None:
                continue
            gbar_S_cm2, ehcn_mV = ih_settings
            clamp = h.IClamp(segment)
            clamp.delay = 0.0
            clamp.dur = 1e9
            clamp.amp = 0.0
            area_cm2 = float(segment.area()) * 1e-8
            segment_state = PythonIhSegment(
                segment=segment,
                clamp=clamp,
                area_cm2=area_cm2,
                gbar_S_cm2=gbar_S_cm2,
                ehcn_mV=ehcn_mV,
            )
            segments.append(segment_state)
            if record_section_name is not None and section.name() == record_section_name:
                record_candidates.append(segment_state)
    if not segments:
        return None
    if record_section_name is not None and record_target_x is not None and record_candidates:
        best_match = min(record_candidates, key=lambda item: abs(float(item.segment.x) - record_target_x))
        best_match.record = True
    controller = PythonIhController(h=h, segments=segments)
    for segment_state in controller.segments:
        _update_python_ih_segment(segment_state, dt_ms=None)
    _snapshot_python_ih(controller)
    return controller


def _build_python_ka_controller(cell, neuron: NeuronConfig, h, record_section=None, record_segment=None) -> PythonKAController | None:
    segments: list[PythonKASegment] = []
    soma_section = _soma_section(cell)
    h.distance(0.0, 0.5, sec=soma_section)
    record_section_name = record_section.name() if record_section is not None else None
    record_target_x = float(record_segment.x) if record_segment is not None else None
    record_candidates: list[PythonKASegment] = []
    for section_name, sec_data in cell.secs.items():
        section = sec_data.get("hObj")
        if section is None:
            continue
        section_type = _section_type_from_name(section_name)
        for segment in section:
            distance_um = float(h.distance(segment.x, sec=section))
            region_name = _region_name_for_segment(section_type, distance_um)
            gbar_S_cm2 = _ka_settings_for_region(region_name, distance_um)
            if gbar_S_cm2 is None:
                continue
            clamp = h.IClamp(segment)
            clamp.delay = 0.0
            clamp.dur = 1e9
            clamp.amp = 0.0
            area_cm2 = float(segment.area()) * 1e-8
            segment_state = PythonKASegment(
                segment=segment,
                clamp=clamp,
                area_cm2=area_cm2,
                gbar_S_cm2=gbar_S_cm2,
                ek_mV=neuron.ek_mV,
            )
            segments.append(segment_state)
            if record_section_name is not None and section.name() == record_section_name:
                record_candidates.append(segment_state)
    if not segments:
        return None
    if record_section_name is not None and record_target_x is not None and record_candidates:
        best_match = min(record_candidates, key=lambda item: abs(float(item.segment.x) - record_target_x))
        best_match.record = True
    controller = PythonKAController(h=h, segments=segments)
    for segment_state in controller.segments:
        _update_python_ka_segment(segment_state, dt_ms=None)
    _snapshot_python_ka(controller)
    return controller


def _build_python_calva_controller(cell, h, record_section=None, record_segment=None) -> PythonCaLVAController | None:
    segments: list[PythonCaLVASegment] = []
    soma_section = _soma_section(cell)
    h.distance(0.0, 0.5, sec=soma_section)
    record_section_name = record_section.name() if record_section is not None else None
    record_target_x = float(record_segment.x) if record_segment is not None else None
    record_candidates: list[PythonCaLVASegment] = []
    for section_name, sec_data in cell.secs.items():
        section = sec_data.get("hObj")
        if section is None:
            continue
        section_type = _section_type_from_name(section_name)
        for segment in section:
            distance_um = float(h.distance(segment.x, sec=section))
            region_name = _region_name_for_segment(section_type, distance_um)
            calva_settings = _calva_settings_for_region(region_name)
            if calva_settings is None:
                continue
            gbar_S_cm2, cai_rest_mM, cao_mM, decay_ms = calva_settings
            clamp = h.IClamp(segment)
            clamp.delay = 0.0
            clamp.dur = 1e9
            clamp.amp = 0.0
            area_cm2 = float(segment.area()) * 1e-8
            segment_state = PythonCaLVASegment(
                segment=segment,
                clamp=clamp,
                area_cm2=area_cm2,
                gbar_S_cm2=gbar_S_cm2,
                cao_mM=cao_mM,
                cai_rest_mM=cai_rest_mM,
                decay_ms=decay_ms,
                influx_scale_mM_per_ms_per_mA_cm2=APICAL_HOTZONE_CALVA_INFLUX_SCALE,
                cai_mM=cai_rest_mM,
            )
            segments.append(segment_state)
            if record_section_name is not None and section.name() == record_section_name:
                record_candidates.append(segment_state)
    if not segments:
        return None
    if record_section_name is not None and record_target_x is not None and record_candidates:
        best_match = min(record_candidates, key=lambda item: abs(float(item.segment.x) - record_target_x))
        best_match.record = True
    controller = PythonCaLVAController(h=h, segments=segments)
    for segment_state in controller.segments:
        _update_python_calva_segment(segment_state, dt_ms=None)
    _snapshot_python_calva(controller)
    return controller


def _simulate_with_python_currents(
    h,
    duration_ms: float,
    dt_ms: float,
    initial_v_mV: float,
    ih_controllers: list[PythonIhController] | None = None,
    ka_controllers: list[PythonKAController] | None = None,
    calva_controllers: list[PythonCaLVAController] | None = None,
    initialize_state=None,
) -> None:
    ih_controllers = ih_controllers or []
    ka_controllers = ka_controllers or []
    calva_controllers = calva_controllers or []
    h.dt = dt_ms
    if hasattr(h, "steps_per_ms"):
        h.steps_per_ms = 1.0 / dt_ms
    h.finitialize(initial_v_mV)
    seeded_from_snapshot = initialize_state is not None
    if initialize_state is not None:
        initialize_state()
        if hasattr(h, "fcurrent"):
            h.fcurrent()
        if hasattr(h, "frecord_init"):
            h.frecord_init()
    for controller in ih_controllers:
        for segment_state in controller.segments:
            if seeded_from_snapshot:
                _refresh_python_ih_segment(segment_state)
            else:
                _update_python_ih_segment(segment_state, dt_ms=None)
        controller.currents_mA_cm2.clear()
        controller.conductances_S_cm2.clear()
        controller.gating_m.clear()
        _snapshot_python_ih(controller)
    for controller in ka_controllers:
        for segment_state in controller.segments:
            if seeded_from_snapshot:
                _refresh_python_ka_segment(segment_state)
            else:
                _update_python_ka_segment(segment_state, dt_ms=None)
        controller.currents_mA_cm2.clear()
        controller.conductances_S_cm2.clear()
        controller.gating_m.clear()
        controller.gating_h.clear()
        _snapshot_python_ka(controller)
    for controller in calva_controllers:
        for segment_state in controller.segments:
            if seeded_from_snapshot:
                _refresh_python_calva_segment(segment_state)
            else:
                _update_python_calva_segment(segment_state, dt_ms=None)
        controller.currents_mA_cm2.clear()
        controller.conductances_S_cm2.clear()
        controller.gating_m.clear()
        controller.gating_h.clear()
        controller.cai_mM.clear()
        _snapshot_python_calva(controller)
    max_time_ms = duration_ms - (0.5 * dt_ms)
    while float(h.t) < max_time_ms:
        h.fadvance()
        for controller in ih_controllers:
            for segment_state in controller.segments:
                _update_python_ih_segment(segment_state, dt_ms=dt_ms)
            _snapshot_python_ih(controller)
        for controller in ka_controllers:
            for segment_state in controller.segments:
                _update_python_ka_segment(segment_state, dt_ms=dt_ms)
            _snapshot_python_ka(controller)
        for controller in calva_controllers:
            for segment_state in controller.segments:
                _update_python_calva_segment(segment_state, dt_ms=dt_ms)
            _snapshot_python_calva(controller)


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
    available_mechanisms = _effective_optional_mechanisms(_ensure_optional_mechanisms_loaded())

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
    steady_state_snapshot = None
    if morphology_name:
        steady_state_snapshot = _steady_state_initialization_snapshot(
            neuron,
            morphology_name,
            available_mechanisms,
        )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell = sim.net.cells[0]
        _apply_section_reversal_potentials(cell, neuron)
        _apply_region_biophysics(cell, neuron, h, available_mechanisms)
        if morphology_name:
            sec, sec_x = _resolve_site(cell, recording_site or default_recording_site(morphology_name))
            segment = sec(sec_x)
        else:
            sec = cell.secs["soma"]["hObj"]
            sec_x = 0.5
            segment = sec(sec_x)

        clamp = h.SEClamp(sec(sec_x))
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
        python_ih_controller = None if "Ih" in available_mechanisms else _build_python_ih_controller(cell, neuron, h, sec, segment)
        python_ka_controller = None if "KA" in available_mechanisms else _build_python_ka_controller(cell, neuron, h, sec, segment)
        python_calva_controller = None if "Ca_LVAst" in available_mechanisms else _build_python_calva_controller(cell, h, sec, segment)
        initialize_state = None
        if steady_state_snapshot is not None:
            initialize_state = _steady_state_initializer(
                cell,
                steady_state_snapshot,
                python_ih_controller=python_ih_controller,
                python_ka_controller=python_ka_controller,
                python_calva_controller=python_calva_controller,
            )
        _simulate_with_python_currents(
            h,
            neuron.duration_ms,
            neuron.dt_ms,
            neuron.v_rest_mV,
            [python_ih_controller] if python_ih_controller else [],
            [python_ka_controller] if python_ka_controller else [],
            [python_calva_controller] if python_calva_controller else [],
            initialize_state=initialize_state,
        )

    # Trim to the common recorded length in case NEURON includes one extra sample.
    recorded_len = min(len(time_rec), len(voltage_rec), len(current_rec), len(command_voltage))
    voltage_trace = [float(voltage_rec[index]) for index in range(recorded_len)]
    record_site = recording_site if morphology_name else MorphologySite(section_name="soma", section_x=0.5)
    if morphology_name and record_site is None:
        record_site = default_recording_site(morphology_name)
    profile = _segment_profile_at_site(cell, record_site, neuron, h, available_mechanisms, morphology_name)
    ionic_currents, gating_variables, conductances = _extract_membrane_traces(
        hh_vectors,
        passive_vectors,
        voltage_trace,
        profile,
        recorded_len,
        python_ih_controller=python_ih_controller,
        python_ka_controller=python_ka_controller,
        python_calva_controller=python_calva_controller,
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
    available_mechanisms = _effective_optional_mechanisms(_ensure_optional_mechanisms_loaded())
    current_unit = neuron.current_injection_unit

    times_ms = [index * neuron.dt_ms for index in range(step_count + 1)]
    configured_command = [neuron.holding_current + command_delta(time_ms, trains) for time_ms in times_ms]
    record_membrane_state = selected_panels is None or any(
        panel in selected_panels for panel in ("ionic_currents", "gating", "conductances")
    )

    net_params, ecl_mV = _build_morphology_cell(specs, neuron, morphology_name)
    sim_config = _build_sim_config(specs, neuron)
    steady_state_snapshot = _steady_state_initialization_snapshot(
        neuron,
        morphology_name,
        available_mechanisms,
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell = sim.net.cells[0]
        _apply_section_reversal_potentials(cell, neuron)
        _apply_region_biophysics(cell, neuron, h, available_mechanisms)
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
        python_ih_controller = None if "Ih" in available_mechanisms else _build_python_ih_controller(cell, neuron, h, record_sec, segment)
        python_ka_controller = None if "KA" in available_mechanisms else _build_python_ka_controller(cell, neuron, h, record_sec, segment)
        python_calva_controller = None if "Ca_LVAst" in available_mechanisms else _build_python_calva_controller(cell, h, record_sec, segment)
        _simulate_with_python_currents(
            h,
            neuron.duration_ms,
            neuron.dt_ms,
            neuron.v_rest_mV,
            [python_ih_controller] if python_ih_controller else [],
            [python_ka_controller] if python_ka_controller else [],
            [python_calva_controller] if python_calva_controller else [],
            initialize_state=_steady_state_initializer(
                cell,
                steady_state_snapshot,
                python_ih_controller=python_ih_controller,
                python_ka_controller=python_ka_controller,
                python_calva_controller=python_calva_controller,
            ),
        )

    recorded_len = min(len(time_rec), len(voltage_rec), len(total_applied_current_nA), len(configured_command))
    current_trace = total_applied_current_nA[:recorded_len]
    voltage_trace = [float(voltage_rec[index]) for index in range(recorded_len)]
    profile = _segment_profile_at_site(cell, recording_site or default_recording_site(morphology_name), neuron, h, available_mechanisms, morphology_name)
    if record_membrane_state:
        ionic_currents, gating_variables, conductances = _extract_membrane_traces(
            hh_vectors,
            passive_vectors,
            voltage_trace,
            profile,
            recorded_len,
            python_ih_controller=python_ih_controller,
            python_ka_controller=python_ka_controller,
            python_calva_controller=python_calva_controller,
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
    available_mechanisms = _effective_optional_mechanisms(_ensure_optional_mechanisms_loaded())

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
    steady_state_snapshots = {
        neuron_spec.neuron_id: _steady_state_initialization_snapshot(
            neuron_spec.neuron,
            neuron_spec.morphology_name,
            available_mechanisms,
        )
        for neuron_spec in neurons
    }

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        sim.create(netParams=net_params, simConfig=sim_config, output=True, clearAll=True)
        cell_by_id = {str(cell.tags["pop"]): cell for cell in sim.net.cells}
        python_ih_controllers: list[PythonIhController] = []
        python_ka_controllers: list[PythonKAController] = []
        python_calva_controllers: list[PythonCaLVAController] = []
        selected_python_ih_controller: PythonIhController | None = None
        selected_python_ka_controller: PythonKAController | None = None
        selected_python_calva_controller: PythonCaLVAController | None = None
        for neuron_spec in neurons:
            cell = cell_by_id[neuron_spec.neuron_id]
            _apply_section_reversal_potentials(cell, neuron_spec.neuron)
            _apply_region_biophysics(cell, neuron_spec.neuron, h, available_mechanisms)

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
        if "Ih" not in available_mechanisms:
            for neuron_spec in neurons:
                cell = cell_by_id[neuron_spec.neuron_id]
                if neuron_spec.neuron_id == selected_neuron_id:
                    controller = _build_python_ih_controller(cell, neuron_spec.neuron, h, record_sec, record_segment)
                    selected_python_ih_controller = controller
                else:
                    controller = _build_python_ih_controller(cell, neuron_spec.neuron, h)
                if controller is not None:
                    python_ih_controllers.append(controller)
        if "KA" not in available_mechanisms:
            for neuron_spec in neurons:
                cell = cell_by_id[neuron_spec.neuron_id]
                if neuron_spec.neuron_id == selected_neuron_id:
                    controller = _build_python_ka_controller(cell, neuron_spec.neuron, h, record_sec, record_segment)
                    selected_python_ka_controller = controller
                else:
                    controller = _build_python_ka_controller(cell, neuron_spec.neuron, h)
                if controller is not None:
                    python_ka_controllers.append(controller)
        if "Ca_LVAst" not in available_mechanisms:
            for neuron_spec in neurons:
                cell = cell_by_id[neuron_spec.neuron_id]
                if neuron_spec.neuron_id == selected_neuron_id:
                    controller = _build_python_calva_controller(cell, h, record_sec, record_segment)
                    selected_python_calva_controller = controller
                else:
                    controller = _build_python_calva_controller(cell, h)
                if controller is not None:
                    python_calva_controllers.append(controller)

        def controller_by_neuron_id(controllers):
            lookup = {}
            for controller in controllers:
                if not controller.segments:
                    continue
                neuron_id = str(controller.segments[0].segment.sec.cell().tags["pop"])
                lookup[neuron_id] = controller
            return lookup

        ih_controller_lookup = controller_by_neuron_id(python_ih_controllers)
        ka_controller_lookup = controller_by_neuron_id(python_ka_controllers)
        calva_controller_lookup = controller_by_neuron_id(python_calva_controllers)
        ih_controller_by_id = {
            neuron_spec.neuron_id: ih_controller_lookup.get(neuron_spec.neuron_id)
            for neuron_spec in neurons
        }
        ka_controller_by_id = {
            neuron_spec.neuron_id: ka_controller_lookup.get(neuron_spec.neuron_id)
            for neuron_spec in neurons
        }
        calva_controller_by_id = {
            neuron_spec.neuron_id: calva_controller_lookup.get(neuron_spec.neuron_id)
            for neuron_spec in neurons
        }

        def initialize_state() -> None:
            for neuron_spec in neurons:
                _apply_steady_state_initialization_snapshot(
                    cell_by_id[neuron_spec.neuron_id],
                    steady_state_snapshots[neuron_spec.neuron_id],
                    python_ih_controller=ih_controller_by_id[neuron_spec.neuron_id],
                    python_ka_controller=ka_controller_by_id[neuron_spec.neuron_id],
                    python_calva_controller=calva_controller_by_id[neuron_spec.neuron_id],
                )

        _simulate_with_python_currents(
            h,
            master_neuron.duration_ms,
            master_neuron.dt_ms,
            master_neuron.v_rest_mV,
            python_ih_controllers,
            python_ka_controllers,
            python_calva_controllers,
            initialize_state=initialize_state,
        )

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
    profile = _segment_profile_at_site(
        selected_cell,
        effective_recording_site,
        selected_spec.neuron,
        h,
        available_mechanisms,
        selected_spec.morphology_name,
    )
    if record_membrane_state:
        ionic_currents, gating_variables, conductances = _extract_membrane_traces(
            hh_vectors,
            passive_vectors,
            voltage_trace,
            profile,
            recorded_len,
            python_ih_controller=selected_python_ih_controller,
            python_ka_controller=selected_python_ka_controller,
            python_calva_controller=selected_python_calva_controller,
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


def _fi_process_pool_enabled() -> bool:
    value = os.environ.get(FI_PROCESS_POOL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


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
    use_process_pool = (
        _fi_process_pool_enabled()
        and len(current_steps) >= FI_PARALLEL_MIN_SWEEP_POINTS
        and max_workers > 1
    )
    if use_process_pool:
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
                    label="Example current pulse",
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
