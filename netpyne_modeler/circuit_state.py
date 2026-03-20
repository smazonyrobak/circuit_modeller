from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .simulator import (
    CURRENT_CLAMP,
    FISweepConfig,
    MorphologySite,
    NeuronConfig,
    VOLTAGE_CLAMP,
    VoltagePulseTrain,
    default_recording_site,
    default_setup,
    list_available_swc_files,
)


DEFAULT_CONNECTION_CURRENT_NA = 1.0
DEFAULT_CONNECTION_PULSE_MS = 1.5


def _default_morphology_name() -> str:
    swc_files = list_available_swc_files()
    return swc_files[0] if swc_files else "pyramidal_neuron.swc"


@dataclass(slots=True)
class CircuitConnection:
    id: str
    source_id: str
    target_id: str
    label: str
    target_site: MorphologySite
    current_nA: float = DEFAULT_CONNECTION_CURRENT_NA
    pulse_width_ms: float = DEFAULT_CONNECTION_PULSE_MS
    delay_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "target_site": asdict(self.target_site),
            "current_nA": self.current_nA,
            "pulse_width_ms": self.pulse_width_ms,
            "delay_ms": self.delay_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitConnection":
        return cls(
            id=str(data["id"]),
            source_id=str(data["source_id"]),
            target_id=str(data["target_id"]),
            label=str(data.get("label") or f"{data['source_id']} -> {data['target_id']}"),
            target_site=MorphologySite(**data.get("target_site", {})),
            current_nA=float(data.get("current_nA", DEFAULT_CONNECTION_CURRENT_NA)),
            pulse_width_ms=float(data.get("pulse_width_ms", DEFAULT_CONNECTION_PULSE_MS)),
            delay_ms=float(data.get("delay_ms", 0.0)),
        )


@dataclass(slots=True)
class CircuitNeuron:
    id: str
    label: str
    color: str
    x: float
    y: float
    morphology_name: str
    neuron_config: NeuronConfig = field(default_factory=NeuronConfig)
    pulse_trains: list[VoltagePulseTrain] = field(default_factory=list)
    voltage_trains: list[VoltagePulseTrain] = field(default_factory=list)
    fi_config: FISweepConfig = field(default_factory=FISweepConfig)
    recording_source_mode: str = "patch"
    recording_site: MorphologySite | None = None
    fi_site: MorphologySite | None = None
    output_site: MorphologySite | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "morphology_name": self.morphology_name,
            "neuron_config": asdict(self.neuron_config),
            "pulse_trains": [asdict(train) for train in self.pulse_trains],
            "voltage_trains": [asdict(train) for train in self.voltage_trains],
            "fi_config": asdict(self.fi_config),
            "recording_source_mode": self.recording_source_mode,
            "recording_site": asdict(self.recording_site) if self.recording_site is not None else None,
            "fi_site": asdict(self.fi_site) if self.fi_site is not None else None,
            "output_site": asdict(self.output_site) if self.output_site is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitNeuron":
        return cls(
            id=str(data["id"]),
            label=str(data.get("label") or data["id"]),
            color=str(data.get("color") or "#2563eb"),
            x=float(data.get("x", 0.0)),
            y=float(data.get("y", 0.0)),
            morphology_name=str(data.get("morphology_name") or _default_morphology_name()),
            neuron_config=NeuronConfig(**data.get("neuron_config", {})),
            pulse_trains=[VoltagePulseTrain(**item) for item in data.get("pulse_trains", [])],
            voltage_trains=[VoltagePulseTrain(**item) for item in data.get("voltage_trains", [])],
            fi_config=FISweepConfig(**data.get("fi_config", {})),
            recording_source_mode=str(data.get("recording_source_mode") or "patch"),
            recording_site=(MorphologySite(**data["recording_site"]) if data.get("recording_site") else None),
            fi_site=(MorphologySite(**data["fi_site"]) if data.get("fi_site") else None),
            output_site=(MorphologySite(**data["output_site"]) if data.get("output_site") else None),
        )


@dataclass(slots=True)
class CircuitProject:
    neurons: list[CircuitNeuron] = field(default_factory=list)
    connections: list[CircuitConnection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "neurons": [neuron.to_dict() for neuron in self.neurons],
            "connections": [connection.to_dict() for connection in self.connections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CircuitProject":
        return cls(
            neurons=[CircuitNeuron.from_dict(item) for item in data.get("neurons", [])],
            connections=[CircuitConnection.from_dict(item) for item in data.get("connections", [])],
        )

    @classmethod
    def default(cls) -> "CircuitProject":
        morphology_name = _default_morphology_name()
        neuron_config, _ = default_setup(CURRENT_CLAMP)
        first = CircuitNeuron(
            id="neuron_1",
            label="Neuron 1",
            color="#2563eb",
            x=120.0,
            y=120.0,
            morphology_name=morphology_name,
            neuron_config=neuron_config,
            pulse_trains=[],
            voltage_trains=[],
            fi_config=FISweepConfig(),
            recording_source_mode="patch",
            recording_site=None,
            fi_site=None,
            output_site=None,
        )
        return cls(neurons=[first], connections=[])

    def neuron_by_id(self, neuron_id: str) -> CircuitNeuron | None:
        return next((neuron for neuron in self.neurons if neuron.id == neuron_id), None)

    def connection_by_id(self, connection_id: str) -> CircuitConnection | None:
        return next((connection for connection in self.connections if connection.id == connection_id), None)

    def next_id(self, prefix: str) -> str:
        existing = {item.id for item in self.neurons} | {item.id for item in self.connections}
        index = 1
        while f"{prefix}_{index}" in existing:
            index += 1
        return f"{prefix}_{index}"

    def add_neuron(
        self,
        label: str,
        color: str,
        x: float,
        y: float,
        morphology_name: str | None = None,
    ) -> CircuitNeuron:
        selected_morphology = morphology_name or _default_morphology_name()
        neuron_config, _ = default_setup(CURRENT_CLAMP)
        neuron = CircuitNeuron(
            id=self.next_id("neuron"),
            label=label,
            color=color,
            x=x,
            y=y,
            morphology_name=selected_morphology,
            neuron_config=neuron_config,
            pulse_trains=[],
            voltage_trains=[],
            fi_config=FISweepConfig(),
            recording_source_mode="patch",
            recording_site=None,
            fi_site=None,
            output_site=None,
        )
        self.neurons.append(neuron)
        return neuron

    def add_connection(self, source_id: str, target_id: str) -> CircuitConnection:
        target = self.neuron_by_id(target_id)
        if target is None:
            raise ValueError(f"Unknown target neuron: {target_id}")
        target_site = default_recording_site(target.morphology_name)
        connection = CircuitConnection(
            id=self.next_id("connection"),
            source_id=source_id,
            target_id=target_id,
            label=f"{source_id} -> {target_id}",
            target_site=target_site,
        )
        self.connections.append(connection)
        return connection

    def remove_neuron(self, neuron_id: str) -> None:
        self.neurons = [neuron for neuron in self.neurons if neuron.id != neuron_id]
        self.connections = [
            connection
            for connection in self.connections
            if connection.source_id != neuron_id and connection.target_id != neuron_id
        ]

    def remove_connection(self, connection_id: str) -> None:
        self.connections = [connection for connection in self.connections if connection.id != connection_id]
