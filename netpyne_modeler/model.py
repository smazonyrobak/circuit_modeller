from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


SCHEMA_VERSION = 1


ROLE_LIBRARY = {
    "thalamic_relay": {
        "label": "Thalamic Relay",
        "sign": "excitatory",
        "representation": "compartmental",
        "size": 20,
    },
    "trn": {
        "label": "TRN",
        "sign": "inhibitory",
        "representation": "compartmental",
        "size": 20,
    },
    "cortex_l4_exc": {
        "label": "Cortex L4 Exc",
        "sign": "excitatory",
        "representation": "compartmental",
        "size": 80,
    },
    "cortex_l6_ct": {
        "label": "Cortex L6 CT",
        "sign": "excitatory",
        "representation": "compartmental",
        "size": 60,
    },
    "cortex_inhibitory": {
        "label": "Cortex Inhibitory",
        "sign": "inhibitory",
        "representation": "compartmental",
        "size": 30,
    },
    "generic_exc": {
        "label": "Generic Exc",
        "sign": "excitatory",
        "representation": "point",
        "size": 20,
    },
    "generic_inh": {
        "label": "Generic Inh",
        "sign": "inhibitory",
        "representation": "point",
        "size": 20,
    },
}


@dataclass(slots=True)
class Electrophysiology:
    v_init: float = -67.0
    cm: float = 1.0
    ra: float = 150.0
    soma_diam: float = 18.0
    soma_L: float = 18.0
    gnabar_hh: float = 0.12
    gkbar_hh: float = 0.036
    gl_hh: float = 0.0003
    el_hh: float = -54.3
    point_model: str = "IntFire1"
    point_params: dict[str, float] = field(
        default_factory=lambda: {
            "tau": 10.0,
            "refrac": 5.0,
            "taum": 10.0,
            "taus": 20.0,
            "ib": 0.0,
        }
    )


@dataclass(slots=True)
class STDPSettings:
    enabled: bool = False
    objective: str = "decorrelation"
    causal_gain: float = 0.01
    noncausal_penalty: float = 0.012
    timing_window_ms: float = 12.0
    max_weight: float = 0.05
    scaling_mode: str = "inverse_pot_proportional_dep"
    notes: str = ""


@dataclass(slots=True)
class Population:
    id: str
    label: str
    role: str
    sign: str
    representation: str
    size: int
    x: float
    y: float
    electrophysiology: Electrophysiology = field(default_factory=Electrophysiology)
    notes: str = ""


@dataclass(slots=True)
class Connection:
    id: str
    label: str
    pre: str
    post: str
    synapse: str = "AMPA"
    weight: float = 0.002
    delay: float = 2.0
    probability: float = 1.0
    stdp: STDPSettings = field(default_factory=STDPSettings)
    notes: str = ""


@dataclass(slots=True)
class ProjectMetadata:
    title: str = "Untitled thalamo-cortical model"
    description: str = ""
    schema_version: int = SCHEMA_VERSION


@dataclass(slots=True)
class ProjectModel:
    metadata: ProjectMetadata = field(default_factory=ProjectMetadata)
    populations: list[Population] = field(default_factory=list)
    connections: list[Connection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectModel":
        metadata = ProjectMetadata(**data.get("metadata", {}))
        populations = [
            Population(
                id=item["id"],
                label=item["label"],
                role=item["role"],
                sign=item["sign"],
                representation=item["representation"],
                size=int(item["size"]),
                x=float(item["x"]),
                y=float(item["y"]),
                electrophysiology=Electrophysiology(**item.get("electrophysiology", {})),
                notes=item.get("notes", ""),
            )
            for item in data.get("populations", [])
        ]
        connections = [
            Connection(
                id=item["id"],
                label=item["label"],
                pre=item["pre"],
                post=item["post"],
                synapse=item.get("synapse", "AMPA"),
                weight=float(item.get("weight", 0.002)),
                delay=float(item.get("delay", 2.0)),
                probability=float(item.get("probability", 1.0)),
                stdp=STDPSettings(**item.get("stdp", {})),
                notes=item.get("notes", ""),
            )
            for item in data.get("connections", [])
        ]
        return cls(metadata=metadata, populations=populations, connections=connections)

    @classmethod
    def load(cls, path: str | Path) -> "ProjectModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2),
            encoding="utf-8",
        )

    def population_by_id(self, population_id: str) -> Population | None:
        return next((item for item in self.populations if item.id == population_id), None)

    def connection_by_id(self, connection_id: str) -> Connection | None:
        return next((item for item in self.connections if item.id == connection_id), None)

    def next_id(self, prefix: str) -> str:
        existing = {
            *(item.id for item in self.populations),
            *(item.id for item in self.connections),
        }
        index = 1
        while f"{prefix}_{index}" in existing:
            index += 1
        return f"{prefix}_{index}"

    def make_population(
        self,
        role: str,
        x: float,
        y: float,
        label: str | None = None,
    ) -> Population:
        template = ROLE_LIBRARY[role]
        prefix = role.replace("cortex_", "").replace("generic_", "")
        population = Population(
            id=self.next_id(prefix),
            label=label or template["label"],
            role=role,
            sign=template["sign"],
            representation=template["representation"],
            size=template["size"],
            x=x,
            y=y,
        )
        if role == "trn":
            population.electrophysiology.v_init = -70.0
            population.electrophysiology.el_hh = -65.0
            population.electrophysiology.gkbar_hh = 0.042
        elif role == "thalamic_relay":
            population.electrophysiology.v_init = -63.0
            population.electrophysiology.soma_diam = 20.0
            population.electrophysiology.soma_L = 24.0
        elif role == "cortex_l6_ct":
            population.electrophysiology.soma_diam = 22.0
            population.electrophysiology.soma_L = 24.0
        return population

    def add_population(self, population: Population) -> None:
        self.populations.append(population)

    def add_connection(self, connection: Connection) -> None:
        self.connections.append(connection)

    def make_connection(
        self,
        pre: str,
        post: str,
        label: str | None = None,
        synapse: str = "AMPA",
    ) -> Connection:
        pre_label = self.population_by_id(pre).label if self.population_by_id(pre) else pre
        post_label = self.population_by_id(post).label if self.population_by_id(post) else post
        return Connection(
            id=self.next_id("conn"),
            label=label or f"{pre_label} -> {post_label}",
            pre=pre,
            post=post,
            synapse=synapse,
        )

    def remove_population(self, population_id: str) -> None:
        self.populations = [item for item in self.populations if item.id != population_id]
        self.connections = [
            item
            for item in self.connections
            if item.pre != population_id and item.post != population_id
        ]

    def remove_connection(self, connection_id: str) -> None:
        self.connections = [item for item in self.connections if item.id != connection_id]
