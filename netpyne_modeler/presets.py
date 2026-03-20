from __future__ import annotations

from .model import ProjectMetadata, ProjectModel


def build_thalamocortical_preset() -> ProjectModel:
    project = ProjectModel(
        metadata=ProjectMetadata(
            title="TRN decorrelation motif",
            description=(
                "Minimal thalamo-cortical loop with relay, TRN, layer 4 cortex, and "
                "layer 6 corticothalamic feedback. Intended as a scaffold for causal "
                "versus non-causal STDP experiments and novelty gating hypotheses."
            ),
        )
    )

    tc = project.make_population("thalamic_relay", 80, 260, "VB relay")
    trn = project.make_population("trn", 360, 90, "TRN gate")
    l4 = project.make_population("cortex_l4_exc", 700, 260, "S1 L4")
    l6 = project.make_population("cortex_l6_ct", 700, 520, "S1 L6 CT")
    inh = project.make_population("cortex_inhibitory", 1020, 260, "Cortical IN")

    for population in [tc, trn, l4, l6, inh]:
        project.add_population(population)

    c1 = project.make_connection(tc.id, l4.id, synapse="AMPA")
    c1.weight = 0.004
    c1.delay = 2.0

    c2 = project.make_connection(tc.id, trn.id, label="Relay collateral -> TRN", synapse="AMPA")
    c2.weight = 0.0035
    c2.delay = 1.5
    c2.stdp.enabled = True
    c2.stdp.objective = "decorrelation"
    c2.stdp.causal_gain = 0.012
    c2.stdp.noncausal_penalty = 0.014
    c2.stdp.timing_window_ms = 10.0
    c2.stdp.max_weight = 0.06

    c3 = project.make_connection(l6.id, trn.id, label="CT feedback -> TRN", synapse="NMDA")
    c3.weight = 0.003
    c3.delay = 4.0
    c3.stdp.enabled = True
    c3.stdp.objective = "state-conditioned gating"
    c3.stdp.causal_gain = 0.01
    c3.stdp.noncausal_penalty = 0.01
    c3.stdp.timing_window_ms = 12.0
    c3.stdp.max_weight = 0.05

    c4 = project.make_connection(trn.id, tc.id, label="TRN inhibition -> relay", synapse="GABA_A")
    c4.weight = 0.005
    c4.delay = 2.5

    c5 = project.make_connection(l4.id, l6.id, label="L4 -> L6 recurrent", synapse="AMPA")
    c5.weight = 0.0025
    c5.delay = 2.0

    c6 = project.make_connection(l6.id, tc.id, label="Layer 6 -> relay", synapse="NMDA")
    c6.weight = 0.002
    c6.delay = 5.0

    c7 = project.make_connection(l4.id, inh.id, label="L4 -> interneuron", synapse="AMPA")
    c7.weight = 0.002
    c7.delay = 1.5

    c8 = project.make_connection(inh.id, l4.id, label="Interneuron -> L4", synapse="GABA_A")
    c8.weight = 0.003
    c8.delay = 1.0

    for connection in [c1, c2, c3, c4, c5, c6, c7, c8]:
        project.add_connection(connection)

    return project
