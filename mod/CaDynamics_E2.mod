TITLE Simple calcium concentration dynamics

NEURON {
    SUFFIX CaDynamics_E2
    USEION ca READ ica,cai WRITE cai VALENCE 2
    RANGE gamma, decay, minCai
}

UNITS {
    (mA) = (milliamp)
    (mM) = (milli/liter)
}

PARAMETER {
    gamma = 0.00064
    decay = 40 (ms)
    minCai = 5e-5 (mM)
}

STATE {
    cai (mM)
}

ASSIGNED {
    ica (mA/cm2)
    drive_channel (mM/ms)
}

INITIAL {
    cai = minCai
}

BREAKPOINT {
    SOLVE state METHOD cnexp
}

DERIVATIVE state {
    drive_channel = -gamma * ica
    if (drive_channel < 0) {
        drive_channel = 0
    }
    cai' = drive_channel + (minCai - cai) / decay
}
