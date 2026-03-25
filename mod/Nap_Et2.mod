TITLE Persistent sodium current

NEURON {
    SUFFIX Nap_Et2
    USEION na READ ena WRITE ina
    RANGE gNap_Et2bar, gNap_Et2, ina_nap, m, minf, mtau
    GLOBAL vhalf, k, tau
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gNap_Et2bar = 0.0004 (S/cm2)
    vhalf = -55 (mV)
    k = 4.5 (mV)
    tau = 1 (ms)
}

ASSIGNED {
    v (mV)
    ena (mV)
    ina (mA/cm2)
    ina_nap (mA/cm2)
    gNap_Et2 (S/cm2)
    minf
    mtau (ms)
}

STATE {
    m
}

INITIAL {
    rates(v)
    m = minf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gNap_Et2 = gNap_Et2bar * m
    ina_nap = gNap_Et2 * (v - ena)
    ina = ina_nap
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v - vhalf) / k))
    mtau = tau
}
