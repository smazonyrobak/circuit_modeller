TITLE A-type potassium current

NEURON {
    SUFFIX KA
    USEION k READ ek WRITE ik
    RANGE gKAbar, gKA, ika, m, h, minf, hinf, mtau, htau
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gKAbar = 0.001 (S/cm2)
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    ika (mA/cm2)
    gKA (S/cm2)
    minf
    hinf
    mtau (ms)
    htau (ms)
}

STATE {
    m
    h
}

INITIAL {
    rates(v)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gKA = gKAbar * m^4 * h
    ika = gKA * (v - ek)
    ik = ika
}

DERIVATIVE states {
    rates(v)
    m' = (minf - m) / mtau
    h' = (hinf - h) / htau
}

PROCEDURE rates(v (mV)) {
    minf = 1 / (1 + exp(-(v + 30) / 8.5))
    hinf = 1 / (1 + exp((v + 58) / 6))
    mtau = 0.2 + 2 / (1 + exp((v + 25) / 10))
    htau = 5 + 20 / (1 + exp((v + 50) / 8))
}
